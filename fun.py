import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.datasets import ImageFolder
import timm, wandb, hydra, tqdm
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
from pathlib import Path


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class HFDataset(torch.utils.data.Dataset):
    def __init__(self, split, V=1):
        self.V = V
        # Load from local imagenette-160 dataset
        data_dir = Path("./datasets/imagenette2-160")
        split_dir = "train" if split == "train" else "val"
        self.ds = ImageFolder(data_dir / split_dir)
        
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(128, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.test = v2.Compose(
            [
                v2.Resize(128),
                v2.CenterCrop(128),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        img, label = self.ds[i]
        transform = self.aug if self.V > 1 else self.test
        return torch.stack([transform(img) for _ in range(self.V)]), label

    def __len__(self):
        return len(self.ds)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    wandb.init(project="LeJEPA", config=dict(cfg))
    torch.manual_seed(0)

    train_ds = HFDataset("train", V=cfg.V)
    test_ds = HFDataset("validation", V=1)
    train = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=8
    )
    test = DataLoader(test_ds, batch_size=256, num_workers=8)

    # modules and loss
    net = ViTEncoder(proj_dim=cfg.proj_dim).to("cuda")
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).to("cuda")
    sigreg = SIGReg().to("cuda")
    # Optimizer and scheduler
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled="cuda" == "cuda")
    # Training
    total_steps = len(train) * cfg.epochs
    with tqdm.tqdm(total=total_steps, desc="Training") as pbar:
        for epoch in range(cfg.epochs):
            net.train(), probe.train()
            for vs, y in train:
                with autocast("cuda", dtype=torch.bfloat16):
                    vs = vs.to("cuda", non_blocking=True)
                    y = y.to("cuda", non_blocking=True)
                    emb, proj = net(vs)
                    inv_loss = (proj.mean(0) - proj).square().mean()
                    sigreg_loss = sigreg(proj)
                    lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                    y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
                    probe_loss = F.cross_entropy(yhat, y_rep)
                    loss = lejepa_loss + probe_loss

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                wandb.log(
                    {
                        "train/probe": probe_loss.item(),
                        "train/lejepa": lejepa_loss.item(),
                        "train/sigreg": sigreg_loss.item(),
                        "train/inv": inv_loss.item(),
                    }
                )
                pbar.update(1)
                pbar.set_postfix(epoch=epoch+1, loss=loss.item())

            # Evaluation - only every 10 epochs to speed up training
            if epoch % 10 == 0 or epoch == cfg.epochs - 1:
                net.eval(), probe.eval()
                correct = 0
                with torch.inference_mode():
                    for vs, y in tqdm.tqdm(test, desc="Validating", leave=False):
                        vs = vs.to("cuda", non_blocking=True)
                        y = y.to("cuda", non_blocking=True)
                        with autocast("cuda", dtype=torch.bfloat16):
                             correct += (probe(net(vs)[0]).argmax(1) == y).sum().item()
                wandb.log({"test/acc": correct / len(test_ds), "test/epoch": epoch})
        
        # Save final model
        save_dir = Path("saved_models")
        save_dir.mkdir(exist_ok=True)
        torch.save({
            'encoder': net.state_dict(),
            'probe': probe.state_dict(),
            'config': dict(cfg)
        }, save_dir / "final_model.pt")
        print(f"\nFinal model saved to {save_dir / 'final_model.pt'}")
        
        wandb.finish()


if __name__ == "__main__":
    main()

