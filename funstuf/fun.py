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


class DiscreteLoss(torch.nn.Module):
    """Loss functions for discrete bottleneck"""
    def __init__(self, N_groups, K_categories, lambda_use=1.0, lambda_ent=0.05, H_min=1.0):
        super().__init__()
        self.N_groups = N_groups
        self.K_categories = K_categories
        self.lambda_use = lambda_use
        # NOTE: lambda_ent is now used to *minimize per-sample entropy* (encourage peaky codes).
        # The previous "entropy floor" regularizer encouraged HIGH entropy, which makes the
        # trivial uniform-distribution solution too easy.
        self.lambda_ent = lambda_ent
        # Backwards-compat: keep H_min in the signature so old configs don't break,
        # but it is no longer used (see entropy_loss()).
        self.H_min = H_min
        # Uniform distribution for usage regularization
        self.register_buffer("uniform", torch.full((K_categories,), 1.0 / K_categories))
    
    def invariance_loss(self, probs):
        """
        Encourage all views to have similar distributions
        probs: [N, V, N_groups, K_categories]
        """
        # Mean distribution across views
        mean_probs = probs.mean(dim=1, keepdim=True)  # [N, 1, N_groups, K_categories]
        
        # KL divergence from each view to mean
        # KL(mean || view) = sum(mean * log(mean / view))
        eps = 1e-8
        log_ratio = (mean_probs + eps).log() - (probs + eps).log()
        kl = (mean_probs * log_ratio).sum(dim=-1)  # [N, V, N_groups]
        return kl.mean()
    
    def usage_loss(self, probs):
        """
        Encourage uniform usage of categories across batch
        probs: [N, V, N_groups, K_categories]
        """
        # Marginal distribution: average over batch and views
        marginal = probs.mean(dim=(0, 1))  # [N_groups, K_categories]
        
        # KL divergence from marginal to uniform
        eps = 1e-8
        log_ratio = (marginal + eps).log() - (self.uniform + eps).log()
        kl = (marginal * log_ratio).sum(dim=-1)  # [N_groups]
        return kl.mean()
    
    def entropy_loss(self, probs):
        """
        Encourage LOWER per-sample entropy (make codes commit / avoid uniform-per-sample).
        Usage regularization already pushes the *marginal* distribution toward uniform,
        so the combo (low conditional entropy, high marginal entropy) avoids collapse.
        probs: [N, V, N_groups, K_categories]
        """
        eps = 1e-8
        H = -(probs * (probs + eps).log()).sum(dim=-1)  # [N, V, N_groups]
        return H.mean()

    def marginal_entropy(self, probs):
        """
        Actual entropy of the batch/view marginal distribution (for logging).
        probs: [N, V, N_groups, K_categories]
        """
        eps = 1e-8
        marginal = probs.mean(dim=(0, 1))  # [N_groups, K_categories]
        Hm = -(marginal * (marginal + eps).log()).sum(dim=-1)  # [N_groups]
        return Hm.mean()
    
    def forward(self, probs):
        """
        Combined discrete loss
        probs: [N, V, N_groups, K_categories]
        """
        inv_loss = self.invariance_loss(probs)
        use_loss = self.usage_loss(probs)
        ent_loss = self.entropy_loss(probs)
        marg_ent = self.marginal_entropy(probs)
        
        total = inv_loss + self.lambda_use * use_loss + self.lambda_ent * ent_loss
        
        return total, {
            'inv': inv_loss.item(),
            'usage': use_loss.item(),
            'entropy': ent_loss.item(),
            'marginal_entropy': marg_ent.item(),
        }


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128, N_groups=8, K_categories=16, use_discrete=False):
        super().__init__()
        self.use_discrete = use_discrete
        self.N_groups = N_groups
        self.K_categories = K_categories
        
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=128,
        )
        
        if use_discrete:
            # Discrete bottleneck: outputs N groups of K-way categorical logits
            self.discrete_head = nn.Sequential(
                nn.Linear(512, 2048),
                nn.BatchNorm1d(2048),
                nn.GELU(),
                nn.Linear(2048, N_groups * K_categories)
            )
        else:
            # Original continuous projection
            self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x, temperature=1.0):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))  # [N*V, 512]
        
        if self.use_discrete:
            logits = self.discrete_head(emb)  # [N*V, N_groups * K_categories]
            logits = logits.reshape(-1, self.N_groups, self.K_categories)  # [N*V, N_groups, K_categories]
            # Return embeddings for probe, logits for discrete loss, and soft probs
            probs = F.softmax(logits / temperature, dim=-1)  # [N*V, N_groups, K_categories]
            # Reshape for multi-view: [N, V, N_groups, K_categories]
            logits_view = logits.reshape(N, V, self.N_groups, self.K_categories)
            probs_view = probs.reshape(N, V, self.N_groups, self.K_categories)
            return emb, logits_view, probs_view
        else:
            proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)
            return emb, proj


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


@hydra.main(version_base=None, config_path="../conf", config_name="config_continuous")
def main(cfg: DictConfig):
    # Set defaults for new discrete parameters
    use_discrete = cfg.get('use_discrete', False)
    N_groups = cfg.get('N_groups', 8)
    K_categories = cfg.get('K_categories', 16)
    temperature = cfg.get('temperature', 1.0)
    lambda_use = cfg.get('lambda_use', 1.0)
    lambda_ent = cfg.get('lambda_ent', 0.05)
    H_min = cfg.get('H_min', 1.0)
    
    wandb.init(project="LeJEPA", config=dict(cfg))
    torch.manual_seed(0)

    train_ds = HFDataset("train", V=cfg.V)
    test_ds = HFDataset("validation", V=1)
    train = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=8
    )
    test = DataLoader(test_ds, batch_size=256, num_workers=8)

    # modules and loss
    net = ViTEncoder(
        proj_dim=cfg.proj_dim, 
        N_groups=N_groups, 
        K_categories=K_categories,
        use_discrete=use_discrete
    ).to("cuda")
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).to("cuda")
    
    if use_discrete:
        discrete_loss_fn = DiscreteLoss(N_groups, K_categories, lambda_use, lambda_ent, H_min).to("cuda")
    else:
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
                    
                    if use_discrete:
                        # Discrete mode
                        emb, logits, probs = net(vs, temperature=temperature)
                        # probs: [N, V, N_groups, K_categories]
                        lejepa_loss, loss_dict = discrete_loss_fn(probs)
                        
                        # Compute category usage stats for monitoring
                        with torch.no_grad():
                            # Argmax per group to see which categories are used
                            codes = probs.argmax(dim=-1)  # [N, V, N_groups]
                            unique_per_group = [codes[..., i].unique().numel() for i in range(N_groups)]
                            avg_unique = sum(unique_per_group) / N_groups
                    else:
                        # Continuous mode
                        emb, proj = net(vs)
                        inv_loss = (proj.mean(0) - proj).square().mean()
                        sigreg_loss = sigreg(proj)
                        lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                        loss_dict = {
                            'sigreg': sigreg_loss.item(),
                            'inv': inv_loss.item()
                        }
                    
                    # Probe loss (same for both modes)
                    y_rep, yhat = y.repeat_interleave(cfg.V), probe(emb.detach())
                    probe_loss = F.cross_entropy(yhat, y_rep)
                    loss = lejepa_loss + probe_loss

                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                
                # Logging
                log_dict = {
                    "train/probe": probe_loss.item(),
                    "train/lejepa": lejepa_loss.item(),
                }
                if use_discrete:
                    log_dict.update({
                        "train/discrete_inv": loss_dict['inv'],
                        "train/discrete_usage": loss_dict['usage'],
                        # Actual entropies (not a ReLU floor penalty):
                        # - discrete_entropy: per-sample entropy (we minimize this with lambda_ent)
                        # - discrete_marginal_entropy: marginal entropy across batch/views (should be high)
                        "train/discrete_entropy": loss_dict['entropy'],
                        "train/discrete_marginal_entropy": loss_dict.get('marginal_entropy', 0.0),
                        "train/avg_unique_codes": avg_unique,
                    })
                else:
                    log_dict.update({
                        "train/sigreg": loss_dict['sigreg'],
                        "train/inv": loss_dict['inv'],
                    })
                wandb.log(log_dict)
                
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
                            if use_discrete:
                                emb = net(vs, temperature=temperature)[0]
                            else:
                                emb = net(vs)[0]
                            correct += (probe(emb).argmax(1) == y).sum().item()
                wandb.log({"test/acc": correct / len(test_ds), "test/epoch": epoch})
        
        # Save final model
        save_dir = Path("saved_models")
        save_dir.mkdir(exist_ok=True)
        model_name = "final_model_discrete.pt" if use_discrete else "final_model.pt"
        torch.save({
            'encoder': net.state_dict(),
            'probe': probe.state_dict(),
            'config': dict(cfg),
            'use_discrete': use_discrete,
            'N_groups': N_groups if use_discrete else None,
            'K_categories': K_categories if use_discrete else None,
        }, save_dir / model_name)
        print(f"\nFinal model saved to {save_dir / model_name}")
        
        wandb.finish()


if __name__ == "__main__":
    main()

