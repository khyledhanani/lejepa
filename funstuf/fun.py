import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import wandb
import hydra
import tqdm
from omegaconf import DictConfig
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from pathlib import Path
import struct
import numpy as np
import math


def load_mnist_images(path):
    """Load MNIST-format image file (idx3-ubyte)"""
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images


def load_mnist_labels(path):
    """Load MNIST-format label file (idx1-ubyte)"""
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


class BlockMaskGenerator:
    """
    I-JEPA style block masking: creates contiguous rectangular target blocks.
    
    Unlike random masking, I-JEPA masks contiguous spatial regions which forces
    the model to learn meaningful spatial relationships.
    """
    def __init__(
        self,
        input_size=(16, 16),      # Grid size (patches per side)
        num_targets=4,             # Number of target blocks to predict
        target_scale=(0.15, 0.2),  # Target block size as fraction of image
        target_aspect_ratio=(0.75, 1.5),  # Aspect ratio range for target blocks
        context_scale=(0.85, 1.0), # How much context to keep
    ):
        self.input_size = input_size
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.context_scale = context_scale
        self.num_patches = input_size[0] * input_size[1]
    
    def _sample_block(self, scale, aspect_ratio_range):
        """Sample a random rectangular block."""
        H, W = self.input_size
        
        # Sample scale and aspect ratio
        s = np.random.uniform(*scale)
        ar = np.random.uniform(*aspect_ratio_range)
        
        # Compute block dimensions
        num_patches = int(self.num_patches * s)
        h = int(round(math.sqrt(num_patches / ar)))
        w = int(round(math.sqrt(num_patches * ar)))
        h = min(h, H)
        w = min(w, W)
        
        # Sample position
        top = np.random.randint(0, H - h + 1)
        left = np.random.randint(0, W - w + 1)
        
        return top, left, h, w
    
    def __call__(self):
        """
        Generate context and target masks.
        
        Returns:
            context_mask: Boolean tensor (num_patches,) - True for context patches
            target_masks: List of boolean tensors - True for each target block
            target_indices: List of patch indices for each target block
        """
        H, W = self.input_size
        
        # Start with all patches as potential context
        available = torch.ones(H, W, dtype=torch.bool)
        target_masks = []
        target_indices_list = []
        
        # Sample target blocks
        for _ in range(self.num_targets):
            top, left, h, w = self._sample_block(self.target_scale, self.target_aspect_ratio)
            
            # Create mask for this target block
            target_mask = torch.zeros(H, W, dtype=torch.bool)
            target_mask[top:top+h, left:left+w] = True
            
            # Get indices of target patches
            indices = torch.where(target_mask.flatten())[0]
            
            target_masks.append(target_mask.flatten())
            target_indices_list.append(indices)
            
            # Remove target from available context
            available[top:top+h, left:left+w] = False
        
        # Context = everything not in targets
        context_mask = available.flatten()
        
        return context_mask, target_masks, target_indices_list


class FashionMNISTDataset(Dataset):
    """
    Fashion MNIST dataset for I-JEPA.
    
    Returns just the image - masks are generated dynamically in the training loop.
    This matches I-JEPA's approach of generating new masks each iteration.
    """
    def __init__(self, data_dir, split="train"):
        self.split = split
        self.is_train = split == "train"
        data_dir = Path(data_dir)
        
        if split == "train":
            self.images = load_mnist_images(data_dir / "train-images-idx3-ubyte")
            self.labels = load_mnist_labels(data_dir / "train-labels-idx1-ubyte")
        else:
            self.images = load_mnist_images(data_dir / "t10k-images-idx3-ubyte")
            self.labels = load_mnist_labels(data_dir / "t10k-labels-idx1-ubyte")
        
        # Base transform: grayscale -> RGB, resize, normalize
        self.base_transform = v2.Compose([
            v2.ToImage(),
            v2.Grayscale(num_output_channels=3),
            v2.Resize(128),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Light augmentation for training
        self.train_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
        ])
    
    def __getitem__(self, i):
        img = self.images[i]
        label = int(self.labels[i])
        
        x = self.base_transform(img)
        if self.is_train:
            x = self.train_transform(x)
        
        return x, label
    
    def __len__(self):
        return len(self.labels)


class SIGReg(torch.nn.Module):
    """SIGReg loss for preventing representation collapse."""
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
        """
        Args:
            proj: Embeddings of shape (..., embed_dim) - will be flattened to (N, embed_dim)
        """
        # Flatten all but last dim for batch statistics
        proj = proj.reshape(-1, proj.size(-1))
        
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        statistic = (err @ self.weights) * proj.size(0)
        return statistic.mean()


class VisionTransformerEncoder(nn.Module):
    """
    Vision Transformer for I-JEPA that supports encoding subsets of patches.
    
    Key I-JEPA requirement: Context encoder must ONLY see context patches,
    not the full image. This prevents information leakage from target patches.
    
    Reference: https://arxiv.org/pdf/2301.08243 (Figure 3)
    """
    def __init__(self, img_size=128, patch_size=8, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 256 for 128/8
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size  # 16
        
        # Patch embedding: Conv2d is equivalent to linear projection of flattened patches
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable position embeddings (no CLS token in I-JEPA)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def patchify(self, x):
        """Convert image to patch embeddings with positions."""
        x = self.patch_embed(x)  # (N, embed_dim, grid, grid)
        x = x.flatten(2).transpose(1, 2)  # (N, num_patches, embed_dim)
        x = x + self.pos_embed
        return x
    
    def forward_patches(self, patches):
        """
        Forward pass on pre-selected patches (no patchification).
        Args:
            patches: (N, num_patches_subset, embed_dim) - already embedded patches with positions
        Returns:
            Encoded patches (N, num_patches_subset, embed_dim)
        """
        x = patches
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
    
    def forward(self, x):
        """
        Standard forward: encode all patches.
        Args:
            x: Images (N, C, H, W)
        Returns:
            All patch embeddings (N, num_patches, embed_dim)
        """
        x = self.patchify(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
    
    def get_pos_embed(self):
        """Return position embeddings for predictor."""
        return self.pos_embed.squeeze(0)  # (num_patches, embed_dim)


class IJEPAPredictor(nn.Module):
    """
    I-JEPA Predictor: predicts target patch representations from context.
    
    Uses cross-attention: target positions query context patch representations.
    This is the key component that makes I-JEPA work - it must predict
    representations of unseen patches from visible context.
    
    Paper notes (Table 12): Deeper predictor improves performance significantly.
    Paper notes (Table 14): Predictor width bottleneck helps.
    """
    def __init__(self, embed_dim=384, predictor_embed_dim=None, num_heads=6, depth=6):
        super().__init__()
        # Bottleneck in predictor width (Table 14 in paper)
        predictor_embed_dim = predictor_embed_dim or embed_dim
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        
        # Input projection (bottleneck)
        self.input_proj = nn.Linear(embed_dim, predictor_embed_dim)
        
        # Learnable mask token (represents "unknown" for target positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Predictor transformer (cross-attention from targets to context)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=predictor_embed_dim,
                nhead=num_heads,
                dim_feedforward=predictor_embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_embed_dim)
        
        # Output projection back to encoder dim
        self.output_proj = nn.Linear(predictor_embed_dim, embed_dim)
    
    def forward(self, context_emb, target_pos_tokens):
        """
        Predict target patch representations from context.
        
        Args:
            context_emb: Context patch embeddings (N, num_context, embed_dim)
            target_pos_tokens: Position tokens for targets (N, num_target, embed_dim)
        
        Returns:
            Predicted target embeddings (N, num_target, embed_dim)
        """
        N, num_target, _ = target_pos_tokens.shape
        
        # Project context to predictor dimension
        context = self.input_proj(context_emb)
        
        # Project target positions and add mask token
        target_queries = self.input_proj(target_pos_tokens) + self.mask_token.expand(N, num_target, -1)
        
        # Cross-attention: target queries attend to context
        x = target_queries
        for block in self.blocks:
            x = block(x, context)
        
        x = self.norm(x)
        x = self.output_proj(x)
        
        return x  # (N, num_target, embed_dim)


def ema_update(online_model, target_model, momentum=0.996):
    """
    Exponential Moving Average update for target encoder.
    This is how I-JEPA prevents collapse - the target encoder
    is a slowly-updating copy of the context encoder.
    """
    with torch.no_grad():
        for online_params, target_params in zip(online_model.parameters(), target_model.parameters()):
            target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """
    True I-JEPA training with SIGReg regularization.
    
    Faithful to paper (https://arxiv.org/pdf/2301.08243):
    1. Context encoder ONLY sees context patches (no information leakage)
    2. Target encoder (EMA) ONLY sees target patches
    3. Predictor predicts target representations from context
    4. Loss = patch-level prediction + SIGReg regularization
    
    Key difference from paper: We add SIGReg for additional collapse prevention
    alongside the EMA target encoder.
    """
    wandb.init(project="IJEPA-SIGReg-FashionMNIST", config=dict(cfg))
    torch.manual_seed(0)
    import copy

    # Dataset
    data_dir = Path("./datasets/fmnist")
    train_ds = FashionMNISTDataset(data_dir, split="train")
    test_ds = FashionMNISTDataset(data_dir, split="test")
    train = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=4)
    test = DataLoader(test_ds, batch_size=cfg.bs, num_workers=4)

    # Model components
    embed_dim = cfg.embed_dim
    
    # Context encoder (trainable) - only sees context patches
    context_encoder = VisionTransformerEncoder(
        img_size=128, patch_size=8, embed_dim=embed_dim, 
        depth=cfg.encoder_depth, num_heads=cfg.num_heads
    ).to("cuda")
    
    # Target encoder (EMA of context encoder) - only sees target patches
    # This is how I-JEPA prevents collapse (Section 2 of paper)
    target_encoder = copy.deepcopy(context_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False  # Target encoder is not trained directly
    
    # Predictor with bottleneck (Table 14 shows this helps)
    predictor = IJEPAPredictor(
        embed_dim=embed_dim, 
        predictor_embed_dim=embed_dim // 2,  # Bottleneck
        num_heads=cfg.num_heads, 
        depth=cfg.predictor_depth
    ).to("cuda")
    
    # Linear probe for evaluation
    probe = nn.Sequential(
        nn.LayerNorm(embed_dim), 
        nn.Linear(embed_dim, 10)
    ).to("cuda")
    
    sigreg = SIGReg().to("cuda")
    
    # Block mask generator (I-JEPA style)
    mask_gen = BlockMaskGenerator(
        input_size=(16, 16),  # 128/8 = 16 patches per side
        num_targets=cfg.num_targets,
        target_scale=(cfg.target_scale_min, cfg.target_scale_max),
    )
    
    # Optimizer (only context encoder and predictor are trained)
    g1 = {"params": context_encoder.parameters(), "lr": cfg.lr, "weight_decay": 0.05}
    g2 = {"params": predictor.parameters(), "lr": cfg.lr, "weight_decay": 0.05}
    g3 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-6}
    opt = torch.optim.AdamW([g1, g2, g3])
    
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    
    scaler = GradScaler(enabled=True)
    
    # EMA momentum schedule (paper uses 0.996 → 1.0)
    ema_momentum = cfg.get("ema_momentum", 0.996)
    
    # Training
    step = 0
    total_steps = len(train) * cfg.epochs
    with tqdm.tqdm(total=total_steps, desc="I-JEPA Training") as pbar:
        for epoch in range(cfg.epochs):
            context_encoder.train(), predictor.train(), probe.train()
            
            for x, y in train:
                x = x.to("cuda", non_blocking=True)  # (N, C, H, W)
                y = y.to("cuda", non_blocking=True)
                N = x.size(0)
                
                with autocast("cuda", dtype=torch.bfloat16):
                    # 1. Patchify image (shared between encoders)
                    # This gives us patch embeddings with position info
                    all_patches = context_encoder.patchify(x)  # (N, 256, embed_dim)
                    pos_embed = context_encoder.get_pos_embed().to("cuda")  # (256, embed_dim)
                    
                    # 2. Generate block masks
                    context_mask, target_masks, target_indices = mask_gen()
                    context_idx = torch.where(context_mask)[0].to("cuda")
                    
                    # 3. Context encoder: ONLY sees context patches (no target leakage!)
                    context_patches = all_patches[:, context_idx, :]  # (N, num_context, embed_dim)
                    context_emb = context_encoder.forward_patches(context_patches)
                    
                    # 4. For each target block, get target representations from EMA encoder
                    pred_loss = 0.0
                    for target_idx_cpu in target_indices:
                        target_idx = target_idx_cpu.to("cuda")
                        
                        # Target encoder: ONLY sees target patches
                        with torch.no_grad():
                            target_patches = all_patches[:, target_idx, :].detach()
                            target_emb = target_encoder.forward_patches(target_patches)
                        
                        # Get position tokens for targets (predictor needs to know WHERE to predict)
                        target_pos_tokens = pos_embed[target_idx].unsqueeze(0).expand(N, -1, -1)
                        
                        # Predictor: context → predicted target
                        predicted = predictor(context_emb, target_pos_tokens)
                        
                        # Smooth L1 loss (stop gradient on target - it's from EMA encoder)
                        pred_loss = pred_loss + F.smooth_l1_loss(predicted, target_emb)
                    
                    pred_loss = pred_loss / len(target_indices)
                    
                    # 5. SIGReg on context embeddings (additional collapse prevention)
                    sigreg_loss = sigreg(context_emb)
                    
                    # 6. Combined I-JEPA + SIGReg loss
                    jepa_loss = pred_loss + cfg.lamb * sigreg_loss
                    
                    # 7. Linear probe (on full image from context encoder for evaluation)
                    with torch.no_grad():
                        full_emb = context_encoder(x)
                        global_emb = full_emb.mean(dim=1)
                    yhat = probe(global_emb)
                    probe_loss = F.cross_entropy(yhat, y)
                    
                    loss = jepa_loss + probe_loss
                
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                
                # 8. EMA update of target encoder (key to I-JEPA stability)
                ema_update(context_encoder, target_encoder, momentum=ema_momentum)
                
                step += 1
                wandb.log({
                    "train/pred_loss": pred_loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/jepa": jepa_loss.item(),
                    "train/probe": probe_loss.item(),
                    "train/step": step,
                })
                pbar.update(1)
                pbar.set_postfix(epoch=epoch+1, pred=f"{pred_loss.item():.3f}")
            
            # Evaluation
            if epoch % 5 == 0 or epoch == cfg.epochs - 1:
                context_encoder.eval(), probe.eval()
                correct = 0
                with torch.inference_mode():
                    for x, y in tqdm.tqdm(test, desc="Eval", leave=False):
                        x = x.to("cuda", non_blocking=True)
                        y = y.to("cuda", non_blocking=True)
                        with autocast("cuda", dtype=torch.bfloat16):
                            patch_emb = context_encoder(x)
                            global_emb = patch_emb.mean(dim=1)
                            correct += (probe(global_emb).argmax(1) == y).sum().item()
                acc = correct / len(test_ds)
                wandb.log({"test/acc": acc, "epoch": epoch})
                pbar.set_postfix(epoch=epoch+1, acc=f"{acc:.3f}")
        
        # Save
        save_dir = Path("saved_models")
        save_dir.mkdir(exist_ok=True)
        torch.save({
            'context_encoder': context_encoder.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'probe': probe.state_dict(),
            'config': dict(cfg)
        }, save_dir / "ijepa_sigreg_model.pt")
        print(f"\nModel saved to {save_dir / 'ijepa_sigreg_model.pt'}")
        
        wandb.finish()


if __name__ == "__main__":
    main()

