"""
Training script for Pix2Pix Manga Colorization

This script trains a Pix2Pix conditional GAN to colorize black-and-white
manga images using **paired** manga dataset (bw/color pages).

Training configuration:
- Resolution: 512x512 (for better text readability)
- Dataset: Manga-only (bw/color pairs)
- Epochs: 150

It implements the standard Pix2Pix training loop with:
- GAN loss (adversarial training)
- L1/SmoothL1 loss (pixel-level reconstruction)
- VGG perceptual loss (optional, for better generalization)

Handles logging, sample saving, and checkpointing.
"""

import os
import signal
import sys
import warnings
from pathlib import Path

# Suppress all PyTorch deprecation warnings BEFORE importing torch
warnings.simplefilter('ignore', FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, message='.*deprecated.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.*')
warnings.filterwarnings('ignore', message='.*torch.amp.*')

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

from dataset import PairedImageDataset
from networks import UNetGenerator, PatchGANDiscriminator
from utils import init_weights, save_image, set_seed, get_device
from vgg_loss import VGGPerceptualLoss
from lab_utils import (
    rgb_to_lab, lab_to_rgb, normalize_lab, denormalize_lab,
    extract_l_channel, extract_ab_channels, combine_l_ab
)


# ============================================================================#
# Hyperparameters
# ============================================================================#

# Data roots (paired datasets)
MANGA_SKETCH_DIR = "data/manga/bw"
MANGA_COLOR_DIR = "data/manga/color"

USE_BOTH_DATASETS = False  # Set to True to use multiple datasets

# Training hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 8  # Adjust based on available GPU memory
LR = 2e-4
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 50.0
LAMBDA_PERCEPTUAL = 12.0
LAMBDA_CHROMA = 2.0
USE_PERCEPTUAL_LOSS = True
USE_SMOOTH_L1 = True
USE_LAB_COLORSPACE = True
USE_CHROMA_LOSS = True

# Logging and output
CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"
LOG_INTERVAL = 1000
SAMPLE_INTERVAL = 5000

# Checkpoint management
KEEP_LAST_N_CHECKPOINTS = 3
SAVE_CHECKPOINT_INTERVAL = 5

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15

# Data loading
NUM_WORKERS = 4  # Adjust based on CPU cores
PIN_MEMORY = True

# Reproducibility
SEED = 42


def create_data_loaders():
    """
    Create data loaders for training and validation.

    Uses manga dataset(s) (bw/color pairs) for manga-specific colorization.
    Can combine old and new datasets if USE_BOTH_DATASETS is True.

    Returns:
        Tuple of (train_loader, val_loader) with 80/20 split.
    """
    datasets = []
    
    # Paired Manga Dataset
    try:
        dataset = PairedImageDataset(
            sketch_dir=MANGA_SKETCH_DIR,
            color_dir=MANGA_COLOR_DIR,
            augment=True,
        )
        datasets.append(dataset)
        print(f"Loaded dataset: {len(dataset)} pairs")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        if not USE_BOTH_DATASETS:
            raise

    # Combine datasets if using multiple (configured via constants)
    if len(datasets) > 1:
        manga_dataset = ConcatDataset(datasets)
    else:
        if not datasets:
            raise ValueError("No datasets loaded!")
        manga_dataset = datasets[0]
    
    # Split into train/val (80/20)
    total_size = len(manga_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(
        manga_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # DataLoaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=4,  # Prefetch 4 batches ahead (faster data loading)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        prefetch_factor=2,
    )

    print(f"Total samples: {total_size}")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    if len(datasets) > 1:
        print(f"  - Combined datasets: {len(datasets)}")
        for i, ds in enumerate(datasets):
            print(f"    Dataset {i+1}: {len(ds)} pairs")
    print(f"Batches per epoch (train): {len(train_loader)}")

    return train_loader, val_loader


def initialize_models(device: torch.device) -> tuple[UNetGenerator, PatchGANDiscriminator]:
    """
    Initialize Generator and Discriminator models.

    Args:
        device: Device to move models to.

    Returns:
        Tuple of (netG, netD) models.
    """
    # Generator - output 2 channels (AB) if using LAB, 3 channels (RGB) otherwise
    out_channels = 2 if USE_LAB_COLORSPACE else 3
    netG = UNetGenerator(in_channels=1, out_channels=out_channels, ngf=64).to(device)
    netG.apply(lambda m: init_weights(m, init_type="normal", init_gain=0.02))
    
    if USE_LAB_COLORSPACE:
        print("Using LAB color space: Model outputs 2 channels (AB)")
    else:
        print("Using RGB color space: Model outputs 3 channels (RGB)")

    # Discriminator (1 channel input + 3 channel target = 4)
    netD = PatchGANDiscriminator(in_channels=4, ndf=64).to(device)
    netD.apply(lambda m: init_weights(m, init_type="normal", init_gain=0.02))

    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

    return netG, netD


def validate(val_loader: DataLoader, netG: UNetGenerator, criterion_L1: torch.nn.Module, device: torch.device) -> float:
    """
    Validate the generator on validation set.
    
    Returns:
        Average L1 loss on validation set.
    """
    netG.eval()
    total_l1 = 0.0
    with torch.no_grad():
        for batch in val_loader:
            real_A = batch["A"].to(device)
            real_B_rgb = batch["B"].to(device)  # RGB
            
            fake_B = netG(real_A)
            
            # Convert to appropriate format for loss computation
            if USE_LAB_COLORSPACE:
                real_B_lab = rgb_to_lab(real_B_rgb)
                real_B_normalized = normalize_lab(real_B_lab)
                target_B = extract_ab_channels(real_B_normalized)
            else:
                target_B = real_B_rgb
            
            total_l1 += criterion_L1(fake_B, target_B).item()
    netG.train()
    return total_l1 / len(val_loader)


def train_one_epoch(
    epoch: int,
    train_loader: DataLoader,
    netG: UNetGenerator,
    netD: PatchGANDiscriminator,
    optimizer_G: torch.optim.Adam,
    optimizer_D: torch.optim.Adam,
    criterion_GAN: torch.nn.BCEWithLogitsLoss,
    criterion_L1: torch.nn.Module,
    criterion_perceptual: torch.nn.Module = None,
    device: torch.device = None,
    scaler: torch.amp.GradScaler = None,
    use_amp: bool = False,
) -> None:
    """
    Train for one epoch.

    Args:
        epoch: Current epoch number.
        train_loader: DataLoader for training data.
        netG: Generator model.
        netD: Discriminator model.
        optimizer_G: Generator optimizer.
        optimizer_D: Discriminator optimizer.
        criterion_GAN: GAN loss criterion.
        criterion_L1: L1 loss criterion.
        device: Device to run training on.
    """
    netG.train()
    netD.train()

    for iteration, batch in enumerate(train_loader):
        real_A = batch["A"].to(device)  # sketch / bw [B, 1, H, W]
        real_B_rgb = batch["B"].to(device)  # color RGB [B, 3, H, W]
        
        # Convert to LAB if using LAB color space
        if USE_LAB_COLORSPACE:
            real_B_lab = rgb_to_lab(real_B_rgb)  # [B, 3, H, W] - full LAB
            real_B_normalized = normalize_lab(real_B_lab)  # Normalized LAB for loss
            real_B_ab_normalized = extract_ab_channels(real_B_normalized)  # [B, 2, H, W] - normalized AB
            target_B = real_B_ab_normalized  # Target is normalized AB channels
        else:
            target_B = real_B_rgb  # Target is RGB

        # ============================================================
        # (a) Train Discriminator
        # ============================================================

        # Use autocast for mixed precision (new API)
        with torch.amp.autocast('cuda', enabled=use_amp):
            with torch.no_grad():
                fake_B = netG(real_A)

        # Convert fake output to RGB for discriminator (if using LAB)
        if USE_LAB_COLORSPACE:
            # Combine input L with predicted AB, convert to RGB
            fake_B_lab_normalized = combine_l_ab(real_A, fake_B)  # [B, 3, H, W] - normalized LAB
            fake_B_lab = denormalize_lab(fake_B_lab_normalized)  # Denormalize
            fake_B_rgb = lab_to_rgb(fake_B_lab)  # Convert to RGB for discriminator
        else:
            fake_B_rgb = fake_B
        
        # Real pair: (A, B)
        with torch.amp.autocast('cuda', enabled=use_amp):
            real_input = torch.cat([real_A, real_B_rgb], dim=1)  # [B, 4, H, W]
            pred_real = netD(real_input)
            target_real = torch.ones_like(pred_real) * 0.9  # Label smoothing: 0.9 instead of 1.0
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake pair: (A, G(A))
            fake_input = torch.cat([real_A, fake_B_rgb.detach()], dim=1)  # [B, 4, H, W]
            pred_fake = netD(fake_input)
            target_fake = torch.zeros_like(pred_fake) + 0.1  # Label smoothing: 0.1 instead of 0.0
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        optimizer_D.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss_D).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            scaler.step(optimizer_D)
            scaler.update()
        else:
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer_D.step()

        # ============================================================
        # (b) Train Generator
        # ============================================================

        # Generate fake_B in autocast
        with torch.amp.autocast('cuda', enabled=use_amp):
            fake_B = netG(real_A)  # [B, 2, H, W] if LAB, [B, 3, H, W] if RGB
        
        # Convert to RGB for discriminator and perceptual loss (outside autocast - LAB conversion uses CPU)
        if USE_LAB_COLORSPACE:
            # Combine L channel with predicted AB
            fake_B_lab_normalized = combine_l_ab(real_A, fake_B)  # [B, 3, H, W]
            fake_B_lab = denormalize_lab(fake_B_lab_normalized)
            fake_B_rgb = lab_to_rgb(fake_B_lab)  # [B, 3, H, W] - for discriminator/perceptual
        else:
            fake_B_rgb = fake_B

        # Discriminator and loss computation in autocast
        with torch.amp.autocast('cuda', enabled=use_amp):
            fake_input_for_G = torch.cat([real_A, fake_B_rgb], dim=1)
            pred_fake_for_G = netD(fake_input_for_G)
            target_real_for_G = torch.ones_like(pred_fake_for_G) * 0.9  # Label smoothing: 0.9

            loss_G_GAN = criterion_GAN(pred_fake_for_G, target_real_for_G)
            
            # Compute L1/SmoothL1 loss (on AB channels if LAB, RGB if RGB)
            loss_G_L1 = criterion_L1(fake_B, target_B) * LAMBDA_L1
            
            # Compute chroma (saturation) loss if using LAB - encourages vibrant colors
            loss_G_chroma = torch.tensor(0.0, device=device)
            if USE_CHROMA_LOSS and USE_LAB_COLORSPACE:
                # Chroma = sqrt(A^2 + B^2)
                ab2_fake = fake_B[:, 0]**2 + fake_B[:, 1]**2
                ab2_real = target_B[:, 0]**2 + target_B[:, 1]**2

                fake_chroma = torch.sqrt(torch.clamp(ab2_fake, min=0.0) + 1e-6)
                real_chroma = torch.sqrt(torch.clamp(ab2_real, min=0.0) + 1e-6)

                loss_G_chroma = torch.nn.functional.l1_loss(fake_chroma, real_chroma) * LAMBDA_CHROMA

            
            # Compute perceptual loss if enabled (always on RGB)
            loss_G_perceptual = torch.tensor(0.0, device=device)
            if USE_PERCEPTUAL_LOSS and criterion_perceptual is not None:
                loss_G_perceptual = criterion_perceptual(fake_B_rgb, real_B_rgb) * LAMBDA_PERCEPTUAL
        
        # Total generator loss
        loss_G = loss_G_GAN + loss_G_L1 + loss_G_chroma + loss_G_perceptual

        optimizer_G.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss_G).backward()
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            scaler.step(optimizer_G)
            scaler.update()
        else:
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer_G.step()

        # ============================================================
        # (c) Logging
        # ============================================================

        if (iteration + 1) % LOG_INTERVAL == 0:
            log_msg = (
                f"Epoch [{epoch}/{NUM_EPOCHS}] "
                f"Iteration [{iteration + 1}/{len(train_loader)}] "
                f"Loss_D: {loss_D.item():.4f} "
                f"Loss_G: {loss_G.item():.4f} "
                f"Loss_G_GAN: {loss_G_GAN.item():.4f} "
                f"Loss_G_L1: {loss_G_L1.item():.4f}"
            )
            if USE_CHROMA_LOSS and USE_LAB_COLORSPACE:
                log_msg += f" Loss_G_Chroma: {loss_G_chroma.item():.4f}"
            if USE_PERCEPTUAL_LOSS and criterion_perceptual is not None:
                log_msg += f" Loss_G_Perceptual: {loss_G_perceptual.item():.4f}"
            print(log_msg)

        # ============================================================
        # (d) Sample Saving
        # ============================================================

        if (iteration + 1) % SAMPLE_INTERVAL == 0:
            sample_real_A = real_A[0:1]   # [1, 1, H, W] - input BW
            sample_real_B_rgb = real_B_rgb[0:1]   # [1, 3, H, W] - ground truth color RGB
            
            # Convert fake output to RGB if using LAB
            if USE_LAB_COLORSPACE:
                sample_fake_B_ab = fake_B[0:1]  # [1, 2, H, W] - generated AB
                sample_fake_B_lab_norm = combine_l_ab(sample_real_A, sample_fake_B_ab)
                sample_fake_B_lab = denormalize_lab(sample_fake_B_lab_norm)
                sample_fake_B_rgb = lab_to_rgb(sample_fake_B_lab)  # [1, 3, H, W]
            else:
                sample_fake_B_rgb = fake_B[0:1]  # [1, 3, H, W]

            # Save input BW (convert to RGB for easier viewing by repeating channel)
            sample_real_A_rgb = sample_real_A.repeat(1, 3, 1, 1)  # [1, 3, H, W]
            save_image(
                sample_real_A_rgb,
                os.path.join(SAMPLE_DIR, f"epoch_{epoch}_iter_{iteration + 1}_real_A.png"),
            )
            save_image(
                sample_real_B_rgb,
                os.path.join(SAMPLE_DIR, f"epoch_{epoch}_iter_{iteration + 1}_real_B.png"),
            )
            save_image(
                sample_fake_B_rgb,
                os.path.join(SAMPLE_DIR, f"epoch_{epoch}_iter_{iteration + 1}_fake_B.png"),
            )

            print(f"Saved sample images at epoch {epoch}, iteration {iteration + 1} (input BW, ground truth, generated)")


# Global variables for emergency checkpoint saving
_emergency_save_vars = {
    'netG': None,
    'netD': None,
    'optimizer_G': None,
    'optimizer_D': None,
    'current_epoch': 0,
    'device': None,
}


def emergency_save_checkpoint(signum=None, frame=None):
    """
    Emergency checkpoint save on interruption (Ctrl+C).
    Saves current state so training can be resumed.
    """
    if _emergency_save_vars['netG'] is None:
        print("\nNo model to save. Exiting...")
        sys.exit(0)
    
    print("\n\n" + "=" * 60)
    print("INTERRUPTED! Saving emergency checkpoint...")
    print("=" * 60)
    
    try:
        emergency_path = os.path.join(CHECKPOINT_DIR, "pix2pix_emergency.pth")
        torch.save({
            "epoch": _emergency_save_vars['current_epoch'],
            "netG_state_dict": _emergency_save_vars['netG'].state_dict(),
            "netD_state_dict": _emergency_save_vars['netD'].state_dict(),
            "optimizer_G_state_dict": _emergency_save_vars['optimizer_G'].state_dict(),
            "optimizer_D_state_dict": _emergency_save_vars['optimizer_D'].state_dict(),
            "emergency": True,
        }, emergency_path)
        print(f"Emergency checkpoint saved: {emergency_path}")
        print("You can resume training by setting:")
        print(f'  CHECKPOINT_PATH = "{emergency_path}"')
        print("=" * 60)
    except Exception as e:
        print(f"ERROR: Could not save emergency checkpoint: {e}")
    
    sys.exit(0)


def cleanup_old_checkpoints(keep_last_n: int = 5) -> None:
    """
    Remove old checkpoints, keeping only the last N and the best checkpoint.
    
    Args:
        keep_last_n: Number of recent checkpoints to keep
    """
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return
    
    # Get all epoch checkpoints (not best checkpoint or emergency checkpoint)
    epoch_checkpoints = sorted(
        [cp for cp in checkpoint_dir.glob("pix2pix_epoch_*.pth") 
         if "best" not in cp.name and "emergency" not in cp.name],
        key=lambda x: int(x.stem.split("_")[-1]) if x.stem.split("_")[-1].isdigit() else 0
    )
    
    # Keep only the last N checkpoints
    if len(epoch_checkpoints) > keep_last_n:
        checkpoints_to_delete = epoch_checkpoints[:-keep_last_n]
        for cp in checkpoints_to_delete:
            try:
                cp.unlink()
                print(f"Deleted old checkpoint: {cp.name}")
            except Exception as e:
                print(f"Warning: Could not delete {cp.name}: {e}")


def save_checkpoint(
    epoch: int,
    netG: UNetGenerator,
    netD: PatchGANDiscriminator,
    optimizer_G: torch.optim.Adam,
    optimizer_D: torch.optim.Adam,
    save_always: bool = False,
) -> None:
    """
    Save training checkpoint.

    Args:
        epoch: Current epoch number.
        netG: Generator model.
        netD: Discriminator model.
        optimizer_G: Generator optimizer.
        optimizer_D: Discriminator optimizer.
        save_always: If True, save regardless of interval (for best checkpoint)
    """
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Only save at intervals (unless save_always is True)
    if not save_always and epoch % SAVE_CHECKPOINT_INTERVAL != 0:
        return

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"pix2pix_epoch_{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "netG_state_dict": netG.state_dict(),
            "netD_state_dict": netD.state_dict(),
            "optimizer_G_state_dict": optimizer_G.state_dict(),
            "optimizer_D_state_dict": optimizer_D.state_dict(),
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Cleanup old checkpoints (keep only last N)
    cleanup_old_checkpoints(KEEP_LAST_N_CHECKPOINTS)


def main() -> None:
    """Main training function."""
    set_seed(SEED)

    device = get_device()
    print(f"Using device: {device}")

    Path(SAMPLE_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Setting up data loaders...")
    print("=" * 60)
    train_loader, val_loader = create_data_loaders()

    print("\n" + "=" * 60)
    print("Initializing models...")
    print("=" * 60)
    netG, netD = initialize_models(device)
    
    # Enable cuDNN benchmark for faster training (input size is consistent: 512x512)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("✅ cuDNN benchmark enabled (faster convolutions)")
    
    # Compile models for fastest training (PyTorch 2.0+)
    # On Windows, use 'aot_eager' (Triton not available). On Linux, try 'inductor' first.
    USE_COMPILE = True
    if USE_COMPILE and hasattr(torch, 'compile'):
        import platform
        is_windows = platform.system() == 'Windows'
        
        if is_windows:
            # Windows: Use aot_eager directly (Triton not available on Windows)
            try:
                print("Compiling models with torch.compile() (Windows - using aot_eager backend)...")
                # aot_eager doesn't support mode='reduce-overhead', so we omit it
                netG = torch.compile(netG, backend='aot_eager')
                netD = torch.compile(netD, backend='aot_eager')
                print("✅ Models compiled with aot_eager backend - expect ~10-15% speedup!")
            except Exception as e:
                print(f"⚠️  torch.compile() failed: {e}")
                print("   Continuing without compilation (training will still work, just slower)")
        else:
            # Linux/Mac: Try inductor first (fastest), fallback to aot_eager
            try:
                print("Compiling models with torch.compile()...")
                try:
                    netG = torch.compile(netG, mode='reduce-overhead', backend='inductor')
                    netD = torch.compile(netD, mode='reduce-overhead', backend='inductor')
                    print("✅ Models compiled with inductor backend (fastest) - expect 20-30% speedup!")
                except (RuntimeError, ImportError, AttributeError) as e:
                    if 'triton' in str(e).lower() or 'inductor' in str(e).lower():
                        print("⚠️  Triton/inductor not available. Falling back to 'aot_eager' backend...")
                        # aot_eager doesn't support mode='reduce-overhead', so we omit it
                        netG = torch.compile(netG, backend='aot_eager')
                        netD = torch.compile(netD, backend='aot_eager')
                        print("✅ Models compiled with aot_eager backend - expect ~10-15% speedup!")
                    else:
                        raise
            except Exception as e:
                print(f"⚠️  torch.compile() failed: {e}")
                print("   Continuing without compilation (training will still work, just slower)")
    elif USE_COMPILE:
        print("⚠️  torch.compile() requires PyTorch 2.0+. Skipping compilation.")

    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    
    # Use SmoothL1 if enabled, otherwise L1
    if USE_SMOOTH_L1:
        criterion_L1 = torch.nn.SmoothL1Loss()
        print("Using SmoothL1 loss (more robust to outliers)")
    else:
        criterion_L1 = torch.nn.L1Loss()
        print("Using L1 loss")
    
    # Initialize VGG perceptual loss if enabled
    criterion_perceptual = None
    if USE_PERCEPTUAL_LOSS:
        print("Initializing VGG perceptual loss...")
        criterion_perceptual = VGGPerceptualLoss().to(device)
        criterion_perceptual.eval()  # VGG is always in eval mode
        print(f"VGG perceptual loss enabled (weight: {LAMBDA_PERCEPTUAL})")
    else:
        print("VGG perceptual loss disabled")

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))
    
    # Learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.95)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.95)
    
    # Mixed precision training (FP16)
    USE_AMP = True  # Enable mixed precision training
    scaler = None
    if USE_AMP and device.type == 'cuda':
        # Use new API: torch.amp.GradScaler('cuda', ...) instead of torch.cuda.amp.GradScaler()
        scaler = torch.amp.GradScaler('cuda')
        print("✅ Mixed precision training (FP16) enabled - expect ~2x speedup!")
    else:
        if USE_AMP and device.type != 'cuda':
            print("⚠️  Mixed precision requires CUDA. Disabling AMP.")
        USE_AMP = False

    RESUME = False  # Set to True to resume training from a checkpoint

    # Check for emergency checkpoint first, then regular checkpoint
    CHECKPOINT_PATH = "checkpoints/pix2pix_best.pth"
    EMERGENCY_CHECKPOINT = "checkpoints/pix2pix_emergency.pth"
    
    # Skip emergency checkpoint prompt (set to True to enable prompt)
    SKIP_EMERGENCY_CHECKPOINT = False
    
    # Prefer emergency checkpoint if it exists (means training was interrupted)
    if not SKIP_EMERGENCY_CHECKPOINT and os.path.exists(EMERGENCY_CHECKPOINT):
        print(f"Found emergency checkpoint: {EMERGENCY_CHECKPOINT}")
        print("This checkpoint was saved when training was interrupted.")
        use_emergency = input("Use emergency checkpoint? (y/n): ").lower().strip() == 'y'
        if use_emergency:
            CHECKPOINT_PATH = EMERGENCY_CHECKPOINT
    elif os.path.exists(EMERGENCY_CHECKPOINT):
        print(f"Emergency checkpoint found: {EMERGENCY_CHECKPOINT}")
        print("Skipping emergency checkpoint (using regular checkpoint path)")

    start_epoch = 1

    if RESUME and os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        netG.load_state_dict(checkpoint["netG_state_dict"])
        netD.load_state_dict(checkpoint["netD_state_dict"])
        
        # Load optimizer states if they exist (best checkpoints don't have them)
        if "optimizer_G_state_dict" in checkpoint and "optimizer_D_state_dict" in checkpoint:
            optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
            print("Loaded optimizer states")
        else:
            print("Warning: Optimizer states not found in checkpoint. Starting with fresh optimizers.")

        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print("Press Ctrl+C to stop and save emergency checkpoint")
    print("=" * 60)

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Early stopping info
    if USE_EARLY_STOPPING:
        print(f"Early stopping enabled (patience: {EARLY_STOPPING_PATIENCE} epochs)")
    else:
        print(f"Early stopping disabled - will train for full {NUM_EPOCHS} epochs")
    
    # Store references for emergency save
    _emergency_save_vars['netG'] = netG
    _emergency_save_vars['netD'] = netD
    _emergency_save_vars['optimizer_G'] = optimizer_G
    _emergency_save_vars['optimizer_D'] = optimizer_D
    _emergency_save_vars['device'] = device
    
    # Register signal handler for Ctrl+C (emergency checkpoint)
    try:
        signal.signal(signal.SIGINT, emergency_save_checkpoint)
        print("✅ Emergency checkpoint handler registered (Ctrl+C)")
    except (ValueError, OSError) as e:
        print(f"⚠️  Could not register signal handler: {e}")
        print("   Emergency checkpoint will still work via try/except")
    
    try:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            _emergency_save_vars['current_epoch'] = epoch
            
            train_one_epoch(
                epoch=epoch,
                train_loader=train_loader,
                netG=netG,
                netD=netD,
                optimizer_G=optimizer_G,
                optimizer_D=optimizer_D,
                criterion_GAN=criterion_GAN,
                criterion_L1=criterion_L1,
                criterion_perceptual=criterion_perceptual,
                device=device,
                scaler=scaler,
                use_amp=USE_AMP,
            )
            
            # Validate and save best checkpoint
            val_loss = validate(val_loader, netG, criterion_L1, device)
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Validation L1 Loss: {val_loss:.4f}")
            
            # Save checkpoint at intervals
            save_checkpoint(epoch, netG, netD, optimizer_G, optimizer_D, save_always=False)
            
            # Check for improvement
            improved = val_loss < best_val_loss
            
            if improved:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                
                # Save with epoch number in filename for clarity
                best_path = os.path.join(CHECKPOINT_DIR, f"pix2pix_best_epoch_{epoch}.pth")
                best_path_simple = os.path.join(CHECKPOINT_DIR, "pix2pix_best.pth")
                
                checkpoint_data = {
                    "epoch": epoch,
                    "netG_state_dict": netG.state_dict(),
                    "netD_state_dict": netD.state_dict(),
                    "val_loss": val_loss,
                }
                
                # Save with epoch number
                torch.save(checkpoint_data, best_path)
                # Also save as simple name for easy loading
                torch.save(checkpoint_data, best_path_simple)
                
                print(f"⭐ Saved BEST checkpoint: Epoch {epoch} with val_loss={val_loss:.4f}")
                print(f"   → {best_path}")
                print(f"   → {best_path_simple} (alias)")
            else:
                epochs_without_improvement += 1
                if USE_EARLY_STOPPING:
                    print(f"   No improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs")
            
            # Early stopping check
            if USE_EARLY_STOPPING and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("\n" + "=" * 60)
                print(f"🛑 EARLY STOPPING TRIGGERED")
                print("=" * 60)
                print(f"Validation loss has not improved for {EARLY_STOPPING_PATIENCE} epochs.")
                print(f"Best validation loss: {best_val_loss:.4f} (at epoch {best_epoch})")
                print(f"Stopping training at epoch {epoch}.")
                print(f"Best checkpoint saved: pix2pix_best.pth")
                print("=" * 60)
                break
            
            # Save last checkpoint (for resuming) - always save the most recent
            if epoch == NUM_EPOCHS or (epoch % SAVE_CHECKPOINT_INTERVAL == 0):
                last_path = os.path.join(CHECKPOINT_DIR, "pix2pix_last.pth")
                torch.save({
                    "epoch": epoch,
                    "netG_state_dict": netG.state_dict(),
                    "netD_state_dict": netD.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "val_loss": val_loss,
                }, last_path)
            
            # Update learning rates
            scheduler_G.step()
            scheduler_D.step()
            print(f"Learning rates: G={scheduler_G.get_last_lr()[0]:.6f}, D={scheduler_D.get_last_lr()[0]:.6f}")
    
    except KeyboardInterrupt:
        # This will be handled by signal handler, but just in case
        emergency_save_checkpoint()
    except Exception as e:
        print(f"\nError during training: {e}")
        emergency_save_checkpoint()
        raise
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()