"""
Training script for Pix2Pix Manga Colorization

This script trains a Pix2Pix conditional GAN to colorize black-and-white
manga images using **paired** manga dataset (bw/color pages).

Training configuration:
- Resolution: 512x512 (for better text readability)
- Dataset: Manga-only (bw/color pairs)
- Epochs: 150

It implements the standard Pix2Pix training loop with GAN loss and L1 loss,
handles logging, sample saving, and checkpointing.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

from dataset import PairedImageDataset
from networks import UNetGenerator, PatchGANDiscriminator
from utils import init_weights, save_image, set_seed, get_device


# ============================================================================#
# Hyperparameters
# ============================================================================#

# Data roots (paired datasets)
# Old dataset (12k matched pairs)
OLD_MANGA_SKETCH_DIR = "data/manga_dataset/bw"
OLD_MANGA_COLOR_DIR = "data/manga_dataset/color"

# New dataset (colored_manga - make sure B/W conversion is complete!)
NEW_MANGA_SKETCH_DIR = "data/colored_manga/bw"
NEW_MANGA_COLOR_DIR = "data/colored_manga/color_full"

USE_BOTH_DATASETS = True

# Training hyperparameters
NUM_EPOCHS = 150
BATCH_SIZE = 2  # Reduced from 4 for 512x512 resolution (4x memory usage)
LR = 2e-4
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 100.0  # Increased for better detail preservation at 512x512

# Logging and output
CHECKPOINT_DIR = "checkpoints"
SAMPLE_DIR = "samples"
LOG_INTERVAL = 100  # iterations
SAMPLE_INTERVAL = 500  # iterations

# Data loading
NUM_WORKERS = 4
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
    
    # Old dataset (12k matched pairs)
    if USE_BOTH_DATASETS:
        try:
            old_dataset = PairedImageDataset(
                sketch_dir=OLD_MANGA_SKETCH_DIR,
                color_dir=OLD_MANGA_COLOR_DIR,
                augment=True,
            )
            datasets.append(old_dataset)
            print(f"Loaded old dataset: {len(old_dataset)} pairs")
        except Exception as e:
            print(f"Warning: Could not load old dataset: {e}")
    
    # New dataset (colored_manga)
    try:
        new_dataset = PairedImageDataset(
            sketch_dir=NEW_MANGA_SKETCH_DIR,
            color_dir=NEW_MANGA_COLOR_DIR,
            augment=True,
        )
        datasets.append(new_dataset)
        print(f"Loaded new dataset: {len(new_dataset)} pairs")
    except Exception as e:
        print(f"Warning: Could not load new dataset: {e}")
        if not USE_BOTH_DATASETS:
            raise
    
    # Combine datasets if using both
    if len(datasets) > 1:
        manga_dataset = ConcatDataset(datasets)
    else:
        manga_dataset = datasets[0]
    
    # Split into train/val (80/20)
    total_size = len(manga_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(
        manga_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
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
    # Generator
    netG = UNetGenerator(in_channels=1, out_channels=3, ngf=64).to(device)
    netG.apply(lambda m: init_weights(m, init_type="normal", init_gain=0.02))

    # Discriminator (1 channel input + 3 channel target = 4)
    netD = PatchGANDiscriminator(in_channels=4, ndf=64).to(device)
    netD.apply(lambda m: init_weights(m, init_type="normal", init_gain=0.02))

    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

    return netG, netD


def validate(val_loader: DataLoader, netG: UNetGenerator, criterion_L1: torch.nn.L1Loss, device: torch.device) -> float:
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
            real_B = batch["B"].to(device)
            fake_B = netG(real_A)
            total_l1 += criterion_L1(fake_B, real_B).item()
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
    criterion_L1: torch.nn.L1Loss,
    device: torch.device,
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
        real_B = batch["B"].to(device)  # color [B, 3, H, W]

        # ============================================================
        # (a) Train Discriminator
        # ============================================================

        with torch.no_grad():
            fake_B = netG(real_A)

        # Real pair: (A, B)
        real_input = torch.cat([real_A, real_B], dim=1)  # [B, 4, H, W]
        pred_real = netD(real_input)
        target_real = torch.ones_like(pred_real) * 0.9  # Label smoothing: 0.9 instead of 1.0
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake pair: (A, G(A))
        fake_input = torch.cat([real_A, fake_B.detach()], dim=1)  # [B, 4, H, W]
        pred_fake = netD(fake_input)
        target_fake = torch.zeros_like(pred_fake) + 0.1  # Label smoothing: 0.1 instead of 0.0
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        optimizer_D.zero_grad()
        loss_D.backward()
        torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer_D.step()

        # ============================================================
        # (b) Train Generator
        # ============================================================

        fake_B = netG(real_A)

        fake_input_for_G = torch.cat([real_A, fake_B], dim=1)
        pred_fake_for_G = netD(fake_input_for_G)
        target_real_for_G = torch.ones_like(pred_fake_for_G) * 0.9  # Label smoothing: 0.9

        loss_G_GAN = criterion_GAN(pred_fake_for_G, target_real_for_G)
        loss_G_L1 = criterion_L1(fake_B, real_B) * LAMBDA_L1
        loss_G = loss_G_GAN + loss_G_L1

        optimizer_G.zero_grad()
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer_G.step()

        # ============================================================
        # (c) Logging
        # ============================================================

        if (iteration + 1) % LOG_INTERVAL == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] "
                f"Iteration [{iteration + 1}/{len(train_loader)}] "
                f"Loss_D: {loss_D.item():.4f} "
                f"Loss_G: {loss_G.item():.4f} "
                f"Loss_G_GAN: {loss_G_GAN.item():.4f} "
                f"Loss_G_L1: {loss_G_L1.item():.4f}"
            )

        # ============================================================
        # (d) Sample Saving
        # ============================================================

        if (iteration + 1) % SAMPLE_INTERVAL == 0:
            sample_real_A = real_A[0:1]   # [1, 1, H, W] - input BW
            sample_real_B = real_B[0:1]   # [1, 3, H, W] - ground truth color
            sample_fake_B = fake_B[0:1]   # [1, 3, H, W] - generated color

            # Save input BW (convert to RGB for easier viewing by repeating channel)
            sample_real_A_rgb = sample_real_A.repeat(1, 3, 1, 1)  # [1, 3, H, W]
            save_image(
                sample_real_A_rgb,
                os.path.join(SAMPLE_DIR, f"epoch_{epoch}_iter_{iteration + 1}_real_A.png"),
            )
            save_image(
                sample_real_B,
                os.path.join(SAMPLE_DIR, f"epoch_{epoch}_iter_{iteration + 1}_real_B.png"),
            )
            save_image(
                sample_fake_B,
                os.path.join(SAMPLE_DIR, f"epoch_{epoch}_iter_{iteration + 1}_fake_B.png"),
            )

            print(f"Saved sample images at epoch {epoch}, iteration {iteration + 1} (input BW, ground truth, generated)")


def save_checkpoint(
    epoch: int,
    netG: UNetGenerator,
    netD: PatchGANDiscriminator,
    optimizer_G: torch.optim.Adam,
    optimizer_D: torch.optim.Adam,
) -> None:
    """
    Save training checkpoint.

    Args:
        epoch: Current epoch number.
        netG: Generator model.
        netD: Discriminator model.
        optimizer_G: Generator optimizer.
        optimizer_D: Discriminator optimizer.
    """
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

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

    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_L1 = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))
    
    # Learning rate schedulers
    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.95)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.95)

    RESUME = True  # set False when you want to start from scratch

    CHECKPOINT_PATH = "checkpoints/pix2pix_epoch_28.pth"  # ← change to latest checkpoint

    start_epoch = 1

    if RESUME and os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        netG.load_state_dict(checkpoint["netG_state_dict"])
        netD.load_state_dict(checkpoint["netD_state_dict"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting training from scratch.")

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_one_epoch(
            epoch=epoch,
            train_loader=train_loader,
            netG=netG,
            netD=netD,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            criterion_GAN=criterion_GAN,
            criterion_L1=criterion_L1,
            device=device,
        )
        
        # Validate and save best checkpoint
        val_loss = validate(val_loader, netG, criterion_L1, device)
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Validation L1 Loss: {val_loss:.4f}")
        
        save_checkpoint(epoch, netG, netD, optimizer_G, optimizer_D)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "pix2pix_best.pth")
            torch.save({
                "epoch": epoch,
                "netG_state_dict": netG.state_dict(),
                "netD_state_dict": netD.state_dict(),
                "val_loss": val_loss,
            }, best_path)
            print(f"Saved best checkpoint with val_loss={val_loss:.4f}")
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
        print(f"Learning rates: G={scheduler_G.get_last_lr()[0]:.6f}, D={scheduler_D.get_last_lr()[0]:.6f}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()