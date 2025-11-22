"""
Training script for Pix2Pix Manga Colorization

This script trains a Pix2Pix conditional GAN to colorize black-and-white
anime/manga images using **paired** datasets:

1) Anime Sketch Colorization Pair (split into sketch/color)
2) Manga dataset with paired bw/color pages

It implements the standard Pix2Pix training loop with GAN loss and L1 loss,
handles logging, sample saving, and checkpointing.
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, ConcatDataset

from dataset import PairedImageDataset
from networks import UNetGenerator, PatchGANDiscriminator
from utils import init_weights, save_image, set_seed, get_device


# ============================================================================#
# Hyperparameters
# ============================================================================#

# Data roots (paired datasets)
ANIME_SKETCH_DIR = "data/anime_pair/sketch"
ANIME_COLOR_DIR = "data/anime_pair/color"

MANGA_SKETCH_DIR = "data/manga_dataset/bw"
MANGA_COLOR_DIR = "data/manga_dataset/color"

# Training hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 4
LR = 2e-4
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 100.0

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


def create_data_loaders() -> DataLoader:
    """
    Create data loader for training.

    Combines two paired datasets:
      - Anime sketch/color pairs
      - Manga bw/color pairs

    Returns:
        DataLoader for the combined dataset.
    """
    # Paired anime dataset (sketch → color)
    anime_dataset = PairedImageDataset(
        sketch_dir=ANIME_SKETCH_DIR,
        color_dir=ANIME_COLOR_DIR,
        augment=True,
    )

    # Paired manga dataset (bw → color)
    manga_dataset = PairedImageDataset(
        sketch_dir=MANGA_SKETCH_DIR,
        color_dir=MANGA_COLOR_DIR,
        augment=True,
    )

    # Combine datasets
    combined_dataset = ConcatDataset([anime_dataset, manga_dataset])

    # DataLoader
    train_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(f"Total training samples: {len(combined_dataset)}")
    print(f"  - Anime paired samples: {len(anime_dataset)}")
    print(f"  - Manga paired samples: {len(manga_dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")

    return train_loader


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
        target_real = torch.ones_like(pred_real)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake pair: (A, G(A))
        fake_input = torch.cat([real_A, fake_B.detach()], dim=1)  # [B, 4, H, W]
        pred_fake = netD(fake_input)
        target_fake = torch.zeros_like(pred_fake)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ============================================================
        # (b) Train Generator
        # ============================================================

        fake_B = netG(real_A)

        fake_input_for_G = torch.cat([real_A, fake_B], dim=1)
        pred_fake_for_G = netD(fake_input_for_G)
        target_real_for_G = torch.ones_like(pred_fake_for_G)

        loss_G_GAN = criterion_GAN(pred_fake_for_G, target_real_for_G)
        loss_G_L1 = criterion_L1(fake_B, real_B) * LAMBDA_L1
        loss_G = loss_G_GAN + loss_G_L1

        optimizer_G.zero_grad()
        loss_G.backward()
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
            sample_real_B = real_B[0:1]   # [1, 3, H, W]
            sample_fake_B = fake_B[0:1]   # [1, 3, H, W]

            save_image(
                sample_real_B,
                os.path.join(SAMPLE_DIR, f"epoch_{epoch}_iter_{iteration + 1}_real_B.png"),
            )
            save_image(
                sample_fake_B,
                os.path.join(SAMPLE_DIR, f"epoch_{epoch}_iter_{iteration + 1}_fake_B.png"),
            )

            print(f"Saved sample images at epoch {epoch}, iteration {iteration + 1}")


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
    train_loader = create_data_loaders()

    print("\n" + "=" * 60)
    print("Initializing models...")
    print("=" * 60)
    netG, netD = initialize_models(device)

    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_L1 = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))

    RESUME = True  # set False when you want to start from scratch

    CHECKPOINT_PATH = "checkpoints/pix2pix_epoch_2.pth"  # ← change to latest checkpoint

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

        save_checkpoint(epoch, netG, netD, optimizer_G, optimizer_D)

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
