from pathlib import Path
import re

import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

from networks import UNetGenerator
from inference_utils import TilingConfig, colorize_with_tiling
from preprocessing import SKETCH_TRANSFORM


CHECKPOINT_PATH = "checkpoints/pix2pix_best.pth"
CHECKPOINT_DIR = Path("checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_netG_cache: dict[str, UNetGenerator] = {}
_checkpoint_label_map: dict[str, str] = {}
TILING_CONFIG = TilingConfig(tile_size=512, overlap=256, blend_mode="feather")


def extract_epoch_number(path: Path) -> int:
    """
    Extract epoch number from checkpoint filename for sorting.
    Returns -1 if not found (for special checkpoints like 'best').
    """
    match = re.search(r"epoch[_\s]*(\d+)", path.stem.lower())
    if match:
        return int(match.group(1))
    return -1


def format_checkpoint_label(path: Path) -> str:
    """
    Produce a clean, human-friendly label without dates.
    """
    stem = path.stem
    
    # Handle special checkpoints
    if "best" in stem.lower():
        return "Best Checkpoint"
    
    # Extract epoch number
    match = re.search(r"epoch[_\s]*(\d+)", stem.lower())
    if match:
        epoch_num = match.group(1)
        return f"Epoch {epoch_num}"
    
    # Fallback: clean up filename
    return stem.replace("pix2pix_", "").replace("_", " ").title()


def list_available_checkpoints() -> list[Path]:
    """
    Return list of checkpoint files sorted by epoch number (numerically).
    """
    if not CHECKPOINT_DIR.exists():
        return []
    
    files = [path for path in CHECKPOINT_DIR.glob("*.pth") if path.is_file()]
    
    # Sort by epoch number (numerically), with special checkpoints at the end
    def sort_key(path: Path) -> tuple[int, str]:
        epoch = extract_epoch_number(path)
        return (epoch, path.name) if epoch >= 0 else (999999, path.name)
    
    return sorted(files, key=sort_key)


def build_checkpoint_choices() -> tuple[list[str], str]:
    """
    Build radio choices sorted by epoch number and default selection.
    """
    _checkpoint_label_map.clear()
    files = list_available_checkpoints()
    
    if not files:
        _checkpoint_label_map["Default"] = CHECKPOINT_PATH
        return ["Default"], "Default"

    choices: list[str] = []
    for path in files:
        label = format_checkpoint_label(path)
        # Handle duplicates by using filename as fallback
        if label in _checkpoint_label_map:
            label = f"{label} ({path.stem})"
        _checkpoint_label_map[label] = str(path)
        choices.append(label)

    # Default to "best" if available, otherwise latest epoch
    default_label = next(
        (label for label in choices if "best" in label.lower()),
        choices[-1] if choices else "Default",
    )

    return choices, default_label


def resolve_checkpoint_path(label: str) -> str:
    """
    Map a UI label back to an absolute checkpoint path.
    """
    if label in _checkpoint_label_map:
        return _checkpoint_label_map[label]

    candidate = CHECKPOINT_DIR / label
    if candidate.exists():
        _checkpoint_label_map[label] = str(candidate)
        return str(candidate)

    if Path(label).exists():
        return label

    return CHECKPOINT_PATH


def load_generator(checkpoint_path: str) -> UNetGenerator:
    """
    Load the generator once and reuse it.
    """
    if checkpoint_path in _netG_cache:
        return _netG_cache[checkpoint_path]

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading generator from {checkpoint_path} on {DEVICE}...")
    netG = UNetGenerator(in_channels=1, out_channels=3, ngf=64)

    state = torch.load(checkpoint_path, map_location=DEVICE)

    # support both formats:
    if isinstance(state, dict) and "netG_state_dict" in state:
        state_dict = state["netG_state_dict"]
    else:
        state_dict = state

    netG.load_state_dict(state_dict)
    netG.to(DEVICE)
    netG.eval()

    _netG_cache[checkpoint_path] = netG
    print("Model loaded.")
    return netG


def colorize(input_image: Image.Image, checkpoint_label: str, use_tiling_option: bool = False) -> Image.Image:
    """
    Gradio callback:
    - takes a PIL image (B/W manga panel)
    - returns a colorized image resized back to original size
    """

    if input_image is None:
        return None

    # remember original size
    orig_w, orig_h = input_image.size

    checkpoint_path = resolve_checkpoint_path(checkpoint_label)

    gray = input_image.convert("L")
    netG = load_generator(checkpoint_path)

    # User can choose to disable tiling to avoid prismatic artifacts
    # Tiling causes color inconsistencies because model has no global context
    # Without tiling: resize to 512x512, process, resize back (no artifacts, but lower res)
    use_tiling = use_tiling_option and max(orig_w, orig_h) > 512

    if use_tiling:
        # For large images, use tiling (will have some artifacts)
        fake_B = colorize_with_tiling(
            gray_image=gray,
            netG=netG,
            device=DEVICE,
            config=TILING_CONFIG,
        )
        fake_B = torch.clamp((fake_B + 1.0) / 2.0, 0.0, 1.0)
        fake_img = fake_B.squeeze(0).cpu()
        fake_img = transforms.ToPILImage()(fake_img)
    else:
        # For smaller images, resize to 512x512 and back (no artifacts, but lower res)
        tensor = SKETCH_TRANSFORM(gray).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            fake_B = netG(tensor)  # [-1,1], [1,3,512,512]

        fake_B = torch.clamp((fake_B + 1.0) / 2.0, 0.0, 1.0)
        fake_img = fake_B.squeeze(0).cpu()
        fake_img = transforms.ToPILImage()(fake_img)
        fake_img = fake_img.resize((orig_w, orig_h), Image.BICUBIC)

    return fake_img

with gr.Blocks(title="Manga Colorization Demo") as demo:
    gr.Markdown(
        """
        # Manga Colorization (Pix2Pix)
        Upload a black-and-white manga panel or line-art.  
        The model will colorize it using a U-Net generator trained with Pix2Pix.
        """
    )

    checkpoint_choices, default_choice = build_checkpoint_choices()

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(
                type="pil",
                label="Input B/W / Sketch",
            )
            checkpoint_selector = gr.Dropdown(
                choices=checkpoint_choices,
                value=default_choice,
                label="Choose Checkpoint",
                interactive=True,
            )
            use_tiling_checkbox = gr.Checkbox(
                value=False,
                label="Use High-Resolution Tiling (may have color artifacts)",
                info="Uncheck to avoid prismatic/rainbow effects. Will resize images instead."
            )
            run_btn = gr.Button("Colorize")
        with gr.Column(scale=1):
            output_color = gr.Image(
                type="pil",
                label="Colorized Output",
            )

    run_btn.click(
        fn=colorize,
        inputs=[input_img, checkpoint_selector, use_tiling_checkbox],
        outputs=output_color,
    )

if __name__ == "__main__":
    demo.launch()