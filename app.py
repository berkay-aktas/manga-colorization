import torch
from torchvision import transforms
from PIL import Image
import gradio as gr

from networks import UNetGenerator


CHECKPOINT_PATH = "checkpoints/pix2pix_epoch_15.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_netG = None


# same preprocessing as training / colorize.py
preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),           # [0,1], shape [1,H,W]
        transforms.Normalize([0.5], [0.5]),  # -> [-1,1]
    ]
)


def load_generator() -> UNetGenerator:
    """
    Load the generator once and reuse it.
    """
    global _netG
    if _netG is not None:
        return _netG

    print(f"Loading generator from {CHECKPOINT_PATH} on {DEVICE}...")
    netG = UNetGenerator(in_channels=1, out_channels=3, ngf=64)

    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # support both formats:
    if isinstance(state, dict) and "netG_state_dict" in state:
        state_dict = state["netG_state_dict"]
    else:
        state_dict = state

    netG.load_state_dict(state_dict)
    netG.to(DEVICE)
    netG.eval()

    _netG = netG
    print("Model loaded.")
    return netG


def colorize(input_image: Image.Image) -> Image.Image:
    """
    Gradio callback:
    - takes a PIL image (B/W manga panel)
    - returns a colorized image resized back to original size
    """

    if input_image is None:
        return None

    # remember original size
    orig_w, orig_h = input_image.size

    # convert to grayscale
    gray = input_image.convert("L")

    # preprocess to tensor [-1,1], shape [1,1,256,256]
    tensor = preprocess(gray).unsqueeze(0).to(DEVICE)

    netG = load_generator()

    with torch.no_grad():
        fake_B = netG(tensor)  # [-1,1], [1,3,256,256]

    # denormalize [-1,1] -> [0,1]
    fake_B = (fake_B + 1.0) / 2.0
    fake_B = torch.clamp(fake_B, 0.0, 1.0)

    # To PIL (256x256)
    fake_img = fake_B.squeeze(0).cpu()
    fake_img = transforms.ToPILImage()(fake_img)

    # resize back to original page size
    fake_img_big = fake_img.resize((orig_w, orig_h), Image.BICUBIC)

    return fake_img_big

with gr.Blocks(title="Manga Colorization Demo") as demo:
    gr.Markdown(
        """
        # Manga Colorization (Pix2Pix)
        Upload a black-and-white manga panel or line-art.  
        The model will colorize it using a U-Net generator trained with Pix2Pix.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(
                type="pil",
                label="Input B/W / Sketch",
            )
            run_btn = gr.Button("Colorize")
        with gr.Column(scale=1):
            output_color = gr.Image(
                type="pil",
                label="Colorized Output",
            )

    run_btn.click(
        fn=colorize,
        inputs=input_img,
        outputs=output_color,
    )

if __name__ == "__main__":
    demo.launch()