from pathlib import Path
from PIL import Image

# Where your combined (left color + right sketch) images are:
RAW_DIR = Path("data/anime_pair/train")

# Where we want to save the split halves:
OUT_COLOR = Path("data/anime_pair/color")
OUT_SKETCH = Path("data/anime_pair/sketch")

def main() -> None:
    OUT_COLOR.mkdir(parents=True, exist_ok=True)
    OUT_SKETCH.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(RAW_DIR.glob("*.png")) +
        list(RAW_DIR.glob("*.jpg")) +
        list(RAW_DIR.glob("*.jpeg"))
    )

    if not image_files:
        raise RuntimeError(f"No images found in {RAW_DIR}")

    for i, img_path in enumerate(image_files, 1):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # LEFT half = color, RIGHT half = sketch
        left_color = img.crop((0, 0, w // 2, h))
        right_sketch = img.crop((w // 2, 0, w, h))

        out_name = img_path.name  # keep same filename

        (OUT_COLOR / out_name).parent.mkdir(parents=True, exist_ok=True)
        (OUT_SKETCH / out_name).parent.mkdir(parents=True, exist_ok=True)

        left_color.save(OUT_COLOR / out_name)
        right_sketch.save(OUT_SKETCH / out_name)

        if i % 500 == 0:
            print(f"Processed {i}/{len(image_files)} images")

    print("Done!")
    print(f"Color images saved to:  {OUT_COLOR}")
    print(f"Sketch images saved to: {OUT_SKETCH}")


if __name__ == "__main__":
    main()