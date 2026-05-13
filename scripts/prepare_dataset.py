from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize image/mask pairs to 512x512 PNG files.")
    parser.add_argument("--images", required=True, help="Directory containing input images.")
    parser.add_argument("--masks", required=True, help="Directory containing binary masks with matching file stems.")
    parser.add_argument("--output", default="data/processed", help="Output root.")
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args()


def center_crop_resize(image, size: int):
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = image[y0:y0 + side, x0:x0 + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    image_dir = Path(args.images)
    mask_dir = Path(args.masks)
    output = Path(args.output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    for image_path in tqdm(image_paths):
        mask_path = next((mask_dir / f"{image_path.stem}{ext}" for ext in [".png", ".jpg", ".jpeg", ".bmp"] if (mask_dir / f"{image_path.stem}{ext}").exists()), None)
        if mask_path is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        image = center_crop_resize(image, args.size)
        mask = center_crop_resize(mask, args.size)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(output / "images" / f"{image_path.stem}.png"), image)
        cv2.imwrite(str(output / "masks" / f"{image_path.stem}.png"), mask)


if __name__ == "__main__":
    main()

