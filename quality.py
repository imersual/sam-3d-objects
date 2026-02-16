import sys
import cv2
import numpy as np
import torch
import piq
import os


def brightness_metrics(gray):
    mean_lum = gray.mean()
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    return mean_lum, p95 - p5


def needs_upscale(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return True

    h, w = img.shape[:2]
    min_dim = min(h, w)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mean_lum, dyn_range = brightness_metrics(gray)

    if mean_lum < 60:
        return True

    if min_dim < 384:
        return True

    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        brisque = piq.brisque(img_tensor).item()

    if min_dim >= 512 and brisque < 35 and dyn_range > 100:
        return False

    if min_dim >= 384 and brisque < 35:
        return False

    return True


if __name__ == "__main__":
    exts = (".jpg", ".jpeg", ".png")

    images = [f for f in os.listdir(".") if f.lower().endswith(exts)]

    if not images:
        print(True)
        sys.exit(0)

    print(needs_upscale(images[0]))
