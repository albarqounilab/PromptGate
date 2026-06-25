"""Preprocess raw dataset files into numpy arrays under data/."""

import os
from glob import glob

import numpy as np
from PIL import Image


def save_npy(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, data)


def prepare_isic(
    raw_dir='data/FedISIC/ISIC_2019_Training_Input_preprocessed',
    output_dir='data/FedISIC_npy',
):
    img_paths = sorted(
        p for p in glob(os.path.join(raw_dir, '*'))
        if p.lower().endswith(('.jpg', '.jpeg', '.png'))
    )
    if not img_paths:
        raise FileNotFoundError(f'No images found under {raw_dir}')

    os.makedirs(output_dir, exist_ok=True)
    for i, img_path in enumerate(img_paths):
        img_np = np.asarray(Image.open(img_path).convert('RGB'))
        stem = os.path.splitext(os.path.basename(img_path))[0]
        save_npy(img_np, os.path.join(output_dir, f'{stem}.npy'))
        if (i + 1) % 100 == 0 or (i + 1) == len(img_paths):
            print(f'[{i + 1}/{len(img_paths)}] saved to {output_dir}')


if __name__ == '__main__':
    prepare_isic()
