from __future__ import absolute_import, division, print_function

import argparse
import math
import os
import os.path as osp
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from torchreid.data.transforms import build_transforms


DEFAULT_TRANSFORMS = [
    'random_flip',
    'random_crop',
    'random_rotation',
    'color_jitter',
    'gaussian_blur',
    'random_erase',
]
NORM_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def find_images(root):
    candidates = []
    for current_root, _, filenames in os.walk(root):
        if osp.basename(current_root).lower() not in {
            'train', 'query', 'gallery'
        } and 'train' not in current_root.lower():
            continue
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                candidates.append(osp.join(current_root, filename))
    return candidates


def tensor_to_image(tensor):
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * NORM_STD + NORM_MEAN
    return np.clip(array, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description='Save a grid of SoccerNet training augmentations.'
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', help='Path to one player crop.')
    source.add_argument(
        '--root',
        help='SoccerNet root containing a train directory.',
    )
    parser.add_argument('--output', default='augmentation_grid.png')
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--num-images', type=int, default=12)
    parser.add_argument('--columns', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    if args.num_images < 1 or args.columns < 1:
        raise ValueError('num-images and columns must be positive')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_path = args.image
    if image_path is None:
        candidates = find_images(args.root)
        if not candidates:
            raise RuntimeError(
                'No player crops were found under {}'.format(args.root)
            )
        image_path = random.choice(candidates)

    image = Image.open(image_path).convert('RGB')
    transform, _ = build_transforms(
        args.height,
        args.width,
        transforms=DEFAULT_TRANSFORMS,
        norm_mean=NORM_MEAN.tolist(),
        norm_std=NORM_STD.tolist(),
    )

    rows = int(math.ceil(float(args.num_images) / args.columns))
    figure, axes = plt.subplots(
        rows,
        args.columns,
        figsize=(2.5 * args.columns, 4.0 * rows),
        squeeze=False,
    )

    for index, axis in enumerate(axes.flat):
        axis.axis('off')
        if index >= args.num_images:
            continue
        augmented = transform(image.copy())
        axis.imshow(tensor_to_image(augmented))
        axis.set_title('Augmentation {:02d}'.format(index + 1))

    figure.suptitle(osp.basename(image_path))
    figure.tight_layout()
    output_dir = osp.dirname(osp.abspath(args.output))
    if output_dir and not osp.exists(output_dir):
        os.makedirs(output_dir)
    figure.savefig(args.output, dpi=160, bbox_inches='tight')
    plt.close(figure)
    print('Saved augmentation grid to {}'.format(osp.abspath(args.output)))


if __name__ == '__main__':
    main()
