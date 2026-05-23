"""
Split a custom flat dataset into train / val / test for use with scripts/train.py.

Provide the paths to your own real and fake image directories.
Files are copied by default; pass --move to relocate them instead.

Usage (run from project root):
    python scripts/prepare_data.py --real path/to/real --fake path/to/fake
    python scripts/prepare_data.py --real path/to/real --fake path/to/fake --move
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import math
import random
import shutil

from src.config import TRAIN_DIR, VAL_DIR, TEST_DIR, TRAIN_RATIO, VAL_RATIO

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def split_list(items: list, train_r: float, val_r: float):
    n = len(items)
    n_train = math.floor(n * train_r)
    n_val   = math.floor(n * val_r)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def prepare_class(src_dir: Path, label: str,
                  train_r: float, val_r: float,
                  seed: int, move: bool):
    paths = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        print(f'  No images found in {src_dir}')
        return

    rng = random.Random(seed)
    rng.shuffle(paths)

    train_paths, val_paths, test_paths = split_list(paths, train_r, val_r)
    split_map = {TRAIN_DIR: train_paths, VAL_DIR: val_paths, TEST_DIR: test_paths}

    op      = shutil.move if move else shutil.copy2
    op_name = 'Moved' if move else 'Copied'

    for dest_root, file_list in split_map.items():
        dest = Path(dest_root) / label
        dest.mkdir(parents=True, exist_ok=True)
        for p in file_list:
            op(str(p), str(dest / p.name))

    print(f'  {label:4s}: {len(train_paths)} train | '
          f'{len(val_paths)} val | {len(test_paths)} test  ({op_name})')


def main():
    parser = argparse.ArgumentParser(
        description='Split a flat dataset into train/val/test for scripts/train.py')
    parser.add_argument('--real', required=True,
                        help='Path to directory containing real face images')
    parser.add_argument('--fake', required=True,
                        help='Path to directory containing fake/deepfake images')
    parser.add_argument('--move', action='store_true',
                        help='Move files instead of copying')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible shuffle (default: 42)')
    args = parser.parse_args()

    real_src = Path(args.real)
    fake_src = Path(args.fake)

    for p, name in [(real_src, '--real'), (fake_src, '--fake')]:
        if not p.exists():
            print(f'Error: {name} path does not exist: {p}')
            return

    print(f'Splitting images  (seed={args.seed}, {"move" if args.move else "copy"} mode) ...')
    prepare_class(real_src, 'real', TRAIN_RATIO, VAL_RATIO, args.seed, args.move)
    prepare_class(fake_src, 'fake', TRAIN_RATIO, VAL_RATIO, args.seed, args.move)

    print()
    for split_name, split_dir in [('train', TRAIN_DIR), ('val', VAL_DIR), ('test', TEST_DIR)]:
        r = len(list((Path(split_dir) / 'real').glob('*'))) if (Path(split_dir) / 'real').exists() else 0
        f = len(list((Path(split_dir) / 'fake').glob('*'))) if (Path(split_dir) / 'fake').exists() else 0
        print(f'  {split_name:5s}: {r} real + {f} fake = {r+f} total')

    print('\nDone. Run  python scripts/train.py  to start fine-tuning.')


if __name__ == '__main__':
    main()
