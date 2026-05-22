"""
Download a labelled face dataset and split it into train / val / test.

Real faces  -- randomuser.me portrait photos (CC-licensed profile pics)
Fake faces  -- thispersondoesnotexist.com  (StyleGAN2-generated faces)

Usage:
    python download_data.py               # 50 per class (default)
    python download_data.py --count 100   # 100 per class
"""

import argparse
import math
import time
from pathlib import Path

import requests

from config import TRAIN_RATIO, VAL_RATIO, TRAIN_DIR, VAL_DIR, TEST_DIR

HEADERS = {'User-Agent': 'Mozilla/5.0 (DeepfakeDetection demo; educational use)'}


def download_image(url: str, save_path: Path, timeout: int = 15) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        if not r.content:
            return False
        save_path.write_bytes(r.content)
        return True
    except Exception as exc:
        print(f'    FAIL: {exc}')
        return False


def _real_url(i: int) -> str:
    gender = 'men' if i % 2 == 0 else 'women'
    idx = (i // 2) % 99 + 1
    return f'https://randomuser.me/api/portraits/{gender}/{idx}.jpg'


def _fake_url(i: int) -> str:
    return f'https://thispersondoesnotexist.com/?v={i}'


def _split_counts(total: int) -> tuple[int, int, int]:
    n_train = math.floor(total * TRAIN_RATIO)
    n_val   = math.floor(total * VAL_RATIO)
    n_test  = total - n_train - n_val
    return n_train, n_val, n_test


def download(count: int):
    n_train, n_val, n_test = _split_counts(count)
    print(f'\n  Split: {n_train} train | {n_val} val | {n_test} test  (per class)')

    for label, url_fn, delay in [('real', _real_url, 0.3), ('fake', _fake_url, 0.6)]:
        dirs = {
            'train': Path(TRAIN_DIR) / label,
            'val':   Path(VAL_DIR)   / label,
            'test':  Path(TEST_DIR)  / label,
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        split_sizes = {'train': n_train, 'val': n_val, 'test': n_test}
        img_idx = 0
        ok = 0

        print(f'\n  Downloading {count} {label} faces...')
        for split_name, size in split_sizes.items():
            for _ in range(size):
                fname = f'{label}_{img_idx + 1:03d}.jpg'
                dst   = dirs[split_name] / fname
                if dst.exists():
                    print(f'    [{img_idx+1}/{count}] skip  {split_name}/{fname}')
                    ok += 1
                else:
                    s = download_image(url_fn(img_idx), dst)
                    print(f'    [{img_idx+1}/{count}] {"OK" if s else "FAIL"}  '
                          f'{split_name}/{fname}')
                    ok += s
                    time.sleep(delay)
                img_idx += 1
        print(f'  Done - {ok}/{count} downloaded.')

    print()
    for name, d in [('train', TRAIN_DIR), ('val', VAL_DIR), ('test', TEST_DIR)]:
        r = len(list((Path(d) / 'real').glob('*.jpg'))) if (Path(d) / 'real').exists() else 0
        f = len(list((Path(d) / 'fake').glob('*.jpg'))) if (Path(d) / 'fake').exists() else 0
        print(f'  {name:5s}: {r} real + {f} fake = {r+f} total')

    print('\n  Run:  python train.py  to fine-tune the models.')
    print('  Then: python main.py   to evaluate on the test split.')


def main():
    parser = argparse.ArgumentParser(description='Download sample deepfake dataset')
    parser.add_argument('--count', type=int, default=50,
                        help='Images per class (default: 50)')
    args = parser.parse_args()

    print('=' * 52)
    print('  Deepfake Detection - Dataset Downloader')
    print('=' * 52)
    print(f'  Downloading {args.count} images per class...')
    download(args.count)


if __name__ == '__main__':
    main()
