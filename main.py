import os
import sys
import logging
import webbrowser
from pathlib import Path

from tqdm import tqdm

from config import (
    TEST_DIR, OUTPUT_HTML, DEVICE,
    MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET,
    IMAGE_SIZE, VIT_IMAGE_SIZE,
    CHECKPOINT_DIR,
)
from preprocessing import load_and_preprocess
from models import load_all_models, run_inference
from evaluation import generate_summary_stats
from report import generate_html_report
from explainability import generate_heatmap, encode_image

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def collect_images() -> list[tuple[Path, str]]:
    """Collect images from images/test/real and images/test/fake."""
    images = []
    for label, folder in [('Real', Path(TEST_DIR) / 'real'),
                           ('Fake', Path(TEST_DIR) / 'fake')]:
        if folder.exists():
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in IMAGE_EXTS:
                    images.append((p, label))
    return images


def has_checkpoints() -> bool:
    ckpt_dir = Path(CHECKPOINT_DIR)
    if not ckpt_dir.exists():
        return False
    return any(ckpt_dir.glob('*_best.pth'))


def print_banner():
    print()
    print('=' * 48)
    print('   Deepfake Detection System v1.0')
    print('=' * 48)
    print(f'  Device      : {DEVICE}')
    trained = 'yes' if has_checkpoints() else 'no (run train.py to improve accuracy)'
    print(f'  Fine-tuned  : {trained}')
    print()


def main():
    print_banner()

    # ── Collect images ──────────────────────────────────────────────────────
    print('Scanning image folders...')
    images = collect_images()

    if not images:
        print('\nNo images found in images/test/.')
        print('  Download data:  python download_data.py --count 50')
        sys.exit(1)

    real_count = sum(1 for _, lb in images if lb == 'Real')
    fake_count = sum(1 for _, lb in images if lb == 'Fake')
    print(f'  Source: images/test/')
    print(f'  Found:  {len(images)} images  ({real_count} real, {fake_count} fake)')

    # ── Load models ─────────────────────────────────────────────────────────
    print('\nLoading models...')
    models = load_all_models(use_checkpoints=True)

    # ── Inference ───────────────────────────────────────────────────────────
    print('\nRunning inference...')

    image_results = []
    all_model_results = {mn: [] for mn in [MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET]}
    skipped = 0

    target_sizes = {
        MODEL_XCEPTION:     IMAGE_SIZE,
        MODEL_VIT:          VIT_IMAGE_SIZE,
        MODEL_EFFICIENTNET: IMAGE_SIZE,
    }

    for img_path, ground_truth in tqdm(images, desc='  Images', unit='img'):
        row = {
            'image_name':   img_path.name,
            'ground_truth': ground_truth,
            'predictions':  {},
            'heatmaps':     {},
            'original_b64': encode_image(img_path, size=224),
        }
        any_success = False

        for mn, model in models.items():
            tensor = load_and_preprocess(img_path, target_size=target_sizes[mn])
            if tensor is None:
                row['predictions'][mn] = {'label': 'N/A', 'confidence': 0.0}
                all_model_results[mn].append({'prediction': None, 'ground_truth': ground_truth})
                continue

            label, confidence = run_inference(model, tensor)
            if label is None:
                row['predictions'][mn] = {'label': 'N/A', 'confidence': 0.0}
                all_model_results[mn].append({'prediction': None, 'ground_truth': ground_truth})
            else:
                row['predictions'][mn] = {'label': label, 'confidence': confidence}
                all_model_results[mn].append({'prediction': label, 'ground_truth': ground_truth})
                row['heatmaps'][mn] = generate_heatmap(model, mn, tensor, img_path)
                any_success = True

        if not any_success:
            skipped += 1
        image_results.append(row)

    if skipped:
        print(f'  Warning: {skipped} image(s) could not be processed.')

    # ── Evaluate ────────────────────────────────────────────────────────────
    print('\nCalculating metrics...')
    summary_stats = generate_summary_stats(all_model_results)

    print()
    print('-' * 46)
    print(f'  {"Model":<28} {"Accuracy":>8}')
    print('-' * 46)
    for mn in [MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET]:
        st = summary_stats[mn]
        print(f'  {st["display_name"]:<28} {st["accuracy"]:>7.1f}%')
    print('-' * 46)

    # ── Report ──────────────────────────────────────────────────────────────
    print('\nGenerating HTML report...')
    report_path = generate_html_report(image_results, summary_stats)
    print(f'  Saved: {report_path}')

    print('\nOpening report in browser...')
    webbrowser.open('file:///' + os.path.abspath(report_path).replace('\\', '/'))

    print('\nDone!\n')


if __name__ == '__main__':
    main()
