import os
import sys
import logging
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import torch
from src.config import (
    TEST_DIR, OUTPUT_HTML, DEVICE, CPU_THREADS,
    MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET,
    IMAGE_SIZE, VIT_IMAGE_SIZE,
    CHECKPOINT_DIR,
)

if CPU_THREADS > 0:
    torch.set_num_threads(CPU_THREADS)

from src.preprocessing import load_and_preprocess
from src.models import load_all_models, run_inference
from src.evaluation import generate_summary_stats
from src.report import generate_html_report
from src.explainability import generate_heatmap, encode_image

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
    trained = 'yes' if has_checkpoints() else 'no (see train_colab.ipynb)'
    print(f'  Fine-tuned  : {trained}')
    print()


def main():
    print_banner()

    print('Scanning image folders...')
    images = collect_images()

    if not images:
        print('\nNo images found in images/test/.')
        print('  Add images to images/test/real/ and images/test/fake/')
        sys.exit(1)

    real_count = sum(1 for _, lb in images if lb == 'Real')
    fake_count = sum(1 for _, lb in images if lb == 'Fake')
    print(f'  Source: images/test/')
    print(f'  Found:  {len(images)} images  ({real_count} real, {fake_count} fake)')

    print('\nLoading models...')
    models = load_all_models(use_checkpoints=True)

    print('\nRunning inference...')

    image_results = []
    all_model_results = {mn: [] for mn in [MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET]}
    skipped = 0

    target_sizes = {
        MODEL_XCEPTION:     IMAGE_SIZE,
        MODEL_VIT:          VIT_IMAGE_SIZE,
        MODEL_EFFICIENTNET: IMAGE_SIZE,
    }

    def _run_model(mn, model, img_path, ground_truth):
        """Run one model on one image — inference + Grad-CAM. Thread-safe per model."""
        tensor = load_and_preprocess(img_path, target_size=target_sizes[mn])
        if tensor is None:
            return mn, None, 0.0, None
        label, confidence = run_inference(model, tensor)
        if label is None:
            return mn, None, 0.0, None
        heatmap = generate_heatmap(model, mn, tensor, img_path)
        return mn, label, confidence, heatmap

    for img_path, ground_truth in tqdm(images, desc='  Images', unit='img'):
        row = {
            'image_name':   img_path.name,
            'ground_truth': ground_truth,
            'predictions':  {},
            'heatmaps':     {},
            'original_b64': encode_image(img_path, size=224),
        }
        any_success = False

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(_run_model, mn, model, img_path, ground_truth): mn
                for mn, model in models.items()
            }
            for future in as_completed(futures):
                mn, label, confidence, heatmap = future.result()
                if label is None:
                    row['predictions'][mn] = {'label': 'N/A', 'confidence': 0.0}
                    all_model_results[mn].append({'prediction': None, 'ground_truth': ground_truth})
                else:
                    row['predictions'][mn] = {'label': label, 'confidence': confidence}
                    all_model_results[mn].append({'prediction': label, 'ground_truth': ground_truth})
                    row['heatmaps'][mn] = heatmap
                    any_success = True

        if not any_success:
            skipped += 1
        image_results.append(row)

    if skipped:
        print(f'  Warning: {skipped} image(s) could not be processed.')

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

    print('\nGenerating HTML report...')
    report_path = generate_html_report(image_results, summary_stats)
    print(f'  Saved: {report_path}')

    print('\nOpening report in browser...')
    webbrowser.open('file:///' + os.path.abspath(report_path).replace('\\', '/'))

    print('\nDone!\n')


if __name__ == '__main__':
    main()
