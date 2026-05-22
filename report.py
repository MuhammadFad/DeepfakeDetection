import json
import os
from datetime import datetime

import plotly.graph_objects as go
import plotly.offline as pyo

from config import (
    OUTPUT_DIR, OUTPUT_HTML,
    MODEL_DISPLAY_NAMES,
    MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET,
    TRAINING_HISTORY_PATH,
)

# Canonical display order for models
MODEL_ORDER = [MODEL_XCEPTION, MODEL_VIT, MODEL_EFFICIENTNET]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _acc_color(acc: float) -> str:
    if acc >= 70:
        return '#3fb950'
    if acc >= 50:
        return '#d29922'
    return '#f85149'


def _generate_accuracy_chart(summary_stats: dict) -> str:
    names, accs, colors = [], [], []
    for mn in MODEL_ORDER:
        st = summary_stats.get(mn, {})
        names.append(st.get('display_name', mn))
        acc = st.get('accuracy', 0.0)
        accs.append(round(acc, 1))
        colors.append(_acc_color(acc))

    fig = go.Figure(data=[go.Bar(
        x=names,
        y=accs,
        marker_color=colors,
        marker_line_color='rgba(255,255,255,0.15)',
        marker_line_width=1,
        text=[f'{a}%' for a in accs],
        textposition='outside',
        textfont=dict(color='#e6edf3', size=15, family='Segoe UI, system-ui, sans-serif'),
        hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>',
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b949e', family='Segoe UI, system-ui, sans-serif'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.06)',
            tickfont=dict(size=13, color='#c9d1d9'),
            showline=False,
        ),
        yaxis=dict(
            range=[0, 115],
            gridcolor='rgba(255,255,255,0.06)',
            tickfont=dict(size=12, color='#8b949e'),
            title=dict(text='Accuracy (%)', font=dict(size=13)),
            showline=False,
            zeroline=False,
        ),
        height=360,
        margin=dict(t=24, b=16, l=56, r=24),
        bargap=0.35,
    )

    return pyo.plot(fig, output_type='div', include_plotlyjs='cdn', config={'displayModeBar': False})


def _cm_html(cm, display_name: str) -> str:
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    return f"""
    <div class="cm-card">
      <p class="cm-title">{display_name}</p>
      <table class="cm-table">
        <thead>
          <tr>
            <th class="cm-corner"></th>
            <th class="cm-head">Pred&nbsp;Real</th>
            <th class="cm-head">Pred&nbsp;Fake</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td class="cm-label">Actual Real</td>
            <td class="cm-ok">{tn}<span class="cm-badge">TN</span></td>
            <td class="cm-bad">{fp}<span class="cm-badge">FP</span></td>
          </tr>
          <tr>
            <td class="cm-label">Actual Fake</td>
            <td class="cm-bad">{fn}<span class="cm-badge">FN</span></td>
            <td class="cm-ok">{tp}<span class="cm-badge">TP</span></td>
          </tr>
        </tbody>
      </table>
    </div>"""


_MODEL_SHORT = {
    'xception':             'Xcept.',
    'vit_base_patch16_224': 'ViT-B',
    'efficientnet_b4':      'EfficientNet',
}


def _image_gallery(image_results: list) -> str:
    """Build a card grid showing each image, its Grad-CAM heatmaps, and model predictions."""
    if not image_results:
        return ''

    has_heatmaps = any(
        any(v is not None for v in row.get('heatmaps', {}).values())
        for row in image_results
    )

    def _img_tag(b64, alt='', cls='gc-thumb'):
        if b64:
            return f'<img src="data:image/jpeg;base64,{b64}" alt="{alt}" class="{cls}">'
        return f'<div class="{cls} gc-no-img">N/A</div>'

    cards = ''
    for row in image_results:
        gt       = row['ground_truth']
        name     = row['image_name']
        preds    = row.get('predictions', {})
        heatmaps = row.get('heatmaps', {})
        orig_b64 = row.get('original_b64')

        correct = sum(1 for mn in MODEL_ORDER if preds.get(mn, {}).get('label') == gt)
        valid   = sum(1 for mn in MODEL_ORDER if preds.get(mn, {}).get('label') not in ('N/A', None, ''))
        if valid == 0:
            card_cls = 'gc-neutral'
        elif correct == valid:
            card_cls = 'gc-correct'
        elif correct == 0:
            card_cls = 'gc-wrong'
        else:
            card_cls = 'gc-mixed'

        gt_badge = (f'<span class="badge-real">Real</span>' if gt == 'Real'
                    else f'<span class="badge-fake">Fake</span>')

        figs = f'''<figure class="gc-fig">
              <div class="gc-img-box">{_img_tag(orig_b64, "original")}</div>
              <figcaption>Original</figcaption>
            </figure>'''

        for mn in MODEL_ORDER:
            short  = _MODEL_SHORT.get(mn, mn[:7])
            pd     = preds.get(mn, {})
            label  = pd.get('label', 'N/A')
            conf   = pd.get('confidence', 0.0)
            hm     = heatmaps.get(mn)

            if label == 'N/A':
                pred_cls, pred_txt = 'pred-na', 'N/A'
            elif label == gt:
                pred_cls, pred_txt = 'pred-correct', f'{label} {conf:.0f}%'
            else:
                pred_cls, pred_txt = 'pred-wrong', f'{label} {conf:.0f}%'

            img_content = _img_tag(hm, f'{short} CAM') if hm else _img_tag(None)
            figs += f'''<figure class="gc-fig">
              <div class="gc-img-box">{img_content}</div>
              <figcaption>{short}<br><span class="{pred_cls}">{pred_txt}</span></figcaption>
            </figure>'''

        cards += f'''<div class="gallery-card {card_cls}">
          <div class="gc-header">
            <span class="gc-name" title="{name}">{name}</span>
            {gt_badge}
          </div>
          <div class="gc-images">{figs}</div>
        </div>'''

    if has_heatmaps:
        note = ('<p class="cam-note">'
                'Heatmaps show Grad-CAM activation &mdash; '
                '<span style="color:#f85149">red&nbsp;/&nbsp;warm</span> regions had the highest '
                'influence on the model\'s prediction; '
                '<span style="color:#58a6ff">blue&nbsp;/&nbsp;cool</span> regions '
                'contributed less.</p>')
    else:
        note = '<p class="cam-note">Grad-CAM heatmaps unavailable (run: pip install grad-cam).</p>'

    return f'''
  <!-- Image Gallery -->
  <section class="section">
    <h2 class="section-title">Image Analysis Gallery</h2>
    <div class="legend" style="margin-bottom:10px">
      <span><span class="legend-dot" style="background:#3fb950"></span>All correct</span>
      <span><span class="legend-dot" style="background:#d29922"></span>Mixed</span>
      <span><span class="legend-dot" style="background:#f85149"></span>All wrong</span>
    </div>
    {note}
    <div class="gallery-grid">{cards}</div>
  </section>'''


def _interpretation(summary_stats: dict) -> str:
    accs = {mn: summary_stats.get(mn, {}).get('accuracy', 0.0) for mn in MODEL_ORDER}
    best = max(accs, key=accs.get)
    worst = min(accs, key=accs.get)

    xc = accs[MODEL_XCEPTION]
    vt = accs[MODEL_VIT]
    ef = accs[MODEL_EFFICIENTNET]
    eff_fake_tpr = summary_stats.get(MODEL_EFFICIENTNET, {}).get('per_class', {}).get('Fake', 0.0)

    xc_vs_vt = "outperforming" if xc > vt else "underperforming compared to"
    best_name = summary_stats[best]['display_name']
    worst_name = summary_stats[worst]['display_name']

    return (
        f"XceptionNet (CNN) achieved <strong>{xc:.1f}%</strong> overall accuracy, "
        f"{xc_vs_vt} the Vision Transformer at <strong>{vt:.1f}%</strong>. "
        f"The hybrid EfficientNet-B4 model recorded a fake-detection rate (TPR) of "
        f"<strong>{eff_fake_tpr:.1f}%</strong> and overall accuracy of <strong>{ef:.1f}%</strong>. "
        f"Across all three architectures, <strong>{best_name}</strong> led with "
        f"<strong>{accs[best]:.1f}%</strong> accuracy, while <strong>{worst_name}</strong> "
        f"produced the lowest result at <strong>{accs[worst]:.1f}%</strong>. "
        f"Note: these models use ImageNet-pretrained backbones with randomly-initialised "
        f"binary classification heads — fine-tuning on labelled deepfake datasets would "
        f"substantially improve detection performance."
    )


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------
# Training curves (optional — only present when training_history.json exists)
# ---------------------------------------------------------------------------

def _load_training_history() -> dict | None:
    if not os.path.exists(TRAINING_HISTORY_PATH):
        return None
    try:
        with open(TRAINING_HISTORY_PATH, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _generate_training_curves(history: dict) -> str:
    """Return a Plotly div showing train/val accuracy per epoch for each model."""
    COLORS = {
        MODEL_XCEPTION:     ('#58a6ff', '#1f6feb'),
        MODEL_VIT:          ('#3fb950', '#238636'),
        MODEL_EFFICIENTNET: ('#d29922', '#9e6a03'),
    }

    fig = go.Figure()
    for mn in MODEL_ORDER:
        if mn not in history:
            continue
        h = history[mn]
        epochs = list(range(1, len(h['train_acc']) + 1))
        solid, faded = COLORS.get(mn, ('#aaa', '#666'))
        name = MODEL_DISPLAY_NAMES.get(mn, mn)

        fig.add_trace(go.Scatter(
            x=epochs, y=h['train_acc'],
            mode='lines+markers',
            name=f'{name} train',
            line=dict(color=solid, width=2, dash='dot'),
            marker=dict(size=5),
            hovertemplate='Epoch %{x}<br>Train acc: %{y:.1f}%<extra>' + name + '</extra>',
        ))
        if any(v > 0 for v in h.get('val_acc', [])):
            fig.add_trace(go.Scatter(
                x=epochs, y=h['val_acc'],
                mode='lines+markers',
                name=f'{name} val',
                line=dict(color=solid, width=2.5),
                marker=dict(size=6, symbol='diamond'),
                hovertemplate='Epoch %{x}<br>Val acc: %{y:.1f}%<extra>' + name + ' val</extra>',
            ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8b949e', family='Segoe UI, system-ui, sans-serif'),
        xaxis=dict(
            title='Epoch',
            gridcolor='rgba(255,255,255,0.06)',
            tickfont=dict(size=12, color='#c9d1d9'),
        ),
        yaxis=dict(
            title='Accuracy (%)',
            range=[0, 105],
            gridcolor='rgba(255,255,255,0.06)',
            tickfont=dict(size=12, color='#8b949e'),
        ),
        legend=dict(
            bgcolor='rgba(22,27,34,0.9)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(size=11),
        ),
        height=380,
        margin=dict(t=16, b=40, l=56, r=24),
        hovermode='x unified',
    )

    return pyo.plot(fig, output_type='div', include_plotlyjs=False, config={'displayModeBar': False})


# ---------------------------------------------------------------------------

def generate_html_report(image_results: list, summary_stats: dict) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total_images = len(image_results)

    chart_div = _generate_accuracy_chart(summary_stats)

    # --- Summary cards ---
    total_card = f"""
    <div class="card">
      <div class="card-label">Total Images</div>
      <div class="card-value" style="color:#79c0ff">{total_images}</div>
      <div class="card-sub">Tested</div>
    </div>"""

    model_cards = ''
    for mn in MODEL_ORDER:
        st = summary_stats.get(mn, {})
        acc = st.get('accuracy', 0.0)
        model_cards += f"""
    <div class="card">
      <div class="card-label">{st.get('display_name', mn)}</div>
      <div class="card-value" style="color:{_acc_color(acc)}">{acc:.1f}%</div>
      <div class="card-sub">Accuracy</div>
    </div>"""

    # --- Results table rows ---
    rows_html = ''
    for row in image_results:
        gt = row['ground_truth']
        gt_badge = f'<span class="badge-real">Real</span>' if gt == 'Real' else f'<span class="badge-fake">Fake</span>'
        cells = f'<td class="td-name">{row["image_name"]}</td><td>{gt_badge}</td>'

        for mn in MODEL_ORDER:
            pred_data = row.get('predictions', {}).get(mn, {})
            label = pred_data.get('label', 'N/A')
            conf  = pred_data.get('confidence', 0.0)

            if label == 'N/A':
                cell_cls = 'td-na'
                conf_str = '—'
            elif label == gt:
                cell_cls = 'td-correct'
                conf_str = f'{conf:.1f}%'
            else:
                cell_cls = 'td-wrong'
                conf_str = f'{conf:.1f}%'

            cells += f'<td class="{cell_cls}">{label}</td><td class="td-conf">{conf_str}</td>'

        rows_html += f'<tr>{cells}</tr>\n'

    # --- Confusion matrices ---
    cm_html = ''
    for mn in MODEL_ORDER:
        st = summary_stats.get(mn, {})
        cm = st.get('confusion_matrix')
        if cm is not None:
            cm_html += _cm_html(cm, st.get('display_name', mn))

    interpretation = _interpretation(summary_stats)
    gallery_section = _image_gallery(image_results)

    # --- Training curves (optional) ---
    training_history = _load_training_history()
    if training_history:
        curves_div = _generate_training_curves(training_history)
        training_section = f"""
  <!-- Training Curves -->
  <section class="section">
    <h2 class="section-title">Training Curves</h2>
    <div class="chart-box">
      {curves_div}
    </div>
    <p style="color:#6e7681;font-size:0.82rem;margin-top:10px;padding-left:4px;">
      Solid lines = validation accuracy &nbsp;|&nbsp; Dotted lines = training accuracy
    </p>
  </section>"""
    else:
        training_section = ''

    # --- Model column headers ---
    model_th = ''.join(
        f'<th colspan="2" class="th-model">{MODEL_DISPLAY_NAMES.get(mn, mn)}</th>'
        for mn in MODEL_ORDER
    )
    model_sub_th = '<th>Prediction</th><th>Confidence</th>' * len(MODEL_ORDER)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Deepfake Detection Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.6;
    min-height: 100vh;
  }}

  /* ── Header ── */
  .header {{
    background: linear-gradient(160deg, #161b22 0%, #0d1117 100%);
    border-bottom: 1px solid #21262d;
    padding: 48px 32px 40px;
    text-align: center;
  }}
  .header-icon {{
    font-size: 2.8rem;
    margin-bottom: 12px;
    display: block;
  }}
  .header h1 {{
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #58a6ff 0%, #79c0ff 50%, #a5f3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin-bottom: 6px;
  }}
  .header .subtitle {{
    color: #8b949e;
    font-size: 1.05rem;
  }}
  .header .timestamp {{
    color: #484f58;
    font-size: 0.82rem;
    margin-top: 6px;
    letter-spacing: 0.03em;
  }}

  /* ── Layout ── */
  .page {{
    max-width: 1440px;
    margin: 0 auto;
    padding: 32px 24px 64px;
  }}
  .section {{ margin-bottom: 48px; }}
  .section-title {{
    font-size: 1.05rem;
    font-weight: 600;
    color: #58a6ff;
    border-left: 3px solid #1f6feb;
    padding-left: 12px;
    margin-bottom: 20px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}

  /* ── Cards ── */
  .cards-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
  }}
  .card {{
    flex: 1;
    min-width: 180px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
    transition: border-color 0.2s, transform 0.15s;
  }}
  .card:hover {{
    border-color: #58a6ff;
    transform: translateY(-2px);
  }}
  .card-label {{
    color: #6e7681;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 6px;
  }}
  .card-value {{
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 4px;
  }}
  .card-sub {{
    color: #6e7681;
    font-size: 0.8rem;
  }}

  /* ── Chart ── */
  .chart-box {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 12px 8px;
    overflow: hidden;
  }}

  /* ── Table ── */
  .table-scroll {{
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid #30363d;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88rem;
    background: #161b22;
    white-space: nowrap;
  }}
  thead tr:first-child {{
    background: #0d1117;
  }}
  thead tr:last-child {{
    background: #10161d;
  }}
  th {{
    padding: 11px 14px;
    color: #6e7681;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #30363d;
    text-align: center;
  }}
  .th-model {{
    color: #79c0ff;
    font-size: 0.8rem;
    border-left: 1px solid #21262d;
  }}
  td {{
    padding: 9px 14px;
    border-bottom: 1px solid #21262d;
    text-align: center;
    vertical-align: middle;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:nth-child(even) {{ background: #0d1117; }}
  tr:hover {{ background: #161b22 !important; }}
  .td-name {{
    text-align: left;
    color: #c9d1d9;
    font-family: 'Cascadia Code', 'Courier New', monospace;
    font-size: 0.8rem;
    max-width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  .td-correct {{ color: #3fb950; font-weight: 600; }}
  .td-wrong   {{ color: #f85149; font-weight: 600; }}
  .td-na      {{ color: #484f58; }}
  .td-conf    {{ color: #6e7681; font-size: 0.82rem; }}
  .badge-real {{
    background: rgba(63,185,80,0.15);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.3);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    font-weight: 600;
  }}
  .badge-fake {{
    background: rgba(248,81,73,0.12);
    color: #f85149;
    border: 1px solid rgba(248,81,73,0.3);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.78rem;
    font-weight: 600;
  }}

  /* ── Confusion matrices ── */
  .cm-row {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }}
  .cm-card {{
    flex: 1;
    min-width: 220px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
  }}
  .cm-title {{
    color: #79c0ff;
    font-size: 0.88rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 14px;
  }}
  .cm-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }}
  .cm-table th, .cm-table td {{
    padding: 9px 12px;
    text-align: center;
    border: none;
    background: transparent;
  }}
  .cm-corner {{ width: 30%; }}
  .cm-head {{
    color: #6e7681;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .cm-label {{
    color: #6e7681;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    text-align: left;
  }}
  .cm-ok {{
    background: rgba(63,185,80,0.12) !important;
    color: #3fb950;
    border-radius: 8px;
    font-size: 1.3rem;
    font-weight: 700;
  }}
  .cm-bad {{
    background: rgba(248,81,73,0.10) !important;
    color: #f85149;
    border-radius: 8px;
    font-size: 1.3rem;
    font-weight: 700;
  }}
  .cm-badge {{
    display: block;
    font-size: 0.62rem;
    font-weight: 400;
    color: #484f58;
    margin-top: 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  /* ── Interpretation ── */
  .interp-box {{
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #1f6feb;
    border-radius: 12px;
    padding: 24px 28px;
    color: #c9d1d9;
    line-height: 1.85;
    font-size: 0.95rem;
  }}
  .interp-box strong {{ color: #79c0ff; }}

  /* ── Legend ── */
  .legend {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 14px;
    font-size: 0.82rem;
    color: #8b949e;
  }}
  .legend-dot {{
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
  }}

  /* ── Gallery ── */
  .gallery-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
  }}
  .gallery-card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
    border-top-width: 3px;
  }}
  .gc-correct {{ border-top-color: #3fb950; }}
  .gc-wrong   {{ border-top-color: #f85149; }}
  .gc-mixed   {{ border-top-color: #d29922; }}
  .gc-neutral {{ border-top-color: #30363d; }}
  .gc-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 12px;
  }}
  .gc-name {{
    color: #8b949e;
    font-family: 'Cascadia Code', 'Courier New', monospace;
    font-size: 0.75rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  .gc-images {{
    display: flex;
    gap: 6px;
  }}
  .gc-fig {{
    flex: 1;
    min-width: 0;
    text-align: center;
    margin: 0;
  }}
  .gc-img-box {{
    width: 100%;
    aspect-ratio: 1;
    background: #0d1117;
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .gc-thumb {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }}
  .gc-no-img {{
    color: #484f58;
    font-size: 0.65rem;
  }}
  .gc-fig figcaption {{
    margin-top: 5px;
    color: #6e7681;
    font-size: 0.68rem;
    line-height: 1.4;
  }}
  .pred-correct {{ color: #3fb950; font-weight: 600; }}
  .pred-wrong   {{ color: #f85149; font-weight: 600; }}
  .pred-na      {{ color: #484f58; }}
  .cam-note {{
    color: #6e7681;
    font-size: 0.82rem;
    margin-bottom: 16px;
    padding-left: 4px;
  }}

  /* ── Footer ── */
  footer {{
    text-align: center;
    padding: 28px;
    color: #484f58;
    font-size: 0.8rem;
    border-top: 1px solid #21262d;
  }}
  footer span {{ color: #6e7681; }}
</style>
</head>
<body>

<header class="header">
  <span class="header-icon">&#x1F50D;</span>
  <h1>Deepfake Detection Report</h1>
  <p class="subtitle">Comparative Analysis &mdash; CNN &nbsp;vs&nbsp; Transformer &nbsp;vs&nbsp; Hybrid</p>
  <p class="timestamp">Generated: {timestamp}</p>
</header>

<main class="page">

  <!-- Summary Cards -->
  <section class="section">
    <h2 class="section-title">Overview</h2>
    <div class="cards-row">
      {total_card}
      {model_cards}
    </div>
  </section>

  <!-- Accuracy Chart -->
  <section class="section">
    <h2 class="section-title">Model Accuracy Comparison</h2>
    <div class="chart-box">
      {chart_div}
    </div>
  </section>

  <!-- Results Table -->
  <section class="section">
    <h2 class="section-title">Per-Image Results</h2>
    <div class="legend">
      <span><span class="legend-dot" style="background:#3fb950"></span>Correct prediction</span>
      <span><span class="legend-dot" style="background:#f85149"></span>Wrong prediction</span>
    </div>
    <div class="table-scroll">
      <table>
        <thead>
          <tr>
            <th rowspan="2" style="text-align:left">Image</th>
            <th rowspan="2">Ground Truth</th>
            {model_th}
          </tr>
          <tr>
            {model_sub_th}
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
  </section>

  {gallery_section}

  <!-- Confusion Matrices -->
  <section class="section">
    <h2 class="section-title">Confusion Matrices</h2>
    <div class="cm-row">
      {cm_html}
    </div>
  </section>

  {training_section}

  <!-- Interpretation -->
  <section class="section">
    <h2 class="section-title">Analysis & Interpretation</h2>
    <div class="interp-box">
      {interpretation}
    </div>
  </section>

</main>

<footer>
  Generated by <span>Deepfake Detection System</span> &bull; {timestamp}
</footer>

</body>
</html>"""

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as fh:
        fh.write(html)

    return OUTPUT_HTML
