"""
AstroClassifier — Gradio Demo
Galaxy morphology classification — 10 classes, Galaxy10 DECals
"""

import sys
import time
import random
from pathlib import Path

# Monkey-patch gradio_client APIInfoParseError bug (schema can be bool)
try:
    from gradio_client import utils as _gc_utils
    _orig_json_schema = _gc_utils.json_schema_to_python_type
    def _safe_json_schema(schema, defs=None):
        try:
            return _orig_json_schema(schema, defs)
        except Exception:
            return "Any"
    _gc_utils.json_schema_to_python_type = _safe_json_schema
except Exception:
    pass

import gradio as gr
import numpy as np
from PIL import Image, ImageFilter

# ── Try to load real model ────────────────────────────────────────────────────
DEMO_MODE = True

try:
    import torch
    import torch.nn.functional as F
    CHECKPOINT_PATH = Path("model/best_model.pth")
    if CHECKPOINT_PATH.exists():
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.models.astro_cnn import AstroCNN
        model = AstroCNN(num_classes=10, channels=[32, 64, 128, 256], dropout_rate=0.0)
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        DEMO_MODE = False
        print("Real model loaded — Run 2 (gamma=3.0, Macro F1=0.607)")
    else:
        print("No checkpoint found — running in demo mode")
except Exception as e:
    print(f"Model load failed ({e}) — running in demo mode")


# ── Class definitions ─────────────────────────────────────────────────────────
CLASSES = [
    "disturbed", "merging", "round_smooth", "in_between", "cigar_shaped",
    "barred_spiral", "unbarred_tight_spiral", "unbarred_loose_spiral",
    "edge_on_no_bulge", "edge_on_with_bulge",
]

CLASS_INFO = {
    "disturbed":             {"label": "Disturbed",            "common": "Damaged Galaxy",          "description": "A galaxy showing gravitational disruption — asymmetric structure or tidal tails caused by a past interaction with another galaxy.", "fact": "The visible distortions may have been caused by a close encounter billions of years ago — the universe's memory of ancient collisions."},
    "merging":               {"label": "Merging",              "common": "Two Galaxies Colliding",  "description": "Two or more galaxies in active merger — visible as overlapping discs, double nuclei, or galaxies connected by bridges of stars.", "fact": "The Milky Way and Andromeda are on a collision course. They will merge in approximately 4.5 billion years."},
    "round_smooth":          {"label": "Round Smooth",         "common": "Elliptical Galaxy",       "description": "A smooth, circular galaxy with no discernible structure — typically a massive elliptical dominated by old stellar populations.", "fact": "Round smooth galaxies are effectively retired — star formation has ceased and they evolve passively, growing only through mergers."},
    "in_between":            {"label": "In-Between Smooth",    "common": "Elliptical Galaxy",       "description": "A smooth galaxy with a mildly elliptical rather than circular projected shape — intermediate between round ellipticals and edge-on discs.", "fact": "Projection effects dominate this class — the same disc galaxy appears round face-on and elongated edge-on."},
    "cigar_shaped":          {"label": "Cigar Shaped",         "common": "Elongated Galaxy",        "description": "A highly elongated smooth galaxy — either an edge-on lenticular or a strongly prolate elliptical seen along its minor axis.", "fact": "The rarest class in this dataset. The model achieves 90.2% recall here despite only 334 training examples — Focal Loss in action."},
    "barred_spiral":         {"label": "Barred Spiral",        "common": "Spiral Galaxy (Barred)",  "description": "A spiral galaxy with a linear bar of stars through its centre, from which the spiral arms emerge. The Milky Way belongs to this class.", "fact": "Galactic bars drive gas inflow toward the centre, fuelling both star formation bursts and supermassive black hole growth."},
    "unbarred_tight_spiral": {"label": "Tight Spiral",         "common": "Spiral Galaxy",           "description": "A spiral galaxy without a central bar, with arms wound tightly around the nucleus at a small pitch angle.", "fact": "Pitch angle — the tightness of the spiral — correlates with the mass of the central black hole, making it a cosmological ruler."},
    "unbarred_loose_spiral": {"label": "Loose Spiral",         "common": "Open Spiral Galaxy",      "description": "A spiral galaxy without a bar, with broadly sweeping arms at a large pitch angle — more open and diffuse than tight spirals.", "fact": "The hardest classification in this dataset. Even Galaxy Zoo citizen scientists showed high disagreement between loose and tight spirals."},
    "edge_on_no_bulge":      {"label": "Edge-On, No Bulge",    "common": "Disc Galaxy (Side View)", "description": "A disc galaxy viewed precisely edge-on, appearing as a thin luminous streak with no prominent central concentration.", "fact": "Edge-on orientation enables measurement of disc scale heights and reveals dust lanes completely hidden in face-on views."},
    "edge_on_with_bulge":    {"label": "Edge-On, With Bulge",  "common": "Disc Galaxy (Side View)", "description": "A disc galaxy viewed edge-on with a prominent spheroidal bulge at the centre — the classical lenticular or early-type spiral profile.", "fact": "NGC 4565, the Needle Galaxy, is the canonical example — a perfect flying-saucer profile visible from Earth."},
}


# ── Inference ─────────────────────────────────────────────────────────────────
def _demo_predict(image: Image.Image) -> dict:
    time.sleep(0.3)
    arr = np.array(image.convert("RGB").resize((128, 128))).astype(float) / 255.0
    brightness   = arr.mean()
    contrast     = arr.std()
    center       = arr[44:84, 44:84].mean()
    edge         = np.concatenate([arr[:16].ravel(), arr[-16:].ravel()]).mean()
    elongation   = abs(arr.mean(axis=0).std() - arr.mean(axis=1).std())
    center_ratio = center / (edge + 1e-6)
    scores = np.array([
        contrast * 1.5,
        contrast * brightness * 2.0,
        (1 - contrast) * center_ratio * 2.5,
        (1 - contrast) * 1.8,
        elongation * 3.0 + (1 - contrast),
        contrast * center_ratio * 1.8,
        center_ratio * contrast * 1.6,
        contrast * (1 - center_ratio) * 1.4,
        elongation * 2.5,
        elongation * center_ratio * 2.0,
    ])
    scores = np.clip(scores + np.random.uniform(0.05, 0.4, 10), 0.01, None)
    noise  = np.random.dirichlet(scores * 10)
    probs  = 0.65 * (scores / scores.sum()) + 0.35 * noise
    return dict(zip(CLASSES, (probs / probs.sum()).tolist()))


def _real_predict(image: Image.Image) -> dict:
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze(0).numpy()
    return dict(zip(CLASSES, probs.tolist()))


def classify(image):
    if image is None:
        return "", "", ""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    probs     = _demo_predict(image) if DEMO_MODE else _real_predict(image)
    predicted = max(probs, key=probs.get)
    confidence = probs[predicted]
    info      = CLASS_INFO[predicted]

    result = f"""### {info['label'].upper()}

**{confidence*100:.1f}% confidence**

{info['description']}

---
*{info['fact']}*
"""

    bars_html = _build_bars(probs, predicted)
    mode = "Pipeline simulation — upload model/best_model.pth for real inference" if DEMO_MODE else "Run 2 — gamma=3.0 — Macro F1: 0.607 — Macro AUC: 0.935"
    return result, bars_html, mode


def _build_bars(probs, predicted):
    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
    rows = []
    for cls, prob in sorted_probs:
        info   = CLASS_INFO[cls]
        is_top = cls == predicted
        pct    = prob * 100
        weight = "600" if is_top else "400"
        color  = "#e2e8f0" if is_top else "#64748b"
        bar_w  = f"{pct:.1f}%"
        bar_bg = "#3b82f6" if is_top else "#1e293b"
        rows.append(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                             color:{color};font-weight:{weight};letter-spacing:0.5px;">
                    {info['label'].upper()}
                </span>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;
                             color:{color};font-weight:{weight};">
                    {pct:.1f}%
                </span>
            </div>
            <div style="background:#0f172a;border-radius:2px;height:4px;overflow:hidden;">
                <div style="width:{bar_w};height:100%;background:{bar_bg};
                            border-radius:2px;transition:width 0.6s ease;"></div>
            </div>
        </div>""")

    return f"""
    <div style="background:#0a0f1e;border:1px solid #1e293b;border-radius:8px;
                padding:20px;font-family:'IBM Plex Mono',monospace;">
        <p style="font-size:10px;color:#334155;text-transform:uppercase;
                  letter-spacing:2px;margin:0 0 16px 0;">Classification Scores</p>
        {''.join(rows)}
    </div>"""


# ── Synthetic example images ──────────────────────────────────────────────────
def _make_example(cls: str) -> Image.Image:
    random.seed(hash(cls) % 999)
    arr = np.zeros((256, 256, 3), dtype=float)
    for _ in range(60):
        x, y = random.randint(0, 255), random.randint(0, 255)
        b = random.randint(80, 180)
        arr[y, x] = [b, b, b]

    if cls in ("round_smooth", "in_between"):
        ratio = 1.0 if cls == "round_smooth" else 0.5
        for r in range(55, 0, -1):
            a = int(200 * (1 - r/55)**1.5)
            ry = max(1, int(r * ratio))
            for dy in range(-ry, ry+1):
                for dx in range(-r, r+1):
                    if (dx/r)**2 + (dy/ry)**2 <= 1:
                        nx, ny = 128+dx, 128+dy
                        if 0 <= nx < 256 and 0 <= ny < 256:
                            arr[ny, nx] = np.minimum(arr[ny, nx] + [a*0.5, a*0.32, a*0.12], 255)

    elif cls == "cigar_shaped":
        for dx in range(-80, 80):
            for dy in range(-12, 12):
                if (dx/80)**2 + (dy/12)**2 <= 1:
                    nx, ny = 128+dx, 128+dy
                    if 0 <= nx < 256 and 0 <= ny < 256:
                        a = max(0, 200 - abs(dx)*1.5)
                        arr[ny, nx] = np.minimum(arr[ny, nx] + [a*0.6, a*0.4, a*0.15], 255)

    elif cls in ("barred_spiral", "unbarred_tight_spiral", "unbarred_loose_spiral"):
        pitch = 0.22 if cls == "unbarred_tight_spiral" else 0.5 if cls == "unbarred_loose_spiral" else 0.35
        if cls == "barred_spiral":
            for dx in range(-40, 40):
                for dy in range(-5, 5):
                    nx, ny = 128+dx, 128+dy
                    if 0 <= nx < 256 and 0 <= ny < 256:
                        a = max(0, 170 - abs(dx)*2)
                        arr[ny, nx] = np.minimum(arr[ny, nx] + [a*0.6, a*0.38, a*0.12], 255)
        for arm in range(2):
            for _ in range(1500):
                t = random.uniform(0, 4*np.pi)
                r = 12 + t * 12 * pitch + random.gauss(0, 3)
                angle = t + arm * np.pi
                x, y = int(128 + r*np.cos(angle)), int(128 + r*np.sin(angle)*0.85)
                if 0 <= x < 256 and 0 <= y < 256:
                    b = max(0, 160 - r*1.1)
                    arr[y, x] = np.minimum(arr[y, x] + [b*0.5, b*0.33, b*0.1], 255)

    elif cls in ("edge_on_no_bulge", "edge_on_with_bulge"):
        for dx in range(-85, 85):
            for dy in range(-4, 4):
                nx, ny = 128+dx, 128+dy
                if 0 <= nx < 256 and 0 <= ny < 256:
                    a = max(0, 170 - abs(dx)*1.1)
                    arr[ny, nx] = np.minimum(arr[ny, nx] + [a*0.5, a*0.33, a*0.1], 255)
        if cls == "edge_on_with_bulge":
            for r in range(20, 0, -1):
                a = int(220*(1-r/20)**1.5)
                for dy in range(-r, r+1):
                    for dx in range(-r, r+1):
                        if dx**2+dy**2 <= r**2:
                            nx, ny = 128+dx, 128+dy
                            if 0 <= nx < 256 and 0 <= ny < 256:
                                arr[ny, nx] = np.minimum(arr[ny, nx] + [a*0.55, a*0.36, a*0.1], 255)

    elif cls in ("disturbed", "merging"):
        centres = [(108, 118), (150, 138)] if cls == "merging" else [(128, 128)]
        for cx, cy in centres:
            for _ in range(1200):
                angle = random.uniform(0, 2*np.pi)
                r = random.gauss(0, 25)
                x, y = int(cx + r*np.cos(angle)), int(cy + r*np.sin(angle)*0.8)
                if 0 <= x < 256 and 0 <= y < 256:
                    b = random.randint(60, 170)
                    arr[y, x] = np.minimum(arr[y, x] + [b*0.5, b*0.32, b*0.1], 255)

    return Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))


EXAMPLE_IMAGES_DIR = Path("examples")
EXAMPLE_IMAGES_DIR.mkdir(exist_ok=True)
EXAMPLE_PATHS = []
for cls in CLASSES:
    path = EXAMPLE_IMAGES_DIR / f"{cls}.png"
    _make_example(cls).save(path)
    EXAMPLE_PATHS.append([str(path)])


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #050810;
    --surface:   #0a0f1e;
    --border:    #1a2035;
    --accent:    #3b82f6;
    --accent2:   #06b6d4;
    --text:      #e2e8f0;
    --muted:     #475569;
    --mono:      'IBM Plex Mono', monospace;
    --display:   'Syne', sans-serif;
    --body:      'DM Sans', sans-serif;
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--body) !important;
}

/* Header */
.app-header {
    padding: 56px 0 40px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 40px;
}
.app-header .eyebrow {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
}
.app-header h1 {
    font-family: var(--display);
    font-size: clamp(40px, 7vw, 72px);
    font-weight: 800;
    color: var(--text);
    letter-spacing: -3px;
    line-height: 1;
    margin: 0 0 20px;
}
.app-header h1 span { color: var(--accent); }
.app-header .subtitle {
    font-family: var(--body);
    font-size: 16px;
    color: var(--muted);
    font-weight: 300;
    max-width: 480px;
    line-height: 1.65;
    margin: 0 0 32px;
}

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    width: fit-content;
}
.stat-item {
    padding: 14px 28px;
    border-right: 1px solid var(--border);
    text-align: center;
}
.stat-item:last-child { border-right: none; }
.stat-value {
    font-family: var(--mono);
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    display: block;
}
.stat-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    display: block;
    margin-top: 2px;
}

/* Panels */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 24px;
}
.panel-label {
    font-family: var(--mono);
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
}

/* Classify button */
.classify-btn button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 6px !important;
    color: #fff !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 14px 0 !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.classify-btn button:hover {
    background: #2563eb !important;
}

/* Result markdown */
.result-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 28px;
    min-height: 160px;
}
.result-box h3 {
    font-family: var(--display) !important;
    font-size: 28px !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    letter-spacing: -1px !important;
    margin: 0 0 8px !important;
}
.result-box strong {
    font-family: var(--mono) !important;
    font-size: 13px !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}
.result-box p { color: #94a3b8 !important; line-height: 1.7 !important; font-size: 14px !important; }
.result-box em { color: var(--muted) !important; font-size: 13px !important; font-style: normal !important; }
.result-box hr { border-color: var(--border) !important; margin: 16px 0 !important; }

/* Mode bar */
.mode-bar {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    color: var(--muted) !important;
    letter-spacing: 1px !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Image upload */
.image-upload {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
}

/* Examples */
.gr-examples { margin-top: 16px !important; }

/* Footer */
.app-footer {
    border-top: 1px solid var(--border);
    margin-top: 48px;
    padding-top: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.app-footer p {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 1px;
    margin: 0;
}
.app-footer a { color: var(--accent); text-decoration: none; }

/* Hide gradio default footer */
footer { display: none !important; }
.gr-prose h3 { font-family: var(--display) !important; }

/* Textbox mode indicator */
textarea, input[type=text] {
    background: transparent !important;
    border: none !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 10px !important;
    resize: none !important;
}
"""

HEADER = """
<div class="app-header">
    <p class="eyebrow">Deep Learning &nbsp;&middot;&nbsp; Galaxy Morphology &nbsp;&middot;&nbsp; Galaxy10 DECals</p>
    <h1>Astro<span>Classifier</span></h1>
    <p class="subtitle">
        A convolutional neural network trained from scratch to classify galaxy morphology
        across 10 structural categories. Custom architecture &middot; Focal Loss &middot; 17,736 images.
    </p>
    <div class="stats-bar">
        <div class="stat-item">
            <span class="stat-value">0.607</span>
            <span class="stat-label">Macro F1</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">0.935</span>
            <span class="stat-label">Macro AUC</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">64.3%</span>
            <span class="stat-label">Accuracy</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">0.42M</span>
            <span class="stat-label">Parameters</span>
        </div>
        <div class="stat-item">
            <span class="stat-value">10</span>
            <span class="stat-label">Classes</span>
        </div>
    </div>
</div>
"""

FOOTER = """
<div class="app-footer">
    <p>AstroClassifier &nbsp;&middot;&nbsp; PyTorch &middot; Focal Loss &middot; WeightedRandomSampler</p>
    <p>
        <a href="https://github.com/YOUR_USERNAME/astro-classifier" target="_blank">GitHub</a>
        &nbsp;&nbsp;
        <a href="https://zenodo.org/records/10845026" target="_blank">Dataset</a>
    </p>
</div>
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="AstroClassifier", css=CSS) as demo:
    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            image_input = gr.Image(
                label="INPUT", type="numpy", height=300,
                elem_classes="image-upload"
            )
            classify_btn = gr.Button(
                "CLASSIFY", variant="primary",
                elem_classes="classify-btn"
            )
            gr.Examples(
                examples=EXAMPLE_PATHS,
                inputs=[image_input],
                label="EXAMPLE MORPHOLOGIES",
            )

        with gr.Column(scale=6):
            result_md  = gr.Markdown(
                value="Upload a galaxy image or select an example to begin classification.",
                elem_classes="result-box"
            )
            bars_html  = gr.HTML("")
            status_txt = gr.Textbox(
                value="",
                show_label=False,
                interactive=False,
                elem_classes="mode-bar",
                lines=1,
            )

    gr.HTML(FOOTER)

    classify_btn.click(
        fn=classify,
        inputs=[image_input],
        outputs=[result_md, bars_html, status_txt]
    )
    image_input.change(
        fn=classify,
        inputs=[image_input],
        outputs=[result_md, bars_html, status_txt]
    )

if __name__ == "__main__":
    demo.launch(show_api=False)