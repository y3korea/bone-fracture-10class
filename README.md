# Bone Fracture 10-Class Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)

End-to-end pipeline for **10-class bone-fracture radiograph classification**, accompanying the manuscripts submitted to *Diagnostics* (MDPI) and *Scientific Reports* (Nature). The repository covers:

- **Training**: F1-optimised pipeline (v8) and Optuna-driven AutoML (v9)
- **Server inference**: FastAPI + ONNX Runtime
- **On-device inference**: Cross-platform app (Android / iOS / macOS / Windows) built with Capacitor + Electron

---

## Repository layout

```
bone-fracture-10class/
├── fracture_08_f1max.ipynb       # v8 — F1-optimised single-model pipeline
├── fracture_09_automl.ipynb      # v9 — Optuna AutoML + Top-K ensemble
├── build_notebook_08.py          # Generator for v8 notebook
├── build_notebook_09.py          # Generator for v9 notebook
├── train_local.py                # Apple-Silicon (MPS) training script
├── requirements.txt              # Python dependencies (training + inference)
├── notebooks_legacy/             # Earlier course iterations (v1 – v7.1)
├── web/                          # FastAPI ONNX inference server
│   ├── app.py
│   ├── requirements.txt
│   └── static/index.html
├── app/                          # Capacitor + Electron cross-platform app
│   ├── README.md                 # Detailed build instructions (Android/iOS/macOS/Win)
│   ├── package.json
│   ├── capacitor.config.json
│   ├── www/                      # Web source bundled into native shells
│   ├── android/                  # Capacitor Android project (source only)
│   ├── ios/                      # Capacitor iOS project (source only)
│   ├── electron/                 # Electron desktop wrapper
│   └── tools/                    # ONNX export + ORT WASM copy scripts
└── output/
    └── run_20260414_045350/      # Reference checkpoint + metrics released with v1.0.0
        ├── best_model.pth
        ├── 06_run_summary.json
        └── *.png                 # Class distribution, training curves, confusion matrix
```

---

## Pipeline evolution (v7 → v8 → v9)

| Item | v7.1 | v8 (F1-max) | v9 (AutoML) |
|---|---|---|---|
| Best-model criterion | val accuracy | val **macro-F1** | val macro-F1 |
| Split | random shuffle | **stratified** | stratified |
| Loss | CE + smoothing | **Focal + smoothing + class weights** | same |
| Augmentation | standard | + **Mixup** (phase 2) | + Mixup (tuned α) |
| Weight averaging | — | **Model EMA** | Model EMA |
| Strategy | 5 models in parallel | **1 model, deep training** | **Optuna selects model + HPs** |
| Candidate models | EfficientNet / ResNet / DenseNet / ConvNeXt | ConvNeXt-Tiny | ConvNeXtV2 / MaxViT / EfficientNetV2 / SwinV2 / FastViT / TinyViT |
| Hyper-parameters | manual | manual | **Optuna TPE + MedianPruner** |
| Ensemble | — | — | **Top-3 trial ensemble** |
| Phase-2 scheduler | CosineAnnealingLR | OneCycleLR + warmup | same |
| Deployment | `.pth` | `.pth + .onnx + metadata.json` | + `optuna_study.pkl` |

---

## Quick start

### 1) Training in Colab (recommended — T4 GPU)

- `fracture_09_automl.ipynb` (AutoML, ~60–90 min) — best results
- `fracture_08_f1max.ipynb` (single model, ~30 min) — faster

### 2) Local training (Apple Silicon, MPS)

```bash
pip install -r requirements.txt

python train_local.py \
  --model convnext_tiny --label ConvNeXt-Tiny \
  --img-size 224 --batch-size 16 \
  --phase1 3 --phase2 15 --patience 6
```

Trained checkpoints land in `output/run_YYYYMMDD_HHMMSS/`; `output/latest` symlinks the most recent run.

### 3) Web inference server

```bash
pip install -r web/requirements.txt
cd web && python3 app.py
# → http://127.0.0.1:8000
```

Endpoints:

- `GET  /api/info` — model metadata + test metrics
- `POST /api/predict?tta=true` — upload an image, receive class probabilities

### 4) Mobile / desktop app (on-device ONNX)

See [`app/README.md`](app/README.md) for full build instructions. Summary:

```bash
cd app
npm install                       # also copies ORT WASM into www/ort
npm run export-model              # .pth → www/model.onnx + www/metadata.json

npm run android:build             # → android/app/build/outputs/apk/debug/app-debug.apk
npm run ios:open                  # → opens Xcode
npm run electron:build:mac        # → electron-dist/*.dmg
npm run electron:build:win        # → electron-dist/*.exe
```

The app inferences entirely on-device through ONNX Runtime Web (WASM) — no network calls.

---

## Reference results

Released checkpoint: `output/run_20260414_045350/best_model.pth` (MobileNetV2, image size 128, 3 epochs, baseline reference).

For the v8 / v9 results reported in the manuscripts, see Table 2 of the corresponding paper.

Local v8 single-model run on Apple Silicon MPS (ConvNeXt-Tiny, 18 epochs, ~3 min 43 s):

- Test accuracy (TTA): **51.33 %**
- Test macro-F1 (TTA): **49.69 %**
- Test weighted-F1 (TTA): **52.60 %**

---

## Data availability

The 10-class bone-fracture radiograph dataset used for training is **not redistributed** in this repository due to patient-privacy and IRB constraints. Requests for de-identified data should be addressed to the corresponding author. Researchers may reproduce the pipeline with any equivalently labelled radiograph dataset by placing it under `input/` with the expected ImageFolder structure:

```
input/
├── Avulsion fracture/
├── Comminuted fracture/
├── Fracture Dislocation/
├── Greenstick fracture/
├── Hairline Fracture/
├── Impacted fracture/
├── Longitudinal fracture/
├── Oblique fracture/
├── Pathological fracture/
└── Spiral Fracture/
```

---

## Citing this work

If you use this repository, please cite:

> Y. Ko, *Bone Fracture 10-Class Classifier — F1-Optimised Pipeline with AutoML and Cross-Platform Deployment*, GitHub repository, 2026. https://github.com/y3korea/bone-fracture-10class

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff). The accompanying journal manuscripts are listed in the *References* section of the GitHub release notes.

---

## License

Released under the [MIT License](LICENSE).

The repository **does not** include patient data; the licence covers source code, build scripts, the reference checkpoint, and exported ONNX models only.

---

## Acknowledgements

This pipeline was developed as part of a *Medical AI* graduate-course module (2026, Week 1). Contributions and issue reports are welcome.
