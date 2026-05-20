"""
Bone Fracture Classifier — FastAPI inference server (ONNX Runtime 기반).

사용법:
    cd web && python3 app.py
    # 또는 uvicorn app:app --host 0.0.0.0 --port 8000 --reload

  자동으로 `../output/latest/*.onnx` + `metadata.json` 를 로드한다.
  다른 모델을 쓰려면 환경 변수 또는 CLI 인자로 지정:

    MODEL_DIR=/path/to/run_dir python3 app.py
"""
import os, io, json, time, pathlib
from typing import List, Dict, Any
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as ort

# ─────────────────────────────────────────────────────────
# 모델 디렉토리 탐색
# ─────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).resolve().parent
OUTPUT_BASE = (HERE.parent / "output").resolve()


def find_model_dir() -> pathlib.Path:
    """환경변수 MODEL_DIR → output/latest → output 안 가장 최근 f1max_ 디렉토리."""
    env = os.environ.get("MODEL_DIR")
    if env:
        p = pathlib.Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"MODEL_DIR not found: {p}")
        return p

    latest = OUTPUT_BASE / "latest"
    if latest.exists():
        return latest.resolve()

    # f1max_* 중 metadata.json 있는 것 중 가장 최근
    candidates = [p for p in OUTPUT_BASE.glob("*")
                  if p.is_dir() and (p / "metadata.json").exists()]
    if not candidates:
        raise FileNotFoundError(
            f"모델 디렉토리를 찾지 못했습니다. {OUTPUT_BASE} 안에 "
            "metadata.json 가 있는 run 폴더가 필요합니다."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


MODEL_DIR = find_model_dir()

# metadata 로드
META_PATH = MODEL_DIR / "metadata.json"
with META_PATH.open(encoding="utf-8") as f:
    META: Dict[str, Any] = json.load(f)

# ONNX 파일 탐색
onnx_files = list(MODEL_DIR.glob("*.onnx"))
if not onnx_files:
    raise FileNotFoundError(f".onnx 파일이 없습니다: {MODEL_DIR}")
ONNX_PATH = onnx_files[0]

CLASS_NAMES: List[str] = META["class_names"]
IMG_SIZE: int = META["img_size"]
MEAN = np.array(META["normalize_mean"]).reshape(1, 3, 1, 1).astype(np.float32)
STD  = np.array(META["normalize_std"]).reshape(1, 3, 1, 1).astype(np.float32)

# ONNX Runtime 세션 — 기본은 CPU (안정).
# USE_COREML=1 로 환경변수 설정 시에만 CoreML 시도 (호환 문제 있을 수 있음).
providers = ['CPUExecutionProvider']
if os.environ.get("USE_COREML") == "1":
    try:
        if 'CoreMLExecutionProvider' in ort.get_available_providers():
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    except Exception:
        pass
try:
    session = ort.InferenceSession(ONNX_PATH.as_posix(), providers=providers)
except Exception:
    session = ort.InferenceSession(ONNX_PATH.as_posix(),
                                    providers=['CPUExecutionProvider'])

print(f"🧠 model dir : {MODEL_DIR}")
print(f"📦 onnx      : {ONNX_PATH.name}  ({ONNX_PATH.stat().st_size/1e6:.1f} MB)")
print(f"🏷️  classes  : {CLASS_NAMES}")
print(f"📏 img size  : {IMG_SIZE}")
print(f"⚙️  providers: {session.get_providers()}")


# ─────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0       # HWC
    arr = arr.transpose(2, 0, 1)[None, ...]               # NCHW
    arr = (arr - MEAN) / STD
    return arr.astype(np.float32)


def preprocess_tta(img: Image.Image) -> List[np.ndarray]:
    """TTA: 원본 + hflip + 10% upscale-center-crop + ±10° rotation."""
    base = img.convert("RGB")
    W, H = base.size
    views: List[Image.Image] = []
    # 1) 원본
    views.append(base.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    # 2) H-flip
    views.append(base.transpose(Image.FLIP_LEFT_RIGHT)
                 .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    # 3) 10% upscale → center crop
    up = int(IMG_SIZE * 1.15)
    big = base.resize((up, up), Image.BILINEAR)
    l = (up - IMG_SIZE) // 2
    views.append(big.crop((l, l, l + IMG_SIZE, l + IMG_SIZE)))
    # 4/5) ±10° rotation
    for deg in (10, -10):
        r = base.resize((IMG_SIZE, IMG_SIZE),
                        Image.BILINEAR).rotate(deg, resample=Image.BILINEAR)
        views.append(r)
    batches = []
    for v in views:
        arr = np.asarray(v, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)[None, ...]
        arr = (arr - MEAN) / STD
        batches.append(arr.astype(np.float32))
    return batches


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ─────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────
app = FastAPI(title="Bone Fracture Classifier",
              description="ONNX 기반 10-class fracture 분류 데모",
              version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = HERE / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR.as_posix()),
          name="static")


@app.get("/")
def index():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(idx.as_posix())
    return HTMLResponse("<h1>index.html 없음</h1>")


@app.get("/api/info")
def info():
    return {
        "model_dir": str(MODEL_DIR),
        "onnx_file": ONNX_PATH.name,
        "model_name": META.get("model_name"),
        "model_label": META.get("model_label"),
        "img_size": IMG_SIZE,
        "class_names": CLASS_NAMES,
        "test_metrics": {
            "acc_simple": META.get("test_acc_simple"),
            "f1_macro_simple": META.get("test_f1_macro_simple"),
            "acc_tta": META.get("test_acc_tta"),
            "f1_macro_tta": META.get("test_f1_macro_tta"),
            "f1_weighted_tta": META.get("test_f1_weighted_tta"),
        },
        "per_class_f1_tta": META.get("per_class_f1_tta"),
        "per_class_support": META.get("per_class_support"),
        "providers": session.get_providers(),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), tta: bool = False):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, f"이미지 파일이 아닙니다: {file.content_type}")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"이미지 로드 실패: {e}")

    t0 = time.time()
    if tta:
        views = preprocess_tta(img)
        batch = np.concatenate(views, axis=0)       # (N,3,H,W)
        logits = session.run(None, {"input": batch})[0]
        probs = softmax(logits, axis=-1).mean(axis=0)
    else:
        x = preprocess(img)
        logits = session.run(None, {"input": x})[0][0]
        probs = softmax(logits, axis=-1)
    dt = (time.time() - t0) * 1000

    order = np.argsort(probs)[::-1]
    top = [{
        "class": CLASS_NAMES[i],
        "prob": float(probs[i]),
        "pct":  float(round(probs[i] * 100, 2)),
    } for i in order]

    return JSONResponse({
        "filename": file.filename,
        "pred_index": int(order[0]),
        "pred_class": CLASS_NAMES[int(order[0])],
        "pred_pct":   float(round(probs[int(order[0])] * 100, 2)),
        "probabilities": top,
        "tta_used": tta,
        "inference_ms": round(dt, 1),
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=port, reload=False)
