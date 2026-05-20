"""
PyTorch (.pth) → ONNX 변환 스크립트.

학생들이 본인이 학습한 best_model.pth 를 앱에 탑재할 때 1회 실행한다.

기본 동작:
  cd app/tools
  python3 export_onnx.py
    → ../../output/run_*/best_model.pth 중 최신을 찾아
    → app/www/model.onnx + app/www/metadata.json 으로 저장

다른 run 디렉토리 지정:
  python3 export_onnx.py --run-dir ../../output/run_20260414_045350
"""
import argparse, json, pathlib, sys
import torch
import torch.nn as nn
from torchvision import models


def find_latest_run(output_base: pathlib.Path) -> pathlib.Path:
    cands = [p for p in output_base.glob("run_*")
             if p.is_dir() and (p / "best_model.pth").exists()
             and (p / "06_run_summary.json").exists()]
    if not cands:
        sys.exit(f"❌ {output_base} 안에 best_model.pth + 06_run_summary.json 가 있는 run_* 폴더를 못 찾았습니다.")
    return max(cands, key=lambda p: p.stat().st_mtime)


def build_model(num_classes: int) -> nn.Module:
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def main():
    here = pathlib.Path(__file__).resolve().parent
    proj = here.parent.parent  # 1week/
    default_out = (here.parent / "www").resolve()

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None,
                    help="output/run_* 디렉토리 경로 (생략하면 최신)")
    ap.add_argument("--out-dir", type=str, default=str(default_out),
                    help="model.onnx + metadata.json 출력 경로 (기본 app/www)")
    args = ap.parse_args()

    if args.run_dir:
        run_dir = pathlib.Path(args.run_dir).resolve()
    else:
        run_dir = find_latest_run(proj / "output")
    print(f"📂 run dir : {run_dir}")

    summary = json.loads((run_dir / "06_run_summary.json").read_text(encoding="utf-8"))
    img_size: int = summary["img_size"]
    class_names = summary["class_names"]
    num_classes = summary.get("num_classes", len(class_names))
    print(f"📐 img_size={img_size}  classes={num_classes}")

    # 모델 구축 + 가중치 로드
    model = build_model(num_classes)
    sd = torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=False)
    # checkpoint 가 dict-of-tensors 일 수도, {state_dict: ...} 일 수도 있음
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:    print("⚠️ missing keys :", missing[:5], "..." if len(missing) > 5 else "")
    if unexpected: print("⚠️ unexpect keys:", unexpected[:5], "..." if len(unexpected) > 5 else "")
    model.eval()

    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    meta_path = out_dir / "metadata.json"

    dummy = torch.randn(1, 3, img_size, img_size)
    # dynamo=False: 레거시 exporter (단일 파일, 광범위 호환). 신형 dynamo exporter는
    # 가중치를 외부 .data 파일로 분리해서 웹/모바일 배포가 까다로워진다.
    torch.onnx.export(
        model, dummy, onnx_path.as_posix(),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    # 이전 export 가 남긴 external-data 흔적 정리
    side = onnx_path.with_suffix(".onnx.data")
    if side.exists():
        side.unlink()
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"✅ saved : {onnx_path}  ({size_mb:.1f} MB)")

    meta = {
        "model_name": summary.get("model", "MobileNetV2"),
        "model_label": summary.get("model", "MobileNetV2"),
        "img_size": img_size,
        "class_names": class_names,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std":  [0.229, 0.224, 0.225],
        "source_run": run_dir.name,
        "test_acc_simple": summary.get("test_acc"),
        "best_val_acc": summary.get("best_val_acc"),
        "per_class_acc": summary.get("per_class_acc"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ saved : {meta_path}")


if __name__ == "__main__":
    main()
