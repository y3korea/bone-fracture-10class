"""
fracture_09_automl.ipynb — 최소 셀 버전 (Colab T4).

구성:
  Cell 0 : Drive mount
  Cell 1 : 전체 파이프라인 (install → setup → HPO → Final → Ensemble → Test → Save)
  Cell 2 : 업로드 추론 (PyTorch + TTA)
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "fracture_09_automl.ipynb"

CELL0_SRC = """from google.colab import drive
drive.mount('/content/drive')
"""

CELL1_SRC = r'''# ================================================================
#  🦴 Bone Fracture 10-Class — v9 AutoML (Colab T4)
#  Optuna HPO (TPE + MedianPruner) + 최신 모델 후보 + Top-K 앙상블
#  Phase A (탐색) → Phase B (Final) → Phase C (Ensemble) → Save
# ================================================================
!pip install -q timm optuna

import os, pathlib, time, json as _json, shutil, warnings, copy, math, random, pickle
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import timm
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
from optuna.trial    import TrialState
from IPython.display import display, HTML
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 100

# ── 경로 / 시드 / 상수 ───────────────────────────────────────
# Drive 경로는 사용자마다 다를 수 있음 (MyDrive / My Drive / 내 드라이브 등)
# → 자동 탐색으로 'Bone Break Classification' 을 찾는다.
LOCAL_DATA = "/content/bone_data"

_CANDIDATES = [
    "/content/drive/MyDrive/2026_lecture/Medical_AI/Medical_Imagining/Bone Break Classification",
    "/content/drive/MyDrive/2026_lecture/Medical_AI/Medical_Imaging/Bone Break Classification",
    "/content/drive/My Drive/2026_lecture/Medical_AI/Medical_Imagining/Bone Break Classification",
    "/content/drive/My Drive/2026_lecture/Medical_AI/Medical_Imaging/Bone Break Classification",
    "/content/drive/내 드라이브/2026_lecture/Medical_AI/Medical_Imagining/Bone Break Classification",
    "/content/drive/내 드라이브/2026_lecture/Medical_AI/Medical_Imaging/Bone Break Classification",
]
INPUT_PATH = None
for _c in _CANDIDATES:
    if os.path.isdir(_c):
        INPUT_PATH = _c
        break
if INPUT_PATH is None:
    # 재귀 검색 (마지막 수단, 느림)
    import glob
    for _pat in ("/content/drive/**/Bone Break Classification",
                 "/content/drive/**/Bone*Break*Classification*"):
        _hits = glob.glob(_pat, recursive=True)
        _hits = [h for h in _hits if os.path.isdir(h)]
        if _hits:
            INPUT_PATH = _hits[0]; break
if INPUT_PATH is None:
    # 힌트 출력
    _roots = [p for p in ("/content/drive/MyDrive",
                          "/content/drive/My Drive",
                          "/content/drive/내 드라이브") if os.path.isdir(p)]
    _hint = ""
    if _roots:
        _hint = "\n  Drive 최상위: " + ", ".join(sorted(os.listdir(_roots[0]))[:20])
    raise FileNotFoundError(
        "❌ 'Bone Break Classification' 폴더를 찾지 못했습니다.\n"
        "  아래 후보 중에 없고 재귀 검색도 실패. 수동 지정 필요:\n"
        "    INPUT_PATH = '/content/drive/MyDrive/...'  (셀 상단)\n"
        + _hint
    )
print(f"📂 INPUT_PATH: {INPUT_PATH}")

# OUTPUT_BASE 도 Drive 루트 자동 탐지
_drv_root = INPUT_PATH.split("/2026_lecture/")[0]  # /content/drive/<root>
OUTPUT_BASE = os.path.join(_drv_root, "2026_lecture", "Medical_AI", "1week", "output")
RUN_STAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = os.path.join(OUTPUT_BASE, f"automl_{RUN_STAMP}")
os.makedirs(OUTPUT_PATH, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.benchmark = True
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
if torch.cuda.is_available():
    g = torch.cuda.get_device_properties(0)
    print(f"⚡ GPU: {g.name}  {g.total_memory/1e9:.1f} GB")
else:
    print("⚠️ CUDA 없음 — 런타임 > 유형 변경 > T4 GPU")
print(f"📁 {OUTPUT_PATH}")

# ── 최신 모델 후보 (T4 16GB 안전 batch size) ────────────────
MODEL_CATALOG = {
    "convnextv2_nano.fcmae_ft_in22k_in1k":   {"img_size": 224, "bs": 32, "label": "ConvNeXtV2-Nano"},
    "convnextv2_tiny.fcmae_ft_in22k_in1k":   {"img_size": 224, "bs": 24, "label": "ConvNeXtV2-Tiny"},
    "maxvit_tiny_tf_224.in1k":               {"img_size": 224, "bs": 20, "label": "MaxViT-Tiny"},
    "efficientnetv2_rw_s.ra2_in1k":          {"img_size": 288, "bs": 16, "label": "EffNetV2-RW-S"},
    "swinv2_tiny_window8_256.ms_in1k":       {"img_size": 256, "bs": 20, "label": "SwinV2-Tiny"},
    "fastvit_sa12.apple_dist_in1k":          {"img_size": 224, "bs": 32, "label": "FastViT-SA12"},
    "tiny_vit_21m_224.dist_in22k_ft_in1k":   {"img_size": 224, "bs": 24, "label": "TinyViT-21M"},
}
MODEL_KEYS = list(MODEL_CATALOG.keys())
print(f"🧠 후보 모델 {len(MODEL_KEYS)}개")

# ── 데이터 로컬 복사 ────────────────────────────────────────
if not os.path.exists(LOCAL_DATA):
    print("📂 Drive → SSD 복사 중...")
    shutil.copytree(INPUT_PATH, LOCAL_DATA)
print("✅ 데이터 준비 완료")

# ── Dataset + stratified split ─────────────────────────────
IMG_EXTS = {'.png','.jpg','.jpeg','.bmp','.tiff','.tif','.webp'}
class MultiExtImageFolder(ImageFolder):
    def is_valid_file(self, p):
        return pathlib.Path(p).suffix.lower() in IMG_EXTS

full_ds = MultiExtImageFolder(root=LOCAL_DATA)
class_names = full_ds.classes
NUM_CLASSES = len(class_names)
targets = np.array(full_ds.targets)
n = len(full_ds)

idx_all = np.arange(n)
idx_train, idx_rest, y_train, y_rest = train_test_split(
    idx_all, targets, test_size=0.2, stratify=targets, random_state=SEED)
idx_val, idx_test, _, _ = train_test_split(
    idx_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED)
train_idx = idx_train.tolist()
val_idx   = idx_val.tolist()
test_idx  = idx_test.tolist()
print(f"📦 Classes({NUM_CLASSES}): {class_names}")
print(f"   Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

# ── Transforms / loaders (모델별 img_size 지원) ─────────────
def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((int(img_size*1.1), int(img_size*1.1))),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.85, 1.15), shear=8),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.03),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(3, (0.1, 2.0)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tta_tfs = [eval_tf,
        transforms.Compose([transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([transforms.Resize((int(img_size*1.15), int(img_size*1.15))),
            transforms.CenterCrop(img_size), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([transforms.Resize((img_size, img_size)),
            transforms.RandomRotation((10, 10)), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        transforms.Compose([transforms.Resize((img_size, img_size)),
            transforms.RandomRotation((-10, -10)), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    ]
    return train_tf, eval_tf, tta_tfs

def build_loaders(img_size, batch_size, num_workers=2):
    train_tf, eval_tf, _ = build_transforms(img_size)
    tt = targets[train_idx]
    cc = np.bincount(tt, minlength=NUM_CLASSES).astype(np.float32)
    cls_w = 1.0 / np.maximum(cc, 1)
    sampler = WeightedRandomSampler(torch.from_numpy(cls_w[tt]).double(),
                                     len(tt), replacement=True)
    tr = Subset(MultiExtImageFolder(LOCAL_DATA, transform=train_tf), train_idx)
    vl = Subset(MultiExtImageFolder(LOCAL_DATA, transform=eval_tf),  val_idx)
    te = Subset(MultiExtImageFolder(LOCAL_DATA, transform=eval_tf),  test_idx)
    p = num_workers > 0
    return (DataLoader(tr, batch_size, sampler=sampler, num_workers=num_workers,
                       pin_memory=True, persistent_workers=p),
            DataLoader(vl, batch_size, shuffle=False, num_workers=num_workers,
                       pin_memory=True, persistent_workers=p),
            DataLoader(te, batch_size, shuffle=False, num_workers=num_workers,
                       pin_memory=True),
            cls_w)

# ── Loss / Mixup / EMA ──────────────────────────────────────
class FocalLabelSmoothLoss(nn.Module):
    def __init__(self, nc, gamma=1.5, smoothing=0.1, class_weights=None):
        super().__init__()
        self.nc, self.gamma, self.smoothing = nc, gamma, smoothing
        self.register_buffer('cw', class_weights if class_weights is not None
                              else torch.ones(nc))
    def forward(self, logits, y):
        logp = F.log_softmax(logits, dim=-1); p = logp.exp()
        with torch.no_grad():
            tgt = torch.full_like(logp, self.smoothing / (self.nc - 1))
            tgt.scatter_(1, y.unsqueeze(1), 1.0 - self.smoothing)
        pt = (p * tgt).sum(-1).clamp_min(1e-8)
        return -(tgt * logp).sum(-1).mul((1-pt).pow(self.gamma) * self.cw[y]).mean()

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_loss(crit, o, ya, yb, lam): return lam*crit(o, ya) + (1-lam)*crit(o, yb)

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters(): p.requires_grad = False
        self.decay = decay
    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(self.decay).add_(msd[k].detach(), alpha=1-self.decay)
                else: v.copy_(msd[k])
    def state_dict(self): return self.ema.state_dict()

# ── timm 범용 classifier / backbone 헬퍼 ────────────────────
def _cls_attr(m): return m.default_cfg.get('classifier', 'classifier')
def _get_mod(m, path):
    for p in path.split('.'): m = getattr(m, p)
    return m
def _set_mod(m, path, mod):
    parts = path.split('.'); parent = m
    for p in parts[:-1]: parent = getattr(parent, p)
    setattr(parent, parts[-1], mod)
def replace_classifier(m, nc, drop=0.3):
    in_f = m.get_classifier().in_features
    _set_mod(m, _cls_attr(m), nn.Sequential(nn.Dropout(drop), nn.Linear(in_f, nc)))
def get_cls_params(m): return list(_get_mod(m, _cls_attr(m)).parameters())

def get_backbone_params(m, name):
    params = []; n = name.lower()
    for cn in ('stages', 'blocks', 'layers'):
        if hasattr(m, cn):
            c = getattr(m, cn)
            try:
                params += list(c[-1].parameters())
                if len(c) >= 2: params += list(c[-2].parameters())
            except Exception: pass
            break
    if 'efficientnet' in n:
        if hasattr(m, 'conv_head'): params += list(m.conv_head.parameters())
        if hasattr(m, 'bn2'):       params += list(m.bn2.parameters())
    if 'fastvit' in n and hasattr(m, 'final_conv'):
        params += list(m.final_conv.parameters())
    if ('swin' in n or 'maxvit' in n) and hasattr(m, 'norm'):
        params += list(m.norm.parameters())
    if hasattr(m, 'head') and hasattr(m.head, 'norm'):
        params += list(m.head.norm.parameters())
    return params

def build_model(name, drop=0.3):
    m = timm.create_model(name, pretrained=True, num_classes=NUM_CLASSES)
    for p in m.parameters(): p.requires_grad = False
    replace_classifier(m, NUM_CLASSES, drop=drop)
    return m.to(DEVICE)
def unfreeze_backbone(m, name):
    for p in get_backbone_params(m, name): p.requires_grad = True
def count_params(m):
    t = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return t, tr

# ── 평가 함수 ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_f1(model, loader, criterion=None):
    model.eval(); probs_l, labels_l, ls, n_ = [], [], 0.0, 0
    for imgs, y in loader:
        imgs, y = imgs.to(DEVICE), y.to(DEVICE)
        with autocast():
            o = model(imgs)
            if criterion is not None: ls += criterion(o, y).item() * imgs.size(0)
        probs_l.append(F.softmax(o.float(), dim=1).cpu().numpy())
        labels_l.append(y.cpu().numpy()); n_ += imgs.size(0)
    pr = np.concatenate(probs_l); la = np.concatenate(labels_l)
    pd = pr.argmax(1)
    return dict(probs=pr, labels=la, preds=pd,
                acc=(pd==la).mean()*100,
                f1_macro=f1_score(la, pd, average='macro', zero_division=0)*100,
                f1_weighted=f1_score(la, pd, average='weighted', zero_division=0)*100,
                loss=(ls/n_) if criterion is not None else None)

# ── 통합 학습 함수 (HPO/Final 공용) ─────────────────────────
def train_one_run(cfg, trial=None, phase_ep=None, return_state=False, verbose=True):
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    train_loader, val_loader, _, cls_w = build_loaders(cfg['img_size'], cfg['batch_size'])
    cls_w_t = torch.tensor(cls_w / cls_w.mean(), dtype=torch.float32, device=DEVICE)
    criterion = FocalLabelSmoothLoss(NUM_CLASSES,
        gamma=cfg['focal_gamma'], smoothing=cfg['label_smoothing'], class_weights=cls_w_t)
    model = build_model(cfg['model_name'], drop=cfg['dropout'])
    scaler = GradScaler(); ema = None
    best_f1, best_state, best_source, pat = -1.0, None, 'raw', 0
    P1 = phase_ep[0] if phase_ep else cfg['phase1_ep']
    P2 = phase_ep[1] if phase_ep else cfg['phase2_ep']
    TOT = P1 + P2

    for epoch in range(1, TOT + 1):
        if epoch == 1:
            head = get_cls_params(model)
            optimizer = optim.AdamW(head, lr=cfg['lr_head'], weight_decay=cfg['weight_decay'])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(P1,1))
        elif epoch == P1 + 1:
            unfreeze_backbone(model, cfg['model_name'])
            bb = get_backbone_params(model, cfg['model_name']); head = get_cls_params(model)
            optimizer = optim.AdamW(
                [{'params': bb,   'lr': cfg['lr_backbone']},
                 {'params': head, 'lr': cfg['lr_head'] * 0.3}],
                weight_decay=cfg['weight_decay'])
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                max_lr=[cfg['lr_backbone']*3, cfg['lr_head']*0.3*3],
                steps_per_epoch=max(len(train_loader),1),
                epochs=max(P2,1), pct_start=0.1, anneal_strategy='cos')
            pat = 0
            ema = ModelEMA(model, decay=cfg['ema_decay'])

        phase = 1 if epoch <= P1 else 2
        model.train()
        for imgs, y in train_loader:
            imgs, y = imgs.to(DEVICE), y.to(DEVICE)
            if phase == 2 and cfg['mixup_alpha'] > 0:
                imgs, ya, yb, lam = mixup_data(imgs, y, cfg['mixup_alpha'])
            else:
                ya, yb, lam = y, y, 1.0
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                o = model(imgs)
                loss = mixup_loss(criterion, o, ya, yb, lam) if lam < 1.0 else criterion(o, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer); scaler.update()
            if phase == 2: scheduler.step()
            if ema is not None: ema.update(model)
        if phase == 1: scheduler.step()

        raw = evaluate_f1(model, val_loader, criterion)
        f1m = raw['f1_macro']
        f1_ema = None
        if ema is not None:
            f1_ema = evaluate_f1(ema.ema, val_loader).get('f1_macro')

        cands = [('raw', f1m, model.state_dict())]
        if ema is not None: cands.append(('ema', f1_ema, ema.state_dict()))
        bc = max(cands, key=lambda x: x[1])
        improved = bc[1] > best_f1
        if improved:
            best_f1 = bc[1]
            if return_state:
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in bc[2].items()})
            best_source = bc[0]; pat = 0
        else:
            pat += 1

        if verbose:
            ems = f" | EMA-F1 {f1_ema:.1f}" if f1_ema is not None else ""
            mk = f" ⭐({bc[0]})" if improved else ""
            print(f"  P{phase} Ep {epoch:2d}/{TOT} | Val Acc {raw['acc']:.1f}% "
                  f"| F1-m {f1m:.1f}/w {raw['f1_weighted']:.1f}{ems}{mk}")

        if trial is not None:
            trial.report(best_f1, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        if phase == 2 and pat >= cfg['patience']:
            if verbose: print(f"  ⏹️ early stop @ {epoch}")
            break

    del model
    if ema is not None: del ema
    torch.cuda.empty_cache()
    return dict(best_f1=float(best_f1), best_source=best_source, best_state=best_state)


# ================================================================
#  Phase A — Optuna HPO (탐색)
# ================================================================
N_TRIALS      = 18    # 시간 예산: T4에서 ~40-60 분
HPO_PHASE1_EP = 2
HPO_PHASE2_EP = 4
HPO_PATIENCE  = 3

def objective(trial):
    mn = trial.suggest_categorical("model_name", MODEL_KEYS)
    spec = MODEL_CATALOG[mn]
    cfg = {
        "model_name": mn,
        "img_size":   spec['img_size'],
        "batch_size": spec['bs'],
        "phase1_ep":  HPO_PHASE1_EP, "phase2_ep": HPO_PHASE2_EP,
        "lr_head":         trial.suggest_float("lr_head", 3e-4, 3e-3, log=True),
        "lr_backbone":     trial.suggest_float("lr_backbone", 1e-5, 1e-4, log=True),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "dropout":         trial.suggest_float("dropout", 0.1, 0.5),
        "mixup_alpha":     trial.suggest_float("mixup_alpha", 0.0, 0.4),
        "focal_gamma":     trial.suggest_float("focal_gamma", 0.5, 2.5),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "ema_decay":       0.999, "patience": HPO_PATIENCE,
    }
    print(f"\n🔬 Trial {trial.number} | {spec['label']} "
          f"| lr_h={cfg['lr_head']:.1e} lr_b={cfg['lr_backbone']:.1e} "
          f"drop={cfg['dropout']:.2f} mix={cfg['mixup_alpha']:.2f}")
    try:
        r = train_one_run(cfg, trial=trial, return_state=False, verbose=True)
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"  ❌ failed: {e}")
        raise optuna.exceptions.TrialPruned()
    return r['best_f1']

sampler = TPESampler(seed=SEED)
pruner  = MedianPruner(n_startup_trials=3, n_warmup_steps=2, interval_steps=1)
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                             study_name=f"fracture_{RUN_STAMP}")
t0 = time.time()
try:
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   catch=(RuntimeError,))
except KeyboardInterrupt:
    print("⏹️ 사용자 중단")
hpo_elapsed = time.time() - t0

done = [t for t in study.trials if t.state == TrialState.COMPLETE]
prun = [t for t in study.trials if t.state == TrialState.PRUNED]
fail = [t for t in study.trials if t.state == TrialState.FAIL]
best = study.best_trial
bspec = MODEL_CATALOG[best.params['model_name']]
print(f"\n{'='*60}\n✅ HPO {hpo_elapsed/60:.1f} min "
      f"| complete={len(done)} pruned={len(prun)} fail={len(fail)}")
print(f"🏆 Best trial #{best.number}: {bspec['label']}  val F1={best.value:.2f}")
for k, v in best.params.items(): print(f"   {k:18s}: {v}")

# ── HPO 시각화 ─────────────────────────────────────────────
try:
    from optuna.visualization.matplotlib import (plot_optimization_history,
                                                  plot_param_importances,
                                                  plot_parallel_coordinate)
    plot_optimization_history(study); plt.title("HPO History"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, '01_hpo_history.png'), dpi=140); plt.show()
    if len(study.trials) >= 5:
        try:
            plot_param_importances(study); plt.title("Param Importances"); plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_PATH, '02_hpo_importance.png'), dpi=140); plt.show()
        except Exception as e: print(f"importance skipped: {e}")
        try:
            plot_parallel_coordinate(study); plt.title("Parallel Coords"); plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_PATH, '03_hpo_parallel.png'), dpi=140); plt.show()
        except Exception as e: print(f"parallel skipped: {e}")
except Exception as e:
    print(f"viz err: {e}")

# Top trials HTML 표
topN = sorted(done, key=lambda t: -t.value)[:min(10, len(done))]
html = "<h3>🏆 Top Trials</h3><table style='border-collapse:collapse; font-size:12px; font-family:monospace;'>"
html += "<tr style='background:#4285F4;color:white;'>"
for h in ["rank","#","model","val F1","lr_h","lr_bb","drop","mix","γ","ls"]:
    html += f"<th style='padding:6px 10px;'>{h}</th>"
html += "</tr>"
for r, t in enumerate(topN, 1):
    bg = '#E8F5E9' if r == 1 else ('#FFF' if r % 2 == 0 else '#F5F5F5')
    mdl = ['🥇','🥈','🥉'][r-1] if r <= 3 else f" {r}"
    p = t.params
    html += (f"<tr style='background:{bg};'><td style='padding:5px 10px;text-align:center;'>{mdl}</td>"
             f"<td style='padding:5px 10px;'>{t.number}</td>"
             f"<td style='padding:5px 10px;'>{MODEL_CATALOG[p['model_name']]['label']}</td>"
             f"<td style='padding:5px 10px;text-align:right;font-weight:bold;'>{t.value:.2f}</td>"
             f"<td style='padding:5px 10px;'>{p['lr_head']:.1e}</td>"
             f"<td style='padding:5px 10px;'>{p['lr_backbone']:.1e}</td>"
             f"<td style='padding:5px 10px;'>{p['dropout']:.2f}</td>"
             f"<td style='padding:5px 10px;'>{p['mixup_alpha']:.2f}</td>"
             f"<td style='padding:5px 10px;'>{p['focal_gamma']:.2f}</td>"
             f"<td style='padding:5px 10px;'>{p['label_smoothing']:.3f}</td></tr>")
html += "</table>"
display(HTML(html))


# ================================================================
#  Phase B — Best config 로 Full Training
# ================================================================
FINAL_PHASE1_EP = 4
FINAL_PHASE2_EP = 25
FINAL_PATIENCE  = 8

bp = study.best_trial.params
bmn = bp['model_name']; bsp = MODEL_CATALOG[bmn]
final_cfg = {
    "model_name": bmn, "img_size": bsp['img_size'], "batch_size": bsp['bs'],
    "phase1_ep": FINAL_PHASE1_EP, "phase2_ep": FINAL_PHASE2_EP,
    "lr_head": bp['lr_head'], "lr_backbone": bp['lr_backbone'],
    "weight_decay": bp['weight_decay'], "dropout": bp['dropout'],
    "mixup_alpha": bp['mixup_alpha'], "focal_gamma": bp['focal_gamma'],
    "label_smoothing": bp['label_smoothing'],
    "ema_decay": 0.999, "patience": FINAL_PATIENCE,
}
print(f"\n🎯 Final: {bsp['label']}")
for k, v in final_cfg.items(): print(f"   {k:18s}: {v}")
t0 = time.time()
final_result = train_one_run(final_cfg, trial=None, return_state=True, verbose=True)
print(f"\n⏱️ Final {(time.time()-t0)/60:.1f} min | Best val F1 {final_result['best_f1']:.2f}")


# ================================================================
#  Phase C — Top-K 앙상블 (선택)
# ================================================================
DO_ENSEMBLE = True
K_ENSEMBLE  = 3
ENS_PHASE1_EP = 3
ENS_PHASE2_EP = 18

ensemble_models = []
if DO_ENSEMBLE:
    topk = sorted([t for t in study.trials if t.state == TrialState.COMPLETE],
                   key=lambda t: -t.value)[:K_ENSEMBLE]
    for i, t in enumerate(topk):
        p = t.params; sp = MODEL_CATALOG[p['model_name']]
        cfg = {
            "model_name": p['model_name'], "img_size": sp['img_size'], "batch_size": sp['bs'],
            "phase1_ep": ENS_PHASE1_EP, "phase2_ep": ENS_PHASE2_EP,
            "lr_head": p['lr_head'], "lr_backbone": p['lr_backbone'],
            "weight_decay": p['weight_decay'], "dropout": p['dropout'],
            "mixup_alpha": p['mixup_alpha'], "focal_gamma": p['focal_gamma'],
            "label_smoothing": p['label_smoothing'],
            "ema_decay": 0.999, "patience": 5,
        }
        print(f"\n{'='*60}\n🧩 Ensemble {i+1}/{K_ENSEMBLE}: {sp['label']} "
              f"(trial #{t.number}, val F1 {t.value:.2f})")
        if i == 0 and cfg['model_name'] == final_cfg['model_name'] \
           and abs(cfg['lr_head'] - final_cfg['lr_head']) < 1e-9:
            print("   (reusing Phase B result)")
            ensemble_models.append((cfg, final_result['best_state'], final_result['best_f1']))
            continue
        r = train_one_run(cfg, trial=None, return_state=True, verbose=True)
        ensemble_models.append((cfg, r['best_state'], r['best_f1']))
    print(f"\n✅ {len(ensemble_models)} 앙상블 준비")
else:
    print("⏭️ Phase C skipped")


# ================================================================
#  Test (일반 + TTA) + per-class F1 + 시각화
# ================================================================
@torch.no_grad()
def _tta_probs(model, indices, img_size, batch_size):
    _, _, tta_tfs = build_transforms(img_size)
    sp, lo = None, None
    for tf in tta_tfs:
        ds = Subset(MultiExtImageFolder(LOCAL_DATA, transform=tf), indices)
        ld = DataLoader(ds, batch_size, shuffle=False, num_workers=2, pin_memory=True)
        pl, ll = [], []
        for imgs, y in ld:
            with autocast(): o = model(imgs.to(DEVICE))
            pl.append(F.softmax(o.float(), dim=1).cpu().numpy())
            ll.append(y.numpy())
        pr = np.concatenate(pl)
        if sp is None: sp = pr; lo = np.concatenate(ll)
        else: sp = sp + pr
    sp /= len(tta_tfs); return sp, lo

print("\n=== Single Best Model (Phase B) ===")
m = build_model(final_cfg['model_name'], drop=final_cfg['dropout'])
unfreeze_backbone(m, final_cfg['model_name'])
m.load_state_dict({k: v.to(DEVICE) for k, v in final_result['best_state'].items()})
m.eval()
_, _, test_loader_f = build_loaders(final_cfg['img_size'], final_cfg['batch_size'])
simple = evaluate_f1(m, test_loader_f)
tta_probs_s, tta_labels = _tta_probs(m, test_idx, final_cfg['img_size'], final_cfg['batch_size'])
tta_preds_s = tta_probs_s.argmax(1)
tta_acc_s = (tta_preds_s == tta_labels).mean()*100
tta_f1m_s = f1_score(tta_labels, tta_preds_s, average='macro',    zero_division=0)*100
tta_f1w_s = f1_score(tta_labels, tta_preds_s, average='weighted', zero_division=0)*100
print(f"  Test raw: Acc {simple['acc']:.2f}% | F1-m {simple['f1_macro']:.2f}% "
      f"| F1-w {simple['f1_weighted']:.2f}%")
print(f"  Test TTA: Acc {tta_acc_s:.2f}% | F1-m {tta_f1m_s:.2f}% | F1-w {tta_f1w_s:.2f}%")

ensemble_tta_probs = None
if len(ensemble_models) > 1:
    print("\n=== Ensemble (Phase C) ===")
    for cfg, state, _ in ensemble_models:
        em = build_model(cfg['model_name'], drop=cfg['dropout'])
        unfreeze_backbone(em, cfg['model_name'])
        em.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}); em.eval()
        pr, _ = _tta_probs(em, test_idx, cfg['img_size'], cfg['batch_size'])
        ensemble_tta_probs = pr if ensemble_tta_probs is None else ensemble_tta_probs + pr
        del em; torch.cuda.empty_cache()
    ensemble_tta_probs /= len(ensemble_models)
    ep = ensemble_tta_probs.argmax(1)
    ea = (ep == tta_labels).mean()*100
    em_ = f1_score(tta_labels, ep, average='macro',    zero_division=0)*100
    ew_ = f1_score(tta_labels, ep, average='weighted', zero_division=0)*100
    print(f"  Ensemble TTA: Acc {ea:.2f}% | F1-m {em_:.2f}% | F1-w {ew_:.2f}%")
    if em_ > tta_f1m_s:
        winner_label, winner_preds = "Ensemble", ep
        winner_acc, winner_f1m, winner_f1w = ea, em_, ew_
    else:
        winner_label, winner_preds = "Single", tta_preds_s
        winner_acc, winner_f1m, winner_f1w = tta_acc_s, tta_f1m_s, tta_f1w_s
else:
    winner_label, winner_preds = "Single", tta_preds_s
    winner_acc, winner_f1m, winner_f1w = tta_acc_s, tta_f1m_s, tta_f1w_s

prec, rec, f1pc, support = precision_recall_fscore_support(
    tta_labels, winner_preds, labels=list(range(NUM_CLASSES)), zero_division=0)
cm = confusion_matrix(tta_labels, winner_preds, labels=list(range(NUM_CLASSES)))
print(f"\n🏆 Winner: {winner_label} | Acc {winner_acc:.2f}% "
      f"| F1-m {winner_f1m:.2f}% | F1-w {winner_f1w:.2f}%")
print(classification_report(tta_labels, winner_preds, target_names=class_names, digits=3))

# 시각화: confusion + per-class F1
short = [c.replace(" fracture","").replace(" Fracture","") for c in class_names]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=short, yticklabels=short, ax=ax, linewidths=0.5)
ax.set_title(f"{winner_label} — CM\nAcc {winner_acc:.1f}% | F1-m {winner_f1m:.1f}%",
             fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.xticks(rotation=30, ha='right'); plt.yticks(rotation=0)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_PATH, '04_confusion.png'), dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#34A853' if v >= 60 else ('#FBBC04' if v >= 40 else '#EA4335') for v in f1pc*100]
bars = ax.bar(range(NUM_CLASSES), f1pc*100, color=colors, edgecolor='white')
for b, v, n_s in zip(bars, f1pc*100, support):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
            f'{v:.1f}%\n(n={n_s})', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels(short, rotation=30, ha='right')
ax.set_ylabel('F1 (%)')
ax.set_title(f'Per-Class F1 — {winner_label} | Macro {winner_f1m:.1f}%', fontweight='bold')
ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_PATH, '05_per_class_f1.png'), dpi=150); plt.show()


# ================================================================
#  💾 저장 — .pth + metadata.json + optuna_trials.json
# ================================================================
pth_path = os.path.join(OUTPUT_PATH, f"best_{final_cfg['model_name'].split('.')[0]}.pth")
torch.save(final_result['best_state'], pth_path)
print(f"💾 pth: {pth_path} ({os.path.getsize(pth_path)/1e6:.1f} MB)")

if len(ensemble_models) > 1:
    ens_dir = os.path.join(OUTPUT_PATH, 'ensemble'); os.makedirs(ens_dir, exist_ok=True)
    for i, (cfg, state, best_f1) in enumerate(ensemble_models):
        p = os.path.join(ens_dir, f"member_{i}_{cfg['model_name'].split('.')[0]}.pth")
        torch.save(state, p)
        with open(os.path.join(ens_dir, f"member_{i}.json"), 'w', encoding='utf-8') as f:
            _json.dump({"model_name": cfg['model_name'], "img_size": cfg['img_size'],
                        "dropout": cfg['dropout'], "best_val_f1": best_f1},
                       f, indent=2, ensure_ascii=False)
    print(f"💾 ensemble: {ens_dir} ({len(ensemble_models)}개)")

metadata = {
    "model_name": final_cfg['model_name'],
    "model_label": MODEL_CATALOG[final_cfg['model_name']]['label'],
    "img_size": final_cfg['img_size'],
    "num_classes": NUM_CLASSES, "class_names": class_names,
    "normalize_mean": IMAGENET_MEAN, "normalize_std": IMAGENET_STD,
    "test_acc_simple":             round(float(simple['acc']), 3),
    "test_f1_macro_simple":        round(float(simple['f1_macro']), 3),
    "test_acc_tta_single":         round(float(tta_acc_s), 3),
    "test_f1_macro_tta_single":    round(float(tta_f1m_s), 3),
    "test_f1_weighted_tta_single": round(float(tta_f1w_s), 3),
    "test_acc_tta_winner":         round(float(winner_acc), 3),
    "test_f1_macro_tta_winner":    round(float(winner_f1m), 3),
    "test_f1_weighted_tta_winner": round(float(winner_f1w), 3),
    "winner": winner_label,
    "per_class_f1_tta":  {class_names[i]: round(float(f1pc[i]*100), 2) for i in range(NUM_CLASSES)},
    "per_class_support": {class_names[i]: int(support[i])             for i in range(NUM_CLASSES)},
    "final_config": final_cfg, "best_source": final_result['best_source'],
    "ensemble_size": len(ensemble_models),
    "run_timestamp": RUN_STAMP,
    "framework": "pytorch+timm+optuna",
    "preprocessing": {"resize": [final_cfg['img_size'], final_cfg['img_size']],
                       "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "color_space": "RGB"},
    "hpo": {"n_trials": N_TRIALS, "n_complete": len(done), "n_pruned": len(prun),
             "n_failed": len(fail), "best_trial_number": study.best_trial.number,
             "best_trial_value": round(float(study.best_value), 3),
             "hpo_elapsed_min": round(hpo_elapsed/60, 2)},
}
with open(os.path.join(OUTPUT_PATH, 'metadata.json'), 'w', encoding='utf-8') as f:
    _json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"💾 metadata: {os.path.join(OUTPUT_PATH, 'metadata.json')}")

with open(os.path.join(OUTPUT_PATH, 'optuna_trials.json'), 'w', encoding='utf-8') as f:
    _json.dump([{"number": t.number, "state": t.state.name,
                  "value": None if t.value is None else float(t.value),
                  "params": t.params} for t in study.trials],
                f, indent=2, ensure_ascii=False)

print(f"\n🎉 완료! winner={winner_label} | F1-macro(TTA) {winner_f1m:.2f}%")
print(f"📁 {OUTPUT_PATH}")
'''

CELL2_SRC = r'''# ================================================================
#  🔬 업로드 추론 (PyTorch + TTA, winner 사용) — 재실행 가능
# ================================================================
from google.colab import files
from PIL import Image

if winner_label == "Ensemble" and len(ensemble_models) > 1:
    inference_members = []
    for cfg, state, _ in ensemble_models:
        m = build_model(cfg['model_name'], drop=cfg['dropout'])
        unfreeze_backbone(m, cfg['model_name'])
        m.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}); m.eval()
        inference_members.append((m, cfg['img_size']))
    print(f"🎼 Ensemble: {len(inference_members)} members")
else:
    m = build_model(final_cfg['model_name'], drop=final_cfg['dropout'])
    unfreeze_backbone(m, final_cfg['model_name'])
    m.load_state_dict({k: v.to(DEVICE) for k, v in final_result['best_state'].items()}); m.eval()
    inference_members = [(m, final_cfg['img_size'])]
    print(f"🏆 Single: {MODEL_CATALOG[final_cfg['model_name']]['label']}")

print(f"   winner={winner_label} | F1-macro(TTA) {winner_f1m:.2f}%")
print("📤 이미지를 업로드하세요...")
uploaded = files.upload()

def predict_with_tta(pil_img):
    pil_img = pil_img.convert('RGB')
    total = 0; all_probs = None
    with torch.no_grad():
        for model, img_size in inference_members:
            _, _, tta_tfs = build_transforms(img_size)
            for tf in tta_tfs:
                x = tf(pil_img).unsqueeze(0).to(DEVICE)
                with autocast(): logits = model(x)
                p = F.softmax(logits.float(), dim=1).cpu().numpy()[0]
                all_probs = p if all_probs is None else all_probs + p
                total += 1
    return all_probs / total

for fname, data in uploaded.items():
    img = Image.open(fname)
    probs = predict_with_tta(img); pi = int(probs.argmax())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(img); ax1.axis('off'); ax1.set_title(fname, fontsize=12)
    order = np.argsort(probs)[::-1]
    clrs = ['#FF4444' if i == pi else '#4285F4' for i in order]
    ax2.barh(range(NUM_CLASSES), probs[order]*100, color=clrs, edgecolor='white')
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_yticklabels([class_names[i] for i in order], fontsize=9)
    for j, i in enumerate(order):
        ax2.text(probs[i]*100+0.5, j, f'{probs[i]*100:.1f}%',
                 va='center', fontweight='bold')
    ax2.set_xlim(0, 105); ax2.set_xlabel('Probability (%)')
    ax2.set_title(f"🦴 {class_names[pi]} ({probs[pi]*100:.1f}%)",
                  fontweight='bold', color='#D32F2F' if probs[pi] > 0.7 else '#F57C00')
    plt.tight_layout(); plt.show(); print()
'''


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


cells = [code_cell(CELL0_SRC), code_cell(CELL1_SRC), code_cell(CELL2_SRC)]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"OK: {NB_PATH}")
print(f"cells: {len(cells)}")
for i, c in enumerate(cells):
    lines = c['source']
    n_lines = len(lines)
    n_chars = sum(len(l) for l in lines)
    print(f"  [{i}] {n_lines} lines, {n_chars:,} chars")
