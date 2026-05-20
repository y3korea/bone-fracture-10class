"""
fracture_08_f1max.ipynb 를 프로그램적으로 생성한다.
- Macro-F1 기반 best model selection / early stopping
- Stratified split
- Focal Loss + Label Smoothing
- Mixup augmentation
- Model EMA
- ConvNeXt-Tiny 1개 집중 학습 (시간 절약 + epoch 증가)
- ONNX export + metadata.json (웹 배포용)
- Per-class F1 로깅
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "fracture_08_f1max.ipynb"


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


cells = []

# ── 0. 마크다운: 설명 ───────────────────────────────────────
cells.append(md_cell("""# 🦴 Bone Fracture 10-Class — F1 Max v8 (Colab)

**v7.1 대비 개선점:**
- ✅ Stratified split (소수 클래스 보호)
- ✅ **Macro-F1 기반** best model & early stopping (val_acc 대신)
- ✅ Focal Loss + Label Smoothing (class imbalance 대응)
- ✅ Mixup augmentation
- ✅ Model EMA (Exponential Moving Average)
- ✅ ConvNeXt-Tiny 1개 집중 학습 (epoch 2배 증가)
- ✅ OneCycleLR + warmup
- ✅ **ONNX export** + `metadata.json` → 웹사이트 배포 바로 가능
- ✅ Per-class F1 로깅 (어떤 class가 약한지 바로 파악)
"""))

# ── 1. Drive mount ────────────────────────────────────────
cells.append(code_cell("""from google.colab import drive
drive.mount('/content/drive')
"""))

# ── 2. Setup + imports + config ───────────────────────────
cells.append(code_cell("""# ============================================================
#  🦴 v8 — F1 Max Training (Colab)
# ============================================================
!pip install -q timm onnx onnxruntime

import os, pathlib, time, json as _json, shutil, warnings, copy, math, random
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
from datetime import datetime
from IPython.display import display, HTML
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 100

# ── 경로 ────────────────────────────────────────────────
INPUT_PATH  = "/content/drive/MyDrive/2026_lecture/Medical_AI/Medical_Imagining/Bone Break Classification"
LOCAL_DATA  = "/content/bone_data"
OUTPUT_BASE = "/content/drive/MyDrive/2026_lecture/Medical_AI/1week/output"
RUN_STAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = os.path.join(OUTPUT_BASE, f"f1max_{RUN_STAMP}")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── 하이퍼파라미터 ────────────────────────────────────────
# 모델 1개에 집중 → epoch 증가 + LR 스케줄 정교화
MODEL_NAME     = "convnext_tiny"   # 이전 실험 best
MODEL_LABEL    = "ConvNeXt-Tiny"
IMG_SIZE       = 320               # 300 → 320 (fracture line은 디테일)
BATCH_SIZE     = 24                # IMG_SIZE 증가로 약간 감소
PHASE1_EPOCHS  = 5                 # head-only warmup
PHASE2_EPOCHS  = 30                # 15 → 30 (single-model 집중 학습)
LR_HEAD        = 1e-3
LR_BACKBONE    = 2e-5              # 3e-5 → 2e-5 (ConvNeXt는 LR 민감)
LABEL_SMOOTH   = 0.1
MIXUP_ALPHA    = 0.2               # 0이면 off
USE_FOCAL      = True              # Focal Loss on/off
FOCAL_GAMMA    = 1.5
EMA_DECAY      = 0.999
PATIENCE       = 8
SEED           = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.benchmark = True
print(f"⚡ Device: {DEVICE} | IMG: {IMG_SIZE} | AMP: ON | EMA: {EMA_DECAY}")
print(f"   Focal: {USE_FOCAL}(γ={FOCAL_GAMMA}) | Mixup α={MIXUP_ALPHA} | Smooth={LABEL_SMOOTH}")
print(f"   Phase1: {PHASE1_EPOCHS}ep (Head) → Phase2: {PHASE2_EPOCHS}ep (Backbone+Head)")

# ── 데이터 로컬 복사 ────────────────────────────────────────
if not os.path.exists(LOCAL_DATA):
    print("📂 Drive → Local SSD 복사 중...")
    shutil.copytree(INPUT_PATH, LOCAL_DATA)
    print("✅ 복사 완료!")
else:
    print("✅ 로컬 데이터 이미 존재")
"""))

# ── 3. Dataset + stratified split ──────────────────────────
cells.append(code_cell("""# ── 데이터셋 & Stratified split ──────────────────────────
IMG_EXTS = {'.png','.jpg','.jpeg','.bmp','.tiff','.tif','.webp'}

class MultiExtImageFolder(ImageFolder):
    def is_valid_file(self, path):
        return pathlib.Path(path).suffix.lower() in IMG_EXTS

class_names = sorted([d for d in os.listdir(LOCAL_DATA)
                      if os.path.isdir(os.path.join(LOCAL_DATA, d))])
NUM_CLASSES = len(class_names)
print(f"📂 Classes({NUM_CLASSES}): {class_names}")

full_ds = MultiExtImageFolder(root=LOCAL_DATA)
targets = np.array(full_ds.targets)
n = len(full_ds)

# Stratified: 80/10/10
idx_all = np.arange(n)
idx_train, idx_rest, y_train, y_rest = train_test_split(
    idx_all, targets, test_size=0.2, stratify=targets, random_state=SEED)
idx_val, idx_test, _, _ = train_test_split(
    idx_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED)

train_idx = idx_train.tolist()
val_idx   = idx_val.tolist()
test_idx  = idx_test.tolist()

# 클래스별 샘플 분포 체크
def count_per_class(indices):
    return np.bincount([targets[i] for i in indices], minlength=NUM_CLASSES)

train_counts = count_per_class(train_idx)
val_counts   = count_per_class(val_idx)
test_counts  = count_per_class(test_idx)
print(f"📦 Train:{len(train_idx)} | Val:{len(val_idx)} | Test:{len(test_idx)}")
print("   per-class (train/val/test):")
for i, c in enumerate(class_names):
    print(f"     {c:26s} {train_counts[i]:4d} / {val_counts[i]:3d} / {test_counts[i]:3d}")
total_images = int(train_counts.sum() + val_counts.sum() + test_counts.sum())
"""))

# ── 4. Transforms ──────────────────────────────────────────
cells.append(code_cell("""# ── Transforms ───────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((int(IMG_SIZE*1.1), int(IMG_SIZE*1.1))),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                            scale=(0.85, 1.15), shear=8),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.03),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
tta_transforms = [
    eval_tf,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((int(IMG_SIZE*1.15), int(IMG_SIZE*1.15))),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(10, 10)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=(-10, -10)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
]

# Sampler: WeightedRandomSampler (class-balanced)
train_targets = targets[train_idx]
cls_count = np.bincount(train_targets, minlength=NUM_CLASSES).astype(np.float32)
cls_weights = 1.0 / np.maximum(cls_count, 1)
sample_w = cls_weights[train_targets]
sampler = WeightedRandomSampler(torch.from_numpy(sample_w).double(),
                                 len(sample_w), replacement=True)

train_ds = Subset(MultiExtImageFolder(root=LOCAL_DATA, transform=train_tf), train_idx)
val_ds   = Subset(MultiExtImageFolder(root=LOCAL_DATA, transform=eval_tf), val_idx)
test_ds  = Subset(MultiExtImageFolder(root=LOCAL_DATA, transform=eval_tf), test_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
"""))

# ── 5. Loss + EMA + Mixup helpers ─────────────────────────
cells.append(code_cell("""# ── Focal Loss + Label Smoothing ─────────────────────────
class FocalLabelSmoothLoss(nn.Module):
    \"\"\"Focal 항과 label smoothing 을 결합한 loss.\"\"\"
    def __init__(self, num_classes, gamma=1.5, smoothing=0.1,
                 class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer('class_weights',
            class_weights if class_weights is not None else
            torch.ones(num_classes))

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=-1)
        p    = logp.exp()
        # label smoothing: soft one-hot
        with torch.no_grad():
            tgt = torch.full_like(logp, self.smoothing / (self.num_classes - 1))
            tgt.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        # focal weight: (1 - p_t)^gamma — only on true-class gives smooth result
        pt = (p * tgt).sum(dim=-1).clamp_min(1e-8)
        focal = (1 - pt).pow(self.gamma)
        w = self.class_weights[targets]
        loss = -(tgt * logp).sum(dim=-1) * focal * w
        return loss.mean()

# ── Mixup ────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

# ── Model EMA ────────────────────────────────────────────
class ModelEMA:
    \"\"\"Exponential moving average of model weights.\"\"\"
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad = False
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(self.decay).add_(msd[k].detach(),
                                            alpha=1 - self.decay)
                else:
                    v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()

# Class-balanced weights for loss (inverse frequency, normalized)
cls_w_tensor = torch.tensor(cls_weights / cls_weights.mean(),
                            dtype=torch.float32, device=DEVICE)
print(f"🧮 Class loss weights: "
      f"[{', '.join(f'{w:.2f}' for w in cls_w_tensor.cpu().numpy())}]")
"""))

# ── 6. Model builder (timm) ────────────────────────────────
cells.append(code_cell("""# ── timm 범용 classifier 교체 헬퍼 ──────────────────────────
def _get_classifier_attr(model):
    return model.default_cfg.get('classifier', 'classifier')

def _set_module_by_path(model, path, module):
    parts = path.split('.'); parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)

def _get_module_by_path(model, path):
    module = model
    for p in path.split('.'):
        module = getattr(module, p)
    return module

def replace_classifier(model, num_classes, drop=0.3):
    in_features = model.get_classifier().in_features
    new_head = nn.Sequential(nn.Dropout(drop),
                             nn.Linear(in_features, num_classes))
    _set_module_by_path(model, _get_classifier_attr(model), new_head)
    return new_head

def get_classifier_params(model):
    return list(_get_module_by_path(model, _get_classifier_attr(model)).parameters())

def get_backbone_params(model, name):
    params = []
    if 'convnext' in name:
        params += list(model.stages[-1].parameters())
        params += list(model.stages[-2].parameters())
        if hasattr(model, 'head') and hasattr(model.head, 'norm'):
            params += list(model.head.norm.parameters())
    elif 'efficientnet' in name:
        params += list(model.blocks[-1].parameters())
        params += list(model.blocks[-2].parameters())
        params += list(model.conv_head.parameters())
        params += list(model.bn2.parameters())
    elif 'resnet' in name:
        params += list(model.layer4.parameters())
        params += list(model.layer3.parameters())
    elif 'densenet' in name:
        params += list(model.features.denseblock4.parameters())
        params += list(model.features.norm5.parameters())
        params += list(model.features.denseblock3.parameters())
    return params

def build_model(name):
    model = timm.create_model(name, pretrained=True, num_classes=NUM_CLASSES)
    for param in model.parameters():
        param.requires_grad = False
    replace_classifier(model, NUM_CLASSES, drop=0.3)
    return model.to(DEVICE)

def unfreeze_backbone(model, name):
    for param in get_backbone_params(model, name):
        param.requires_grad = True

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train
"""))

# ── 7. Evaluation helpers (F1 focus) ──────────────────────
cells.append(code_cell("""# ── Evaluation ──────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    all_probs, all_labels, loss_sum, n = [], [], 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with autocast():
            out = model(imgs)
            if criterion is not None:
                loss = criterion(out, labels)
                loss_sum += loss.item() * imgs.size(0)
        all_probs.append(F.softmax(out.float(), dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        n += imgs.size(0)
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = probs.argmax(axis=1)
    acc    = (preds == labels).mean() * 100
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0) * 100
    f1_weighted = f1_score(labels, preds, average='weighted',
                           zero_division=0) * 100
    avg_loss = (loss_sum / n) if criterion is not None else None
    return dict(probs=probs, labels=labels, preds=preds,
                acc=acc, f1_macro=f1_macro, f1_weighted=f1_weighted,
                loss=avg_loss)

@torch.no_grad()
def evaluate_tta(model, indices):
    model.eval()
    sum_probs, labels_out = None, None
    for tta_tf in tta_transforms:
        ds = Subset(MultiExtImageFolder(root=LOCAL_DATA, transform=tta_tf), indices)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        probs_list, labels_list = [], []
        for imgs, labels in loader:
            with autocast():
                out = model(imgs.to(DEVICE))
            probs_list.append(F.softmax(out.float(), dim=1).cpu().numpy())
            labels_list.append(labels.numpy())
        probs = np.concatenate(probs_list)
        if sum_probs is None:
            sum_probs = probs
            labels_out = np.concatenate(labels_list)
        else:
            sum_probs += probs
    sum_probs /= len(tta_transforms)
    preds = sum_probs.argmax(axis=1)
    return preds, labels_out, sum_probs
"""))

# ── 8. Training loop ───────────────────────────────────────
cells.append(code_cell("""# ==============================================================
#  학습: Progressive (Phase1 head → Phase2 backbone+head)
#  + Mixup + EMA + Macro-F1 best selection
# ==============================================================
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

model = build_model(MODEL_NAME)
print(f"🧠 {MODEL_LABEL} classifier='{_get_classifier_attr(model)}'")

if USE_FOCAL:
    criterion = FocalLabelSmoothLoss(NUM_CLASSES, gamma=FOCAL_GAMMA,
                                     smoothing=LABEL_SMOOTH,
                                     class_weights=cls_w_tensor)
else:
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH,
                                    weight=cls_w_tensor)

scaler = GradScaler()
ema = None  # Phase2 진입 후 생성
history = {'train_loss': [], 'train_acc': [],
           'val_loss': [], 'val_acc': [],
           'val_f1_macro': [], 'val_f1_weighted': [],
           'val_f1_ema': [],
           'phase': []}
best_f1 = -1.0
best_state = None       # raw model
best_state_ema = None   # ema model (phase2 이후만)
best_source = "raw"     # "raw" | "ema"
patience_counter = 0
t0 = time.time()
total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS

for epoch in range(1, total_epochs + 1):
    if epoch == 1:
        head_params = get_classifier_params(model)
        optimizer = optim.AdamW(head_params, lr=LR_HEAD, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=PHASE1_EPOCHS)
        total_p, train_p = count_params(model)
        print(f"  📌 Phase 1 — Head only | Trainable: {train_p:,} "
              f"({train_p/total_p*100:.1f}%)")
    elif epoch == PHASE1_EPOCHS + 1:
        unfreeze_backbone(model, MODEL_NAME)
        backbone_params = get_backbone_params(model, MODEL_NAME)
        head_params = get_classifier_params(model)
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': LR_BACKBONE},
            {'params': head_params,     'lr': LR_HEAD * 0.3},
        ], weight_decay=1e-4)
        # OneCycleLR with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[LR_BACKBONE*3, LR_HEAD*0.3*3],
            steps_per_epoch=max(len(train_loader), 1),
            epochs=PHASE2_EPOCHS, pct_start=0.1, anneal_strategy='cos')
        patience_counter = 0
        ema = ModelEMA(model, decay=EMA_DECAY)
        total_p, train_p = count_params(model)
        print(f"  📌 Phase 2 — Backbone+Head | Trainable: {train_p:,} "
              f"({train_p/total_p*100:.1f}%) | EMA on")

    phase = 1 if epoch <= PHASE1_EPOCHS else 2

    # ── Train ──
    model.train(); t_loss, t_ok, t_n = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        # Mixup (Phase2 에서만 사용 — head-warmup 단계에선 off)
        if phase == 2 and MIXUP_ALPHA > 0:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
        else:
            y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out = model(imgs)
            if lam < 1.0:
                loss = mixup_loss(criterion, out, y_a, y_b, lam)
            else:
                loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        scaler.step(optimizer); scaler.update()

        if phase == 2:
            scheduler.step()  # OneCycleLR: per-step
        if ema is not None:
            ema.update(model)

        t_loss += loss.item() * imgs.size(0)
        t_ok += (out.argmax(1) == labels).sum().item()
        t_n += imgs.size(0)

    if phase == 1:
        scheduler.step()  # CosineAnnealing: per-epoch

    # ── Validate (raw model) ──
    raw_eval = evaluate(model, val_loader, criterion)
    tl, ta = t_loss/t_n, t_ok/t_n*100
    vl, va = raw_eval['loss'], raw_eval['acc']
    f1m, f1w = raw_eval['f1_macro'], raw_eval['f1_weighted']

    # ── Validate (EMA if available) ──
    f1_ema = None
    if ema is not None:
        ema_eval = evaluate(ema.ema, val_loader, criterion)
        f1_ema = ema_eval['f1_macro']

    history['train_loss'].append(tl); history['train_acc'].append(ta)
    history['val_loss'].append(vl); history['val_acc'].append(va)
    history['val_f1_macro'].append(f1m)
    history['val_f1_weighted'].append(f1w)
    history['val_f1_ema'].append(f1_ema if f1_ema is not None else float('nan'))
    history['phase'].append(phase)

    # best: raw vs ema 중 더 높은 F1 선택
    candidates = [('raw', f1m, model.state_dict())]
    if f1_ema is not None:
        candidates.append(('ema', f1_ema, ema.state_dict()))
    best_cand = max(candidates, key=lambda x: x[1])

    flag = ""
    if best_cand[1] > best_f1:
        best_f1 = best_cand[1]
        best_state = copy.deepcopy(best_cand[2])
        best_source = best_cand[0]
        patience_counter = 0
        flag = f" ⭐ ({best_cand[0]})"
    else:
        patience_counter += 1

    ema_str = f" | EMA-F1 {f1_ema:.1f}" if f1_ema is not None else ""
    print(f"  P{phase} Ep {epoch:2d}/{total_epochs} | "
          f"Tr L{tl:.3f}/A{ta:.1f}% | "
          f"Val L{vl:.3f}/A{va:.1f}% | "
          f"F1-m {f1m:.1f}/w {f1w:.1f}{ema_str}{flag}")

    if phase == 2 and patience_counter >= PATIENCE:
        print(f"  ⏹️ Early stop (patience={PATIENCE}) — best F1={best_f1:.2f}")
        break

elapsed = time.time() - t0
print(f"\\n⏱️ Training: {elapsed:.1f}s  |  Best Val Macro-F1: {best_f1:.2f}%  (from {best_source})")
"""))

# ── 9. Test + per-class F1 + save ─────────────────────────
cells.append(code_cell("""# ==============================================================
#  Test (일반 + TTA) + per-class F1
# ==============================================================
# best 상태 로드
final_model = build_model(MODEL_NAME)
unfreeze_backbone(final_model, MODEL_NAME)
final_model.load_state_dict(best_state)
final_model.eval()

# 일반 test
simple_eval = evaluate(final_model, test_loader)
# TTA test
tta_preds, tta_labels, tta_probs = evaluate_tta(final_model, test_idx)
tta_acc = (tta_preds == tta_labels).mean() * 100
tta_f1_macro = f1_score(tta_labels, tta_preds, average='macro',
                        zero_division=0) * 100
tta_f1_weighted = f1_score(tta_labels, tta_preds, average='weighted',
                           zero_division=0) * 100

# Per-class metrics (TTA)
prec, rec, f1, support = precision_recall_fscore_support(
    tta_labels, tta_preds, labels=list(range(NUM_CLASSES)), zero_division=0)

cm = confusion_matrix(tta_labels, tta_preds,
                      labels=list(range(NUM_CLASSES)))

print(f"\\n🎯 Test (raw):  Acc {simple_eval['acc']:.2f}% | "
      f"F1-macro {simple_eval['f1_macro']:.2f}% | "
      f"F1-weighted {simple_eval['f1_weighted']:.2f}%")
print(f"🎯 Test (TTA):  Acc {tta_acc:.2f}% | "
      f"F1-macro {tta_f1_macro:.2f}% | "
      f"F1-weighted {tta_f1_weighted:.2f}%")
print(f"\\n📋 Per-class (TTA):")
print(f"   {'class':28s} {'P':>6s} {'R':>6s} {'F1':>6s} {'n':>4s}")
for i, c in enumerate(class_names):
    print(f"   {c:28s} {prec[i]*100:6.1f} {rec[i]*100:6.1f} "
          f"{f1[i]*100:6.1f} {support[i]:4d}")
print(f"\\n{classification_report(tta_labels, tta_preds, target_names=class_names, digits=3)}")
"""))

# ── 10. Save artifacts: .pth + .onnx + metadata.json ──────
cells.append(code_cell("""# ==============================================================
#  💾 저장: .pth + .onnx + metadata.json (웹 배포용)
# ==============================================================
import onnx, onnxruntime as ort

# 1) PyTorch state_dict
pth_path = os.path.join(OUTPUT_PATH, f'best_{MODEL_NAME}.pth')
torch.save(best_state, pth_path)
print(f"💾 PyTorch: {pth_path}  ({os.path.getsize(pth_path)/1e6:.1f} MB)")

# 2) ONNX export
onnx_path = os.path.join(OUTPUT_PATH, f'best_{MODEL_NAME}.onnx')
final_model.eval().cpu()
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
export_kwargs = dict(
    input_names=['input'], output_names=['logits'],
    dynamic_axes={'input':  {0: 'batch'},
                  'logits': {0: 'batch'}},
    opset_version=17, do_constant_folding=True,
)
try:
    # legacy TorchScript exporter (self-contained, CoreML 호환)
    torch.onnx.export(final_model, dummy, onnx_path,
                      dynamo=False, **export_kwargs)
except TypeError:
    torch.onnx.export(final_model, dummy, onnx_path, **export_kwargs)
# ONNX 검증
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print(f"💾 ONNX:    {onnx_path}  ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

# 3) ONNX Runtime sanity check
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
ort_out = sess.run(None, {'input': dummy.numpy()})[0]
print(f"   ORT smoke test: logits shape={ort_out.shape}")

# 다시 DEVICE로 (이후 셀 사용 가능하도록)
final_model.to(DEVICE)

# 4) metadata.json — 웹 배포 필수
metadata = {
    "model_name": MODEL_NAME,
    "model_label": MODEL_LABEL,
    "img_size": IMG_SIZE,
    "num_classes": NUM_CLASSES,
    "class_names": class_names,
    "normalize_mean": IMAGENET_MEAN,
    "normalize_std":  IMAGENET_STD,
    "test_acc_simple": round(float(simple_eval['acc']), 3),
    "test_f1_macro_simple": round(float(simple_eval['f1_macro']), 3),
    "test_acc_tta": round(float(tta_acc), 3),
    "test_f1_macro_tta": round(float(tta_f1_macro), 3),
    "test_f1_weighted_tta": round(float(tta_f1_weighted), 3),
    "per_class_f1_tta": {class_names[i]: round(float(f1[i]*100), 2)
                         for i in range(NUM_CLASSES)},
    "per_class_support": {class_names[i]: int(support[i])
                          for i in range(NUM_CLASSES)},
    "tta_count": len(tta_transforms),
    "label_smoothing": LABEL_SMOOTH,
    "focal_gamma": FOCAL_GAMMA if USE_FOCAL else None,
    "mixup_alpha": MIXUP_ALPHA,
    "ema_decay": EMA_DECAY,
    "best_source": best_source,
    "run_timestamp": RUN_STAMP,
    "train_samples": len(train_idx),
    "val_samples":   len(val_idx),
    "test_samples":  len(test_idx),
    "train_time_sec": round(elapsed, 1),
    "framework": "pytorch+timm",
    "onnx_opset": 17,
    "onnx_input_layout": "NCHW",
    "preprocessing": {
        "resize": [IMG_SIZE, IMG_SIZE],
        "mean":   IMAGENET_MEAN,
        "std":    IMAGENET_STD,
        "color_space": "RGB",
    },
}
meta_path = os.path.join(OUTPUT_PATH, 'metadata.json')
with open(meta_path, 'w', encoding='utf-8') as f:
    _json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"💾 Metadata: {meta_path}")

# 5) History + confusion matrix JSON (시각화 재현용)
hist_path = os.path.join(OUTPUT_PATH, 'training_history.json')
with open(hist_path, 'w', encoding='utf-8') as f:
    _json.dump({
        'history': history,
        'confusion_matrix_tta': cm.tolist(),
        'classification_report_tta': classification_report(
            tta_labels, tta_preds, target_names=class_names,
            digits=3, output_dict=True),
    }, f, indent=2, ensure_ascii=False)
print(f"💾 History:  {hist_path}")
"""))

# ── 11. Visualizations ─────────────────────────────────────
cells.append(code_cell("""# ==============================================================
#  📊 시각화
# ==============================================================
short = [c.replace(" fracture","").replace(" Fracture","") for c in class_names]

# 학습 곡선
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
ep = range(1, len(history['val_f1_macro']) + 1)
axes[0].plot(ep, history['train_loss'], '-o', label='train', markersize=3)
axes[0].plot(ep, history['val_loss'],   '-o', label='val',   markersize=3)
axes[0].axvline(x=PHASE1_EPOCHS+0.5, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, history['val_acc'],      '-o', label='Val Acc', markersize=3)
axes[1].plot(ep, history['val_f1_macro'], '-o',
             label='Val F1-macro', markersize=3, linewidth=2)
axes[1].plot(ep, history['val_f1_weighted'], '--o',
             label='Val F1-weighted', markersize=3, alpha=0.7)
if any(not math.isnan(v) for v in history['val_f1_ema']):
    axes[1].plot(ep, history['val_f1_ema'], '-s',
                 label='EMA F1-macro', markersize=3)
axes[1].axvline(x=PHASE1_EPOCHS+0.5, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('Validation Accuracy & F1'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '01_curves.png'), dpi=150)
plt.show()

# Confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=short, yticklabels=short, ax=ax,
            linewidths=0.5, cbar=True)
ax.set_title(f"{MODEL_LABEL} — Confusion Matrix (TTA)\\n"
             f"Acc {tta_acc:.1f}% | F1-macro {tta_f1_macro:.1f}%",
             fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.xticks(rotation=30, ha='right'); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '02_confusion.png'), dpi=150)
plt.show()

# Per-class F1 bar
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#34A853' if v >= 60 else ('#FBBC04' if v >= 40 else '#EA4335')
          for v in f1*100]
bars = ax.bar(range(NUM_CLASSES), f1*100, color=colors, edgecolor='white')
for b, v, n_s in zip(bars, f1*100, support):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
            f'{v:.1f}%\\n(n={n_s})', ha='center', fontsize=9, fontweight='bold')
ax.set_xticks(range(NUM_CLASSES))
ax.set_xticklabels(short, rotation=30, ha='right')
ax.set_ylabel('F1 (%)')
ax.set_title(f'Per-Class F1 (TTA) — Macro: {tta_f1_macro:.1f}%',
             fontweight='bold')
ax.set_ylim(0, 105); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '03_per_class_f1.png'), dpi=150)
plt.show()

print(f"\\n🎉 완료! 📁 {OUTPUT_PATH}")
print(f"   웹 배포용 파일:")
print(f"     - best_{MODEL_NAME}.onnx")
print(f"     - metadata.json")
"""))

# ── 12. Inference demo ────────────────────────────────────
cells.append(code_cell("""# ==============================================================
#  🔬 업로드 추론 (ONNX Runtime 사용 — 웹과 동일 경로)
# ==============================================================
from google.colab import files
from PIL import Image
import onnxruntime as ort

print(f"🏆 {MODEL_LABEL} | TTA F1-macro: {tta_f1_macro:.2f}%")
sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

mean_t = np.array(IMAGENET_MEAN).reshape(1, 3, 1, 1).astype(np.float32)
std_t  = np.array(IMAGENET_STD).reshape(1, 3, 1, 1).astype(np.float32)

def preprocess(pil_img):
    pil_img = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE),
                                             Image.BILINEAR)
    arr = np.array(pil_img, dtype=np.float32) / 255.0  # HWC
    arr = arr.transpose(2, 0, 1)[None, ...]             # NCHW
    arr = (arr - mean_t) / std_t
    return arr.astype(np.float32)

print("📤 이미지를 업로드하세요...")
uploaded = files.upload()

for fname, data in uploaded.items():
    img = Image.open(fname)
    x = preprocess(img)
    logits = sess.run(None, {'input': x})[0][0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    pred_idx = int(probs.argmax())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(img); ax1.axis('off'); ax1.set_title(fname)
    order = np.argsort(probs)[::-1]
    clrs = ['#FF4444' if i == pred_idx else '#4285F4' for i in order]
    ax2.barh(range(NUM_CLASSES), probs[order]*100,
             color=clrs, edgecolor='white')
    ax2.set_yticks(range(NUM_CLASSES))
    ax2.set_yticklabels([class_names[i] for i in order], fontsize=9)
    for j, i in enumerate(order):
        ax2.text(probs[i]*100+0.5, j, f'{probs[i]*100:.1f}%',
                 va='center', fontweight='bold')
    ax2.set_xlim(0, 105); ax2.set_xlabel('Probability (%)')
    ax2.set_title(f"🦴 {class_names[pred_idx]} "
                  f"({probs[pred_idx]*100:.1f}%)", fontweight='bold',
                  color='#D32F2F' if probs[pred_idx]>0.7 else '#F57C00')
    plt.tight_layout(); plt.show(); print()
"""))

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
