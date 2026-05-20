"""
로컬 Apple Silicon (MPS) 용 학습 스크립트
— Colab fracture_08_f1max.ipynb 의 경량 버전
— 실제 학습을 돌려서 .pth + .onnx + metadata.json 생성
  → 웹사이트 배포 테스트에 바로 사용 가능

특징:
- Macro-F1 기반 best-model 선택 & early stopping
- Stratified 80/10/10 split
- Focal + Label Smoothing + class weights
- Mixup (Phase2 에서만)
- Model EMA
- OneCycleLR warmup
- ONNX export (opset 17, dynamic batch)
"""
import os, sys, json, time, copy, random, math, pathlib, argparse, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from datetime import datetime
import timm
from PIL import Image

warnings.filterwarnings('ignore')
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

# ── 경로 ────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_DATA = "/Users/y3korea/Library/CloudStorage/GoogleDrive-y3korea@gmail.com/내 드라이브/2026_lecture/Medical_AI/Medical_Imagining/Bone Break Classification"
OUTPUT_BASE  = BASE_DIR / "output"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


class MultiExtImageFolder(ImageFolder):
    def is_valid_file(self, path):
        return pathlib.Path(path).suffix.lower() in IMG_EXTS


# ─────────────────────────────────────────────────────────
# Model helpers (timm 범용)
# ─────────────────────────────────────────────────────────
def _get_cls_attr(m): return m.default_cfg.get('classifier', 'classifier')

def _get_mod(m, path):
    for p in path.split('.'): m = getattr(m, p)
    return m

def _set_mod(m, path, mod):
    parts = path.split('.'); parent = m
    for p in parts[:-1]: parent = getattr(parent, p)
    setattr(parent, parts[-1], mod)


def replace_classifier(m, num_classes, drop=0.3):
    in_features = m.get_classifier().in_features
    new_head = nn.Sequential(nn.Dropout(drop),
                             nn.Linear(in_features, num_classes))
    _set_mod(m, _get_cls_attr(m), new_head)
    return new_head


def get_cls_params(m):
    return list(_get_mod(m, _get_cls_attr(m)).parameters())


def get_backbone_params(m, name):
    params = []
    if 'convnext' in name:
        params += list(m.stages[-1].parameters())
        params += list(m.stages[-2].parameters())
        if hasattr(m, 'head') and hasattr(m.head, 'norm'):
            params += list(m.head.norm.parameters())
    elif 'efficientnet' in name:
        params += list(m.blocks[-1].parameters())
        params += list(m.blocks[-2].parameters())
        params += list(m.conv_head.parameters())
        params += list(m.bn2.parameters())
    elif 'resnet' in name:
        params += list(m.layer4.parameters())
        params += list(m.layer3.parameters())
    elif 'mobilenet' in name:
        params += list(m.blocks[-1].parameters())
        params += list(m.blocks[-2].parameters())
        params += list(m.conv_head.parameters())
        params += list(m.bn2.parameters())
    return params


def build_model(name, num_classes, device):
    model = timm.create_model(name, pretrained=True, num_classes=num_classes)
    for p in model.parameters(): p.requires_grad = False
    replace_classifier(model, num_classes, drop=0.3)
    return model.to(device)


def unfreeze_backbone(model, name):
    for p in get_backbone_params(model, name):
        p.requires_grad = True


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


# ─────────────────────────────────────────────────────────
# Loss / Mixup / EMA
# ─────────────────────────────────────────────────────────
class FocalLabelSmoothLoss(nn.Module):
    def __init__(self, num_classes, gamma=1.5, smoothing=0.1,
                 class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer(
            'class_weights',
            class_weights if class_weights is not None
            else torch.ones(num_classes))

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        with torch.no_grad():
            tgt = torch.full_like(logp,
                                  self.smoothing / (self.num_classes - 1))
            tgt.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        pt = (p * tgt).sum(dim=-1).clamp_min(1e-8)
        focal = (1 - pt).pow(self.gamma)
        w = self.class_weights[targets]
        loss = -(tgt * logp).sum(dim=-1) * focal * w
        return loss.mean()


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_loss(crit, logits, ya, yb, lam):
    return lam * crit(logits, ya) + (1 - lam) * crit(logits, yb)


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
                    v.mul_(self.decay).add_(msd[k].detach().to(v.dtype),
                                            alpha=1 - self.decay)
                else:
                    v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()


# ─────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    probs_all, labels_all, loss_sum, n = [], [], 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        if criterion is not None:
            loss_sum += criterion(out, labels).item() * imgs.size(0)
        probs_all.append(F.softmax(out.float(), dim=1).cpu().numpy())
        labels_all.append(labels.cpu().numpy())
        n += imgs.size(0)
    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    preds = probs.argmax(axis=1)
    return dict(
        probs=probs, labels=labels, preds=preds,
        acc=(preds == labels).mean() * 100,
        f1_macro=f1_score(labels, preds, average='macro',
                          zero_division=0) * 100,
        f1_weighted=f1_score(labels, preds, average='weighted',
                             zero_division=0) * 100,
        loss=(loss_sum / n) if criterion is not None else None,
    )


@torch.no_grad()
def evaluate_tta(model, indices, tta_tfs, data_root, device,
                 batch_size, num_classes):
    model.eval()
    sum_probs, labels_out = None, None
    for tf in tta_tfs:
        ds = Subset(MultiExtImageFolder(root=data_root, transform=tf), indices)
        loader = DataLoader(ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)
        p_list, l_list = [], []
        for imgs, labels in loader:
            out = model(imgs.to(device))
            p_list.append(F.softmax(out.float(), dim=1).cpu().numpy())
            l_list.append(labels.numpy())
        probs = np.concatenate(p_list)
        if sum_probs is None:
            sum_probs, labels_out = probs, np.concatenate(l_list)
        else:
            sum_probs += probs
    sum_probs /= len(tta_tfs)
    return sum_probs.argmax(axis=1), labels_out, sum_probs


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--model', default='efficientnet_b0')
    ap.add_argument('--label', default='EfficientNet-B0')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--phase1', type=int, default=3)
    ap.add_argument('--phase2', type=int, default=12)
    ap.add_argument('--lr-head', type=float, default=1e-3)
    ap.add_argument('--lr-backbone', type=float, default=3e-5)
    ap.add_argument('--label-smooth', type=float, default=0.1)
    ap.add_argument('--focal-gamma', type=float, default=1.5)
    ap.add_argument('--no-focal', action='store_true')
    ap.add_argument('--mixup', type=float, default=0.2)
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--run-name-prefix', default='local')
    args = ap.parse_args()

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():   device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else: device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    # Seed
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    random.seed(args.seed)

    # Output
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = pathlib.Path(args.output_dir) if args.output_dir \
        else OUTPUT_BASE / f"{args.run_name_prefix}_{run_stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"⚙️  device={device} | model={args.label} | img={args.img_size} "
          f"| bs={args.batch_size}")
    print(f"    focal={not args.no_focal}(γ={args.focal_gamma}) "
          f"mixup={args.mixup} ema={args.ema} smooth={args.label_smooth}")
    print(f"    phase1={args.phase1} phase2={args.phase2} "
          f"lr_head={args.lr_head} lr_bb={args.lr_backbone}")
    print(f"📁 output: {output_dir}")

    # Dataset & stratified split
    full_ds = MultiExtImageFolder(root=args.data)
    class_names = full_ds.classes
    num_classes = len(class_names)
    targets = np.array(full_ds.targets)
    n = len(full_ds)

    idx_all = np.arange(n)
    idx_train, idx_rest, y_train, y_rest = train_test_split(
        idx_all, targets, test_size=0.2, stratify=targets,
        random_state=args.seed)
    idx_val, idx_test, _, _ = train_test_split(
        idx_rest, y_rest, test_size=0.5, stratify=y_rest,
        random_state=args.seed)
    train_idx = idx_train.tolist()
    val_idx   = idx_val.tolist()
    test_idx  = idx_test.tolist()

    print(f"📦 Classes({num_classes}): {class_names}")
    print(f"   Train {len(train_idx)} | Val {len(val_idx)} "
          f"| Test {len(test_idx)}")

    # Transforms
    IMG = args.img_size
    train_tf = transforms.Compose([
        transforms.Resize((int(IMG*1.1), int(IMG*1.1))),
        transforms.RandomCrop(IMG),
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
        transforms.Resize((IMG, IMG)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tta_tfs = [
        eval_tf,
        transforms.Compose([
            transforms.Resize((IMG, IMG)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((int(IMG*1.15), int(IMG*1.15))),
            transforms.CenterCrop(IMG),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((IMG, IMG)),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize((IMG, IMG)),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    ]

    # Sampler
    train_targets = targets[train_idx]
    cls_count = np.bincount(train_targets, minlength=num_classes).astype(np.float32)
    cls_w_np = 1.0 / np.maximum(cls_count, 1)
    sample_w = cls_w_np[train_targets]
    sampler = WeightedRandomSampler(torch.from_numpy(sample_w).double(),
                                     len(sample_w), replacement=True)

    train_ds = Subset(MultiExtImageFolder(root=args.data, transform=train_tf),
                      train_idx)
    val_ds   = Subset(MultiExtImageFolder(root=args.data, transform=eval_tf),
                      val_idx)
    test_ds  = Subset(MultiExtImageFolder(root=args.data, transform=eval_tf),
                      test_idx)

    persist = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=False, persistent_workers=persist)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=False, persistent_workers=persist)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=False)

    # Loss
    cls_w_tensor = torch.tensor(cls_w_np / cls_w_np.mean(),
                                dtype=torch.float32, device=device)
    if args.no_focal:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth,
                                        weight=cls_w_tensor)
    else:
        criterion = FocalLabelSmoothLoss(num_classes,
                                         gamma=args.focal_gamma,
                                         smoothing=args.label_smooth,
                                         class_weights=cls_w_tensor)

    # Model
    model = build_model(args.model, num_classes, device)
    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[],
                   val_f1_macro=[], val_f1_weighted=[], val_f1_ema=[],
                   phase=[])
    best_f1 = -1.0
    best_state = None
    best_source = 'raw'
    ema = None
    patience_counter = 0
    t0 = time.time()
    total_epochs = args.phase1 + args.phase2

    for epoch in range(1, total_epochs + 1):
        if epoch == 1:
            head_params = get_cls_params(model)
            optimizer = optim.AdamW(head_params, lr=args.lr_head,
                                    weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.phase1, 1))
            tp, trp = count_params(model)
            print(f"  📌 P1 Head | trainable={trp:,} ({trp/tp*100:.1f}%)")
        elif epoch == args.phase1 + 1:
            unfreeze_backbone(model, args.model)
            bb = get_backbone_params(model, args.model)
            head = get_cls_params(model)
            optimizer = optim.AdamW(
                [{'params': bb,   'lr': args.lr_backbone},
                 {'params': head, 'lr': args.lr_head * 0.3}],
                weight_decay=1e-4)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[args.lr_backbone * 3, args.lr_head * 0.3 * 3],
                steps_per_epoch=max(len(train_loader), 1),
                epochs=args.phase2, pct_start=0.1,
                anneal_strategy='cos')
            patience_counter = 0
            ema = ModelEMA(model, decay=args.ema)
            tp, trp = count_params(model)
            print(f"  📌 P2 Backbone+Head | trainable={trp:,} "
                  f"({trp/tp*100:.1f}%) | EMA on")

        phase = 1 if epoch <= args.phase1 else 2

        # Train
        model.train(); tloss, tok, tn = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if phase == 2 and args.mixup > 0:
                imgs, ya, yb, lam = mixup_data(imgs, labels, args.mixup)
            else:
                ya, yb, lam = labels, labels, 1.0
            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            if lam < 1.0:
                loss = mixup_loss(criterion, out, ya, yb, lam)
            else:
                loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0)
            optimizer.step()
            if phase == 2: scheduler.step()
            if ema is not None: ema.update(model)
            tloss += loss.item() * imgs.size(0)
            tok  += (out.argmax(1) == labels).sum().item()
            tn   += imgs.size(0)
        if phase == 1: scheduler.step()

        # Eval (raw)
        raw = evaluate(model, val_loader, device, criterion)
        tl, ta = tloss/tn, tok/tn*100
        vl, va = raw['loss'], raw['acc']
        f1m, f1w = raw['f1_macro'], raw['f1_weighted']

        # Eval (ema)
        f1_ema = float('nan')
        if ema is not None:
            ema_eval = evaluate(ema.ema, val_loader, device, criterion)
            f1_ema = ema_eval['f1_macro']

        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl); history['val_acc'].append(va)
        history['val_f1_macro'].append(f1m)
        history['val_f1_weighted'].append(f1w)
        history['val_f1_ema'].append(f1_ema)
        history['phase'].append(phase)

        cands = [('raw', f1m, model.state_dict())]
        if ema is not None:
            cands.append(('ema', f1_ema, ema.state_dict()))
        best_c = max(cands, key=lambda x: x[1])
        flag = ""
        if best_c[1] > best_f1:
            best_f1 = best_c[1]
            best_state = copy.deepcopy({k: v.detach().cpu()
                                        for k, v in best_c[2].items()})
            best_source = best_c[0]
            patience_counter = 0
            flag = f" ⭐({best_c[0]})"
        else:
            patience_counter += 1

        ema_str = f" | EMA-F1 {f1_ema:.1f}" if ema is not None else ""
        print(f"  P{phase} Ep {epoch:2d}/{total_epochs} | "
              f"Tr L{tl:.3f}/A{ta:.1f}% | Val L{vl:.3f}/A{va:.1f}% | "
              f"F1-m {f1m:.1f}/w {f1w:.1f}{ema_str}{flag}",
              flush=True)
        if phase == 2 and patience_counter >= args.patience:
            print(f"  ⏹️ Early stop (patience={args.patience}) "
                  f"best F1={best_f1:.2f}")
            break

    elapsed = time.time() - t0
    print(f"\n⏱️ train: {elapsed:.1f}s | best val F1={best_f1:.2f} "
          f"(from {best_source})")

    # Final eval on test (raw + TTA)
    final = build_model(args.model, num_classes, device)
    unfreeze_backbone(final, args.model)
    final.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final.eval()

    simple = evaluate(final, test_loader, device)
    tta_preds, tta_labels, _ = evaluate_tta(
        final, test_idx, tta_tfs, args.data, device,
        args.batch_size, num_classes)
    tta_acc  = (tta_preds == tta_labels).mean() * 100
    tta_f1m  = f1_score(tta_labels, tta_preds, average='macro',
                        zero_division=0) * 100
    tta_f1w  = f1_score(tta_labels, tta_preds, average='weighted',
                        zero_division=0) * 100
    prec, rec, f1, support = precision_recall_fscore_support(
        tta_labels, tta_preds, labels=list(range(num_classes)),
        zero_division=0)
    cm = confusion_matrix(tta_labels, tta_preds,
                          labels=list(range(num_classes)))

    print(f"\n🎯 Test raw: Acc {simple['acc']:.2f}% | "
          f"F1-m {simple['f1_macro']:.2f}% | F1-w {simple['f1_weighted']:.2f}%")
    print(f"🎯 Test TTA: Acc {tta_acc:.2f}% | "
          f"F1-m {tta_f1m:.2f}% | F1-w {tta_f1w:.2f}%")
    print("📋 per-class (TTA):")
    for i, c in enumerate(class_names):
        print(f"   {c:28s} P={prec[i]*100:5.1f} R={rec[i]*100:5.1f} "
              f"F1={f1[i]*100:5.1f} n={support[i]}")

    # Save .pth
    pth_path = output_dir / f"best_{args.model}.pth"
    torch.save(best_state, pth_path)
    print(f"\n💾 pth: {pth_path} ({os.path.getsize(pth_path)/1e6:.1f} MB)")

    # Save ONNX (CPU export — MPS ONNX export는 일부 op 미지원)
    final_cpu = build_model(args.model, num_classes, torch.device('cpu'))
    unfreeze_backbone(final_cpu, args.model)
    final_cpu.load_state_dict(best_state)
    final_cpu.eval()
    onnx_path = output_dir / f"best_{args.model}.onnx"
    dummy = torch.randn(1, 3, IMG, IMG)
    # dynamo=False → legacy TorchScript exporter: self-contained, CoreML 호환
    export_kwargs = dict(
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=17, do_constant_folding=True,
    )
    try:
        torch.onnx.export(final_cpu, dummy, onnx_path.as_posix(),
                          dynamo=False, **export_kwargs)
    except TypeError:
        # 구 버전 PyTorch는 dynamo 인자 없음
        torch.onnx.export(final_cpu, dummy, onnx_path.as_posix(),
                          **export_kwargs)
    print(f"💾 onnx: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.1f} MB)")

    # ONNX sanity
    import onnx, onnxruntime as ort
    onnx.checker.check_model(onnx.load(onnx_path.as_posix()))
    sess = ort.InferenceSession(onnx_path.as_posix(),
                                providers=['CPUExecutionProvider'])
    smoke = sess.run(None, {'input': dummy.numpy()})[0]
    print(f"   ORT smoke: logits shape={smoke.shape}")

    # Metadata
    metadata = {
        "model_name": args.model,
        "model_label": args.label,
        "img_size": IMG,
        "num_classes": num_classes,
        "class_names": class_names,
        "normalize_mean": IMAGENET_MEAN,
        "normalize_std":  IMAGENET_STD,
        "test_acc_simple": round(float(simple['acc']), 3),
        "test_f1_macro_simple":    round(float(simple['f1_macro']), 3),
        "test_acc_tta": round(float(tta_acc), 3),
        "test_f1_macro_tta":       round(float(tta_f1m), 3),
        "test_f1_weighted_tta":    round(float(tta_f1w), 3),
        "per_class_f1_tta": {class_names[i]: round(float(f1[i]*100), 2)
                             for i in range(num_classes)},
        "per_class_support": {class_names[i]: int(support[i])
                              for i in range(num_classes)},
        "tta_count": len(tta_tfs),
        "label_smoothing": args.label_smooth,
        "focal_gamma": None if args.no_focal else args.focal_gamma,
        "mixup_alpha":  args.mixup,
        "ema_decay":    args.ema,
        "best_source":  best_source,
        "run_timestamp": run_stamp,
        "train_samples": len(train_idx),
        "val_samples":   len(val_idx),
        "test_samples":  len(test_idx),
        "train_time_sec": round(elapsed, 1),
        "framework": "pytorch+timm",
        "onnx_opset": 17,
        "onnx_input_layout": "NCHW",
        "device": str(device),
        "preprocessing": {
            "resize": [IMG, IMG],
            "mean":   IMAGENET_MEAN,
            "std":    IMAGENET_STD,
            "color_space": "RGB",
        },
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(output_dir / 'training_history.json', 'w', encoding='utf-8') as f:
        json.dump({
            'history': history,
            'confusion_matrix_tta': cm.tolist(),
            'classification_report_tta': classification_report(
                tta_labels, tta_preds, target_names=class_names,
                digits=3, output_dict=True),
        }, f, indent=2, ensure_ascii=False)

    # Convenience symlink: output/latest → this run
    latest = OUTPUT_BASE / 'latest'
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(output_dir.name)
    except Exception as e:
        print(f"   (symlink skipped: {e})")

    print(f"\n🎉 완료! {output_dir}")
    print(f"   .pth + .onnx + metadata.json + training_history.json")


if __name__ == '__main__':
    main()
