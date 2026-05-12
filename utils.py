import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


@dataclass
class TrainingHistory:
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_cifar10_datasets(data_root: str = "./data") -> Tuple[Dataset, Dataset, Dataset]:
    train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=build_transforms(True))
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=build_transforms(False))

    val_size = 5000
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=generator)

    return train_set, val_set, test_set


def get_cifar10_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
) -> DataBundle:
    train_set, val_set, test_set = get_cifar10_datasets(data_root=data_root)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return DataBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)


def denormalize_tensor(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=x.device).view(1, 3, 1, 1)
    return x * std + mean


def denormalize_image(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(0)
    out = denormalize_tensor(x).clamp(0, 1)
    return out[0]


def to_numpy_image(x: torch.Tensor) -> np.ndarray:
    if x.ndim == 4:
        x = x[0]
    x = denormalize_image(x).detach().cpu()
    return x.permute(1, 2, 0).numpy()


def make_image_grid(images: torch.Tensor, nrow: int = 8) -> torch.Tensor:
    return make_grid(denormalize_tensor(images), nrow=nrow, padding=2)


def show_image_grid(
    images: torch.Tensor,
    labels: Optional[Sequence[int]] = None,
    class_names: Sequence[str] = CIFAR10_CLASS_NAMES,
    nrow: int = 8,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> None:
    grid = make_image_grid(images, nrow=nrow)
    plt.figure(figsize=figsize)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        batch_size = targets.size(0)

        total_loss += loss.item()
        total_samples += batch_size
        all_logits.append(logits.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_probs.append(probs.detach().cpu())

    logits_np = torch.cat(all_logits).numpy()
    preds_np = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()
    probs_np = torch.cat(all_probs).numpy()
    avg_loss = total_loss / max(total_samples, 1)
    acc = float((preds_np == targets_np).mean())

    return {
        "loss": avg_loss,
        "acc": acc,
        "logits": logits_np,
        "preds": preds_np,
        "targets": targets_np,
        "probs": probs_np,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return {
        "loss": running_loss / max(total, 1),
        "acc": running_correct / max(total, 1),
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return {
        "loss": running_loss / max(total, 1),
        "acc": running_correct / max(total, 1),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    log_dir: str = "runs/cifar10_cnn",
    weights_path: str = "cifar10_cnn_best.pt",
    use_tensorboard: bool = True,
    max_grad_norm: Optional[float] = 5.0,
) -> TrainingHistory:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=log_dir) if (use_tensorboard and SummaryWriter is not None) else None

    history = TrainingHistory(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
    best_val_acc = -1.0
    best_state = None

    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, criterion, max_grad_norm=max_grad_norm)
        val_metrics = validate_one_epoch(model, val_loader, device, criterion)
        scheduler.step()

        history.train_loss.append(train_metrics["loss"])
        history.train_acc.append(train_metrics["acc"])
        history.val_loss.append(val_metrics["loss"])
        history.val_acc.append(val_metrics["acc"])

        if writer is not None:
            writer.add_scalar("loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("acc/train", train_metrics["acc"], epoch)
            writer.add_scalar("acc/val", val_metrics["acc"], epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            first_param = None
            for name, param in model.named_parameters():
                if param.requires_grad and param.ndim >= 2:
                    first_param = (name, param.detach().cpu())
                    break
            if first_param is not None:
                writer.add_histogram(f"weights/{first_param[0]}", first_param[1], epoch)

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": best_state,
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                },
                weights_path,
            )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    if writer is not None:
        writer.close()

    return history


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    logits_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    preds_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    images_all: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        logits_all.append(logits.cpu().numpy())
        probs_all.append(probs.cpu().numpy())
        preds_all.append(preds.cpu().numpy())
        targets_all.append(targets.numpy())
        images_all.append(images.cpu().numpy())

    return {
        "logits": np.concatenate(logits_all, axis=0),
        "probs": np.concatenate(probs_all, axis=0),
        "preds": np.concatenate(preds_all, axis=0),
        "targets": np.concatenate(targets_all, axis=0),
        "images": np.concatenate(images_all, axis=0),
    }


def compute_confusion_matrix(targets: np.ndarray, preds: np.ndarray, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str] = CIFAR10_CLASS_NAMES,
    normalize: bool = False,
    figsize: Tuple[int, int] = (8, 7),
) -> None:
    cm_plot = cm.astype(np.float64)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm_plot / row_sums

    plt.figure(figsize=figsize)
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title("Confusion matrix" + (" (normalized)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def plot_prediction_gallery(
    images: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    indices: Sequence[int],
    class_names: Sequence[str] = CIFAR10_CLASS_NAMES,
    ncols: int = 5,
    title: str = "Predictions",
) -> None:
    n = len(indices)
    nrows = math.ceil(n / ncols)
    plt.figure(figsize=(3.2 * ncols, 3.4 * nrows))
    for i, idx in enumerate(indices, start=1):
        img = np.transpose(np.clip(denormalize_array(images[idx]), 0, 1), (1, 2, 0))
        ax = plt.subplot(nrows, ncols, i)
        ax.imshow(img)
        ax.axis("off")
        t = class_names[int(targets[idx])]
        p = class_names[int(preds[idx])]
        c = float(probs[idx, int(preds[idx])])
        ax.set_title(f"t={t}\np={p}\n{c:.2f}", fontsize=9)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def denormalize_array(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 3:
        mean = np.array(CIFAR10_MEAN).reshape(3, 1, 1)
        std = np.array(CIFAR10_STD).reshape(3, 1, 1)
    elif images.ndim == 4:
        mean = np.array(CIFAR10_MEAN).reshape(1, 3, 1, 1)
        std = np.array(CIFAR10_STD).reshape(1, 3, 1, 1)
    else:
        raise ValueError("Expected a 3D or 4D tensor/array")
    return images * std + mean


def worst_prediction_indices(
    probs: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    k: int = 50,
) -> np.ndarray:
    correct_prob = probs[np.arange(len(targets)), targets]
    wrong = preds != targets
    candidate_idx = np.where(wrong)[0]
    if candidate_idx.size == 0:
        return np.array([], dtype=int)
    ranked = candidate_idx[np.argsort(correct_prob[candidate_idx])]
    return ranked[:k]


def plot_worst_images(
    images: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str] = CIFAR10_CLASS_NAMES,
    k: int = 50,
) -> np.ndarray:
    idx = worst_prediction_indices(probs, targets, preds, k=k)
    if idx.size == 0:
        print("No wrong predictions to display.")
        return idx
    plot_prediction_gallery(
        images=images,
        targets=targets,
        preds=preds,
        probs=probs,
        indices=idx,
        class_names=class_names,
        ncols=5,
        title=f"Worst {len(idx)} mistakes",
    )
    return idx


def plot_first_conv_filters(
    weight: torch.Tensor,
    nrow: int = 8,
    title: str = "First-layer filters",
) -> None:
    w = weight.detach().cpu()
    if w.ndim != 4:
        raise ValueError("Expected weight tensor with shape (out_channels, in_channels, H, W)")
    if w.size(1) not in (1, 3):
        raise ValueError("Expected 1 or 3 input channels for visualization")

    w = w.clone()
    w_min = w.amin(dim=(1, 2, 3), keepdim=True)
    w_max = w.amax(dim=(1, 2, 3), keepdim=True)
    w = (w - w_min) / (w_max - w_min + 1e-8)

    grid = make_grid(w, nrow=nrow, padding=1)
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.show()
