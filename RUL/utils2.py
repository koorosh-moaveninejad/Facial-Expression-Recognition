import torch
import cv2
import numpy as np
import random
import torch.nn as nn

# ─── image augmentation ───────────────────────────────────────────────────────

def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def filp_image(image_array):
    return cv2.flip(image_array, 1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ─── mixup ────────────────────────────────────────────────────────────────────

def mixup_data(x, y, att, use_cuda=True):
    """
    Original RUL mixup — random pairing, uncertainty as weights.
    Used by: baseline, label_smoothing, dynamic_loss, entropy_uncertainty
    """
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size, device=x.device)
    else:
        index = torch.randperm(batch_size)
    att1 = att / (att + att[index])
    att2 = att[index] / (att + att[index])
    mixed_x = att1 * x + att2 * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, att1, att2


def mixup_data_hard_negative(x, y, att, use_cuda=True):
    """
    EXPERIMENT: hard_negative_pairing
    Instead of random pairing, pair each sample with the most similar
    sample from a DIFFERENT class. This forces harder comparisons and
    should produce better-calibrated uncertainty values.

    How it works:
    - Compute cosine similarity between all pairs in the batch
    - For each sample i, find the most similar sample j where label_j != label_i
    - Use those as the mixup pairs instead of random ones

    Expected effect: harder comparisons → more discriminative uncertainty learning
    → potentially better accuracy on ambiguous classes (disgust, fear, contempt)
    """
    batch_size = x.size()[0]

    # Normalise features for cosine similarity
    x_norm = nn.functional.normalize(x, dim=1)             # (N, D)
    sim = torch.mm(x_norm, x_norm.T)                        # (N, N)

    # Mask out same-class pairs — we want different-class partners only
    same_class = y.unsqueeze(0) == y.unsqueeze(1)           # (N, N) bool
    sim[same_class] = -1e9                                   # ignore same class

    # For each sample, pick the most similar different-class sample
    index = sim.argmax(dim=1)                                # (N,)

    att1 = att / (att + att[index])
    att2 = att[index] / (att + att[index])
    mixed_x = att1 * x + att2 * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, att1, att2


# ─── loss functions ───────────────────────────────────────────────────────────

def mixup_criterion(y_a, y_b):
    """
    Original RUL loss — fixed 50/50 weighting of both labels.
    Used by: baseline, label_smoothing, entropy_uncertainty, hard_negative_pairing
    """
    return lambda criterion, pred: 0.5 * criterion(pred, y_a) + 0.5 * criterion(pred, y_b)


def mixup_criterion_dynamic(y_a, y_b, att1, att2):
    """
    EXPERIMENT: dynamic_loss
    Weight each label's loss by the OPPOSITE uncertainty value.

    Reasoning from the paper: att1 is image_i's uncertainty weight in the mix.
    A large att1 means image_i is MORE uncertain and dominates the mixed feature.
    So the loss for label_i should also be weighted more heavily — we want the
    model to focus on getting the uncertain sample's label right.

    Original paper says "add-up loss" but uses fixed 0.5/0.5 in code.
    This version is closer to the theoretical motivation.

    att1, att2 shape: (N, 1) — squeeze to (N,) for weighting
    """
    w1 = att1.squeeze().detach()   # weight for y_a loss
    w2 = att2.squeeze().detach()   # weight for y_b loss

    def loss_fn(criterion, pred):
        # Per-sample loss requires reduction='none'
        # We recreate criterion here with no reduction
        ce = nn.CrossEntropyLoss(reduction='none')
        loss_a = ce(pred, y_a)   # (N,)
        loss_b = ce(pred, y_b)   # (N,)
        return (w1 * loss_a + w2 * loss_b).mean()

    return loss_fn


# ─── uncertainty measures ─────────────────────────────────────────────────────

def compute_uncertainty_mean(logvar):
    """
    Original RUL: mean of exp(logvar) across embedding dimensions.
    Simple, fast, works well in practice.
    """
    return logvar.exp().mean(dim=1, keepdim=True)


def compute_uncertainty_entropy(logvar):
    """
    EXPERIMENT: entropy_uncertainty
    Treat softmax(logvar) as a probability distribution over embedding
    dimensions and compute its entropy.

    High entropy → uncertainty is spread across many dimensions → more uncertain
    Low entropy  → uncertainty is concentrated in few dimensions → more certain

    This is a richer measure because it captures the SPREAD of uncertainty
    across the feature space, not just the average magnitude.
    """
    probs = torch.softmax(logvar, dim=1)                          # (N, D)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=1, keepdim=True)  # (N, 1)
    return entropy


# ─── evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, fc, loader, device):
    model.eval()
    fc.eval()

    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    data_num = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for imgs, labels, indexes in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).long()

            features = model(imgs, labels, phase='test')
            outputs = fc(features)

            loss = criterion(outputs, labels)

            _, predicts = torch.max(outputs, 1)
            correct_sum += torch.eq(predicts, labels).sum().item()

            running_loss += loss.item()
            data_num += labels.size(0)
            iter_cnt += 1

    avg_loss = running_loss / iter_cnt
    acc = correct_sum / data_num
    return avg_loss, acc


def evaluate_with_predictions(model, fc, loader, device):
    model.eval()
    fc.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    data_num = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for imgs, labels, indexes in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).long()

            features = model(imgs, labels, phase='test')
            outputs = fc(features)

            loss = criterion(outputs, labels)
            _, predicts = torch.max(outputs, 1)

            correct_sum += torch.eq(predicts, labels).sum().item()
            running_loss += loss.item()
            data_num += labels.size(0)
            iter_cnt += 1

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicts.cpu().numpy().tolist())

    avg_loss = running_loss / iter_cnt
    acc = correct_sum / data_num
    return avg_loss, acc, y_true, y_pred