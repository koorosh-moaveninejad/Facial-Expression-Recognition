import argparse
import csv
import json
import os


import time
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from dataset import RafDataset
from rul2 import res18feature
from utils2 import *

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FERPLUS_DIR = os.path.join(BASE_DIR, 'FERplus')

parser = argparse.ArgumentParser()

parser.add_argument('--raf_path', type=str, default=FERPLUS_DIR)
parser.add_argument('--train_label_path', type=str, default=os.path.join(FERPLUS_DIR, 'train_labels.csv'))
parser.add_argument('--test_label_path',  type=str, default=os.path.join(FERPLUS_DIR, 'test_labels.csv'))

parser.add_argument('--pretrained_backbone_path', type=str, default='resnet18_msceleb.pth')
parser.add_argument('--workers',       type=int, default=4)
parser.add_argument('--batch_size',    type=int, default=32)
parser.add_argument('--epochs',        type=int, default=30)
parser.add_argument('--out_dimension', type=int, default=64)

# ── Experiment selector ────────────────────────────────────────────────────────
# Run one experiment at a time and compare results.
#
#   baseline            — original RUL unchanged
#   label_smoothing     — adds label smoothing (0.1) to CrossEntropyLoss
#   dynamic_loss        — weights each label's loss by its uncertainty value
#                         instead of fixed 0.5 / 0.5
#   entropy_uncertainty — uses entropy of logvar distribution as uncertainty
#                         measure instead of mean variance
#   hard_negative       — pairs each sample with most similar different-class
#                         sample instead of random pairing
#
# Usage examples:
#   python main.py --experiment baseline
#   python main.py --experiment label_smoothing
#   python main.py --experiment dynamic_loss
#   python main.py --experiment entropy_uncertainty
#   python main.py --experiment hard_negative
# ──────────────────────────────────────────────────────────────────────────────
parser.add_argument(
    '--experiment',
    type=str,
    default='baseline',
    choices=['baseline', 'label_smoothing', 'dynamic_loss', 'entropy_uncertainty', 'hard_negative'],
    help='Which experiment variant to run'
)

args = parser.parse_args()


def train():
    setup_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using: {device}")
    print(f"Experiment: {args.experiment}")

    # Save outputs in experiment-specific subfolders so results don't overwrite
    checkpoints_dir = f"../checkpoints/{args.experiment}"
    reports_dir     = f"../reports/{args.experiment}"
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(reports_dir,     exist_ok=True)

    with open(f"{reports_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    metrics_path = f"{reports_dir}/metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss",
            "train_eval_loss", "train_eval_acc",
            "test_loss", "test_acc",
            "acc_gap", "loss_gap",
            "lr", "epoch_time_sec", "best_test_acc_so_far"
        ])

    # ── Model ─────────────────────────────────────────────────────────────────
    res18 = res18feature(args)   # experiment is passed via args.experiment
    fc    = nn.Linear(args.out_dimension, 8)

    # ── Transforms ────────────────────────────────────────────────────────────
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25))
    ])

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset      = RafDataset(args, phase='train', transform=data_transforms)
    train_dataset_eval = RafDataset(args, phase='train', basic_aug=False, transform=data_transforms_val)
    test_dataset       = RafDataset(args, phase='test',  transform=data_transforms_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=False
    )
    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset_eval, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=(device.type == 'cuda')
    )

    res18 = res18.to(device)
    fc    = fc.to(device)

    optimizer = torch.optim.Adam([
        {'params': res18.parameters()},
        {'params': fc.parameters(), 'lr': 0.002}
    ], lr=0.0002, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # ── Loss function ─────────────────────────────────────────────────────────
    # label_smoothing experiment: smoothing=0.1 reduces overconfidence on noisy
    # FERPlus labels. All other experiments use standard CE.
    if args.experiment == 'label_smoothing':
        base_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("Using label smoothing = 0.1")
    else:
        base_criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_acc   = 0.0
    best_epoch = 0

    for i in range(1, args.epochs + 1):
        epoch_start  = time.time()
        running_loss = 0.0
        iter_cnt     = 0

        res18.train()
        fc.train()

        train_bar = tqdm(train_loader, desc=f"Epoch {i}/{args.epochs} [{args.experiment}]")

        for batch_i, (imgs, labels, indexes) in enumerate(train_bar):
            imgs   = imgs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            mixed_x, y_a, y_b, att1, att2 = res18(imgs, labels, phase='train')
            outputs = fc(mixed_x)

            # ── Loss selection ─────────────────────────────────────────────
            if args.experiment == 'dynamic_loss':
                # EXPERIMENT: weight each label's loss by its uncertainty value
                # att1 = uncertainty weight of image_i in the mix
                # att2 = uncertainty weight of image_j in the mix
                # More uncertain image should contribute more to the loss
                loss_fn = mixup_criterion_dynamic(y_a, y_b, att1, att2)
                loss = loss_fn(base_criterion, outputs)

            else:
                # BASELINE (and all other experiments): fixed 0.5 / 0.5
                loss_fn = mixup_criterion(y_a, y_b)
                loss = loss_fn(base_criterion, outputs)

            loss.backward()
            optimizer.step()

            iter_cnt     += 1
            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        running_loss /= iter_cnt

        train_eval_loss, train_eval_acc = evaluate(res18, fc, train_eval_loader, device)
        test_loss, test_acc             = evaluate(res18, fc, test_loader,       device)

        acc_gap    = train_eval_acc - test_acc
        loss_gap   = test_loss - train_eval_loss
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        print(f'Epoch {i}: train_loss={running_loss:.4f}')
        print(f'Epoch {i}: train_eval_acc={train_eval_acc:.4f}  train_eval_loss={train_eval_loss:.4f}')
        print(f'Epoch {i}: test_acc={test_acc:.4f}  test_loss={test_loss:.4f}')
        print(f'Epoch {i}: acc_gap={acc_gap:.4f}  loss_gap={loss_gap:.4f}')

        # Save every epoch checkpoint
        torch.save({
            'model_state_dict': res18.state_dict(),
            'fc_state_dict':    fc.state_dict(),
            'epoch':            i,
            'test_acc':         test_acc
        }, f'{checkpoints_dir}/epoch_{i}_acc_{test_acc:.4f}.pth')

        # Save latest checkpoint (overwrites each epoch)
        torch.save({
            'model_state_dict': res18.state_dict(),
            'fc_state_dict':    fc.state_dict(),
            'epoch':            i,
            'test_acc':         test_acc
        }, f'{checkpoints_dir}/last_model.pth')

        # Save best checkpoint
        if test_acc > best_acc:
            best_acc   = test_acc
            best_epoch = i
            torch.save({
                'model_state_dict': res18.state_dict(),
                'fc_state_dict':    fc.state_dict(),
                'epoch':            i,
                'test_acc':         test_acc
            }, f'{checkpoints_dir}/best_model.pth')
            print('Best model updated.')

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i, running_loss,
                train_eval_loss, train_eval_acc,
                test_loss, test_acc,
                acc_gap, loss_gap,
                current_lr, epoch_time, best_acc
            ])

    print(f'Best acc: {best_acc:.4f}  Best epoch: {best_epoch}')

    # ── Final evaluation and reports ──────────────────────────────────────────
    final_test_loss, final_test_acc, y_true, y_pred = evaluate_with_predictions(
        res18, fc, test_loader, device
    )

    cm = confusion_matrix(y_true, y_pred)
    with open(f"{reports_dir}/test_confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cm.tolist())

    report = classification_report(y_true, y_pred, digits=4)
    with open(f"{reports_dir}/test_classification_report.txt", "w") as f:
        f.write(report)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    with open(f"{reports_dir}/per_class_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "precision", "recall", "f1_score", "support"])
        for class_id in range(len(precision)):
            writer.writerow([
                class_id,
                precision[class_id], recall[class_id],
                f1[class_id],        support[class_id]
            ])

    with open(f"{reports_dir}/summary.txt", "w") as f:
        f.write(f"Experiment: {args.experiment}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best test accuracy:  {best_acc:.6f}\n")
        f.write(f"Final test accuracy: {final_test_acc:.6f}\n")
        f.write(f"Final test loss:     {final_test_loss:.6f}\n")


if __name__ == '__main__':
    train()
