import argparse
import csv
import json
import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from dataset import RafDataset
from rul import res18feature
from utils import *

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score
)

parser = argparse.ArgumentParser()

parser.add_argument('--raf_path', type=str, default='../../DATASET',
                    help='Root path of dataset folder')

parser.add_argument('--train_label_path', type=str, default='../../DATASET/train_labels.csv',
                    help='Path to train_labels.csv')

parser.add_argument('--test_label_path', type=str, default='../../DATASET/test_labels.csv',
                    help='Path to test_labels.csv')

parser.add_argument('--pretrained_backbone_path', type=str,
                    default='../pretrained_model/resnet18_msceleb.pth',
                    help='Path to pretrained backbone weights')

parser.add_argument('--workers', type=int, default=4,
                    help='Number of dataloader workers')

parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size')

parser.add_argument('--epochs', type=int, default=60,
                    help='Number of epochs')

parser.add_argument('--out_dimension', type=int, default=64,
                    help='Feature dimension')

args = parser.parse_args()


def train():
    setup_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs("../checkpoints", exist_ok=True)
    os.makedirs("../reports", exist_ok=True)

    with open("../reports/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    metrics_path = "../reports/metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_eval_loss",
            "train_eval_acc",
            "test_loss",
            "test_acc",
            "acc_gap",
            "loss_gap",
            "lr",
            "epoch_time_sec",
            "best_test_acc_so_far"
        ])

    res18 = res18feature(args)
    fc = nn.Linear(args.out_dimension, 7)

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
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

    train_dataset = RafDataset(args, phase='train', transform=data_transforms)
    train_dataset_eval = RafDataset(args, phase='train', basic_aug=False, transform=data_transforms_val)
    test_dataset = RafDataset(args, phase='test', transform=data_transforms_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda')
    )

    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda')
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda')
    )

    res18 = res18.to(device)
    fc = fc.to(device)

    optimizer = torch.optim.Adam([
        {'params': res18.parameters()},
        {'params': fc.parameters(), 'lr': 0.002}
    ], lr=0.0002, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_acc = 0.0
    best_epoch = 0

    for i in range(1, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        iter_cnt = 0

        res18.train()
        fc.train()

        train_bar = tqdm(train_loader, desc=f"Epoch {i}/{args.epochs} [Train]")
        for batch_i, (imgs, labels, indexes) in enumerate(train_bar):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            mixed_x, y_a, y_b, att1, att2 = res18(imgs, labels, phase='train')
            outputs = fc(mixed_x)

            criterion = nn.CrossEntropyLoss()
            loss_func = mixup_criterion(y_a, y_b)
            loss = loss_func(criterion, outputs)

            loss.backward()
            optimizer.step()

            iter_cnt += 1
            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        running_loss /= iter_cnt

        train_eval_loss, train_eval_acc = evaluate(res18, fc, train_eval_loader, device)
        test_loss, test_acc = evaluate(res18, fc, test_loader, device)

        acc_gap = train_eval_acc - test_acc
        loss_gap = test_loss - train_eval_loss
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        print('Epoch : %d, train_loss: %.4f' % (i, running_loss))
        print('Epoch : %d, train_eval_acc : %.4f, train_eval_loss: %.4f' % (i, train_eval_acc, train_eval_loss))
        print('Epoch : %d, test_acc : %.4f, test_loss: %.4f' % (i, test_acc, test_loss))
        print('Epoch : %d, acc_gap : %.4f, loss_gap: %.4f' % (i, acc_gap, loss_gap))

        torch.save({
            'model_state_dict': res18.state_dict(),
            'fc_state_dict': fc.state_dict(),
            'epoch': i,
            'test_acc': test_acc
        }, f'../checkpoints/epoch_{i}_acc_{test_acc:.4f}.pth')

        torch.save({
            'model_state_dict': res18.state_dict(),
            'fc_state_dict': fc.state_dict(),
            'epoch': i,
            'test_acc': test_acc
        }, '../checkpoints/last_model.pth')

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = i

            torch.save({
                'model_state_dict': res18.state_dict(),
                'fc_state_dict': fc.state_dict(),
                'epoch': i,
                'test_acc': test_acc
            }, '../checkpoints/best_model.pth')

            print('Best model updated.')

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                i,
                running_loss,
                train_eval_loss,
                train_eval_acc,
                test_loss,
                test_acc,
                acc_gap,
                loss_gap,
                current_lr,
                epoch_time,
                best_acc
            ])

    print('best acc: ', best_acc, 'best epoch: ', best_epoch)

    final_test_loss, final_test_acc, y_true, y_pred = evaluate_with_predictions(res18, fc, test_loader, device)

    cm = confusion_matrix(y_true, y_pred)
    with open("../reports/test_confusion_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cm.tolist())

    report = classification_report(y_true, y_pred, digits=4)
    with open("../reports/test_classification_report.txt", "w") as f:
        f.write(report)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    with open("../reports/per_class_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "precision", "recall", "f1_score", "support"])
        for class_id in range(len(precision)):
            writer.writerow([class_id, precision[class_id], recall[class_id], f1[class_id], support[class_id]])

    with open("../reports/summary.txt", "w") as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best test accuracy: {best_acc:.6f}\n")
        f.write(f"Final test accuracy: {final_test_acc:.6f}\n")
        f.write(f"Final test loss: {final_test_loss:.6f}\n")


if __name__ == '__main__':
    train()