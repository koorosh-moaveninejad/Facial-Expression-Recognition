
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from dataset import RafDataset
from rul import res18feature
from utils import *
from utils import evaluate

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
    test_dataset = RafDataset(args, phase='test', transform=data_transforms_val)
    train_dataset_eval = RafDataset(args, phase='train', basic_aug=False, transform=data_transforms_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
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
        pin_memory=True
    )

    res18 = res18.to(device)
    fc = fc.to(device)

    params = res18.parameters()
    params2 = fc.parameters()

    optimizer = torch.optim.Adam([
        {'params': params},
        {'params': params2, 'lr': 0.002}
    ], lr=0.0002, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_acc = 0.0
    best_epoch = 0

    for i in range(1, args.epochs + 1):
        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0

        res18.train()
        fc.train()

        for batch_i, (imgs, labels, indexes) in enumerate(train_loader):
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
            _, predicts = torch.max(outputs, 1)

            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num
            running_loss += loss.item()

        scheduler.step()

        running_loss = running_loss / iter_cnt
        print('Epoch : %d, train_loss: %.4f' % (i, running_loss))

        train_eval_loss, train_eval_acc = evaluate(res18, fc, train_eval_loader, device)
        test_loss, test_acc = evaluate(res18, fc, test_loader, device)

        print('Epoch : %d, train_eval_acc : %.4f, train_eval_loss: %.4f' % (i, train_eval_acc, train_eval_loss))
        print('Epoch : %d, test_acc : %.4f, test_loss: %.4f' % (i, test_acc, test_loss))

        with torch.no_grad():
            res18.eval()
            fc.eval()

            running_loss = 0.0
            iter_cnt = 0
            correct_sum = 0
            data_num = 0

            train_bar = tqdm(train_loader, desc=f"Epoch {i}/{args.epochs} [Train]", leave=False)
            for batch_i, (imgs, labels, indexes) in enumerate(train_bar):
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = res18(imgs, labels, phase='test')
                outputs = fc(outputs)

                loss = nn.CrossEntropyLoss()(outputs, labels)

                iter_cnt += 1


                running_loss += loss.item()
                data_num += outputs.size(0)

            running_loss = running_loss / iter_cnt
            test_acc = correct_sum.float() / float(data_num)

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = i

                # Criteria to save the trained model
                if best_acc >= 0.888:
                    torch.save({
                        'model_state_dict': res18.state_dict(),
                        'fc_state_dict': fc.state_dict()
                    }, "acc_888.pth")
                    print('Model saved.')

            print('Epoch : %d, test_acc : %.4f, test_loss: %.4f' % (i, test_acc, running_loss))

    print('best acc: ', best_acc, 'best epoch: ', best_epoch)


if __name__ == '__main__':
    train()
