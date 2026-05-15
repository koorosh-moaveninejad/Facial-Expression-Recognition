# debug_visualize.py
import torch
import matplotlib.pyplot as plt
import cv2
import os
import argparse

from torchvision import transforms
from sklearn.decomposition import PCA

from dataset import RafDataset
from rul import res18feature
from utils import mixup_data

# ====================== CONFIG ======================
args = argparse.Namespace(
    raf_path='DATASET',
    train_label_path='DATASET/train_labels.csv',
    test_label_path='DATASET/test_labels.csv',
    pretrained_backbone_path='resnet18_msceleb.pth',
    out_dimension=64,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
model = res18feature(args).to(device)
model.eval()

# Load checkpoint
checkpoint_path = '../checkpoints/best_model.pth'
if not os.path.exists(checkpoint_path):
    checkpoint_path = '../checkpoints/last_model.pth'

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

fc = torch.nn.Linear(args.out_dimension, 7).to(device)
fc.load_state_dict(checkpoint['fc_state_dict'])
fc.eval()

print(f"✅ Loaded model from {checkpoint_path}")

# Datasets - IMPORTANT: basic_aug=False for test to avoid cv2.flip issues
test_dataset_raw = RafDataset(args, phase='test', transform=None, basic_aug=False)
test_dataset = RafDataset(args, phase='test', transform=None, basic_aug=False)  # We'll apply transform manually if needed

print(f"Test dataset size: {len(test_dataset)}")


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    img = tensor * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def show_image_with_uncertainty(idx=0, save=False):
    img_path = test_dataset_raw.file_paths[idx]
    orig_image = cv2.imread(img_path)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(orig_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = model.features(input_tensor)
        mu = model.mu(features)
        logvar = model.log_var(features)
        
        uncertainty = torch.exp(logvar).mean(dim=1).item()
        output = fc(mu)
        pred = output.argmax(dim=1).item()
    
    true_label = test_dataset_raw.label[idx]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_image)
    plt.title(f"Original Image\nTrue Label: {true_label}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(orig_image)
    plt.title(f"Uncertainty: {uncertainty:.4f}\nPredicted: {pred}")
    plt.axis('off')
    
    plt.suptitle(f"Sample {idx} — Relative Uncertainty", fontsize=14)
    plt.tight_layout()
    
    if save:
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig(f"visualizations/sample_{idx:04d}_unc_{uncertainty:.4f}.png", dpi=200, bbox_inches='tight')
    plt.show()


def visualize_mixup(num_examples=4):
    # Use a simple DataLoader with basic_aug=False and no heavy transform to avoid stride issue
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False
    )
    batch = next(iter(test_loader))
    imgs, labels, _ = batch
    imgs = imgs.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        x = model.features(imgs)
        mu = model.mu(x)
        logvar = model.log_var(x)
        sigma = torch.exp(logvar).mean(dim=1, keepdim=True)
        
        mixed_x, y_a, y_b, att1, att2 = mixup_data(mu, labels, sigma)
    
    for i in range(num_examples):
        plt.figure(figsize=(16, 5))
        j = (i + 1) % len(imgs)
        
        plt.subplot(1, 4, 1)
        plt.imshow(denormalize(imgs[i]))
        plt.title(f"Image A\nLabel: {labels[i].item()}\nUnc: {sigma[i].item():.4f}")
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(denormalize(imgs[j]))
        plt.title(f"Image B\nLabel: {labels[j].item()}\nUnc: {sigma[j].item():.4f}")
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.bar(['Weight A', 'Weight B'], [att1[i].item(), att2[i].item()], color=['blue','orange'])
        plt.ylim(0, 1)
        plt.title("Mixing Weights")
        
        plt.subplot(1, 4, 4)
        plt.text(0.5, 0.5, "Mixed Feature\n→ Add-up Loss", ha='center', va='center', fontsize=12)
        plt.axis('off')
        
        plt.suptitle("RUL Mixup Visualization", fontsize=14)
        plt.show()


def feature_visualization(n_samples=1000):
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False
    )
    features_list, labels_list, unc_list = [], [], []
    
    print("Extracting features for PCA...")
    with torch.no_grad():
        for imgs, lbls, _ in test_loader:
            if len(features_list) * imgs.size(0) > n_samples:
                break
            feats = model.features(imgs.to(device))
            mu = model.mu(feats)
            sigma = torch.exp(model.log_var(feats)).mean(dim=1)
            
            features_list.append(mu.cpu())
            labels_list.append(lbls)
            unc_list.append(sigma.cpu())
    
    features = torch.cat(features_list)[:n_samples]
    labels = torch.cat(labels_list)[:n_samples]
    uncertainties = torch.cat(unc_list)[:n_samples]
    
    pca = PCA(n_components=2)
    feats_pca = pca.fit_transform(features.numpy())
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(feats_pca[:,0], feats_pca[:,1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(label='Class')
    plt.title("PCA of Features (by Class)")
    
    plt.subplot(1, 2, 2)
    plt.scatter(feats_pca[:,0], feats_pca[:,1], c=uncertainties, cmap='viridis_r', alpha=0.6)
    plt.colorbar(label='Uncertainty')
    plt.title("PCA colored by Learned Uncertainty")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    os.makedirs("visualizations", exist_ok=True)
    
    print("\n=== RUL Visualization Tool Started ===\n")
    
    for i in [0, 5, 10, 20, 30, 50, 80, 100]:
        show_image_with_uncertainty(i, save=True)
    
    visualize_mixup(num_examples=4)
    feature_visualization(n_samples=1000)
    
    print("\n🎉 Done! Check the 'visualizations' folder.")