import os
import cv2
import torch.utils.data as data
import pandas as pd
import random

from utils import *

class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform

        if phase == 'train':
            csv_path = args.train_label_path
            split_dir = 'train'
        else:
            csv_path = args.test_label_path
            split_dir = 'test'

        df = pd.read_csv(csv_path)

        # Expected CSV columns:
        # image_name,label_index
        # Example:
        # train_00001_aligned.jpg,4
        # test_00001_aligned.jpg,2
        image_col = 'image_name'
        label_col = 'label_index'

        if image_col not in df.columns or label_col not in df.columns:
            raise ValueError(
                f"CSV must contain columns '{image_col}' and '{label_col}'. "
                f"Found columns: {list(df.columns)}"
            )

        # Convert labels from 1..7 to 0..6 for PyTorch CrossEntropyLoss
        self.label = df[label_col].astype(int).values - 1
        self.file_paths = []

        self.aug_func = [filp_image, add_g]

        for img_name, lbl in zip(df[image_col].values, df[label_col].values):
            img_name = str(img_name)
            class_folder = str(int(lbl))   # folder names are 1,2,...,7
            file_path = os.path.join(self.raf_path, split_dir, class_folder, img_name)
            self.file_paths.append(file_path)

        #  safety check
        missing = [p for p in self.file_paths if not os.path.exists(p)]
        if len(missing) > 0:
            raise FileNotFoundError(
                f"{len(missing)} image files were not found. "
                f"First missing file: {missing[0]}"
            )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]

        image = cv2.imread(self.file_paths[idx])
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.file_paths[idx]}")

        image = image[:, :, ::-1]  # BGR -> RGB

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx