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

        elif phase == 'val':
            csv_path = args.val_label_path
            split_dir = 'validation'

        elif phase == 'test':
            csv_path = args.test_label_path
            split_dir = 'test'

        else:
            raise ValueError(f"Unknown phase: {phase}")

        df = pd.read_csv(csv_path)

        # Expected CSV columns:
        # image_name,label_index
        # Example:
        # train_00001_aligned.jpg,4
        # test_00001_aligned.jpg,2
        image_col = 'image'
        folder_col = 'folder'
        label_col = 'label'

        if image_col not in df.columns or label_col not in df.columns:
            raise ValueError(
                f"CSV must contain columns '{image_col}' and '{label_col}'. "
                f"Found columns: {list(df.columns)}"
            )

     
        self.label = df[label_col].astype(int).values - 1
        self.file_paths = []

        self.aug_func = [filp_image, add_g]


        for _, row in df.iterrows():

            img_name = str(row[image_col])

            if folder_col in df.columns:
                class_folder = str(row[folder_col])
            else:
                class_folder = str(int(row[label_col]))

            file_path = os.path.join(
                self.raf_path,
                split_dir,
                class_folder,
                img_name
            )
          
            self.file_paths.append(file_path)

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