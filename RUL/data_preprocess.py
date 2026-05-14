import os
import pandas as pd

ROOT = r'F:\Master\Computer Vision\Group Project\RUL\src_code\version 3 ( FERplus dataset)\FERplus'

LABEL_MAP = {
    'neutral':  0,
    'happy':    1,
    'surprise': 2,
    'sad':      3,
    'angry':    4,
    'disgust':  5,
    'fear':     6,
    'contempt': 7
}

def build_csv_to_df(root, split):
    rows = []
    split_path = os.path.join(root, split)
    for folder in sorted(os.listdir(split_path)):
        folder_path = os.path.join(split_path, folder)
        if not os.path.isdir(folder_path):
            continue
        folder_lower = folder.lower()
        if folder_lower not in LABEL_MAP:
            print(f"Skipping unknown folder: {folder}")
            continue
        label = LABEL_MAP[folder_lower]
        for img in os.listdir(folder_path):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                rows.append({
                    'image':  img,
                    'label':  label,
                    'folder': folder
                })
    df = pd.DataFrame(rows)
    print(f"  {split}: {len(df)} images across {df['folder'].nunique()} classes")
    return df

# Delete old CSVs first
for f in ['train_labels.csv', 'test_labels.csv']:
    path = os.path.join(ROOT, f)
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted old {f}")

# Regenerate from scratch using absolute ROOT
train_df = build_csv_to_df(ROOT, 'train')
val_df   = build_csv_to_df(ROOT, 'validation')
pd.concat([train_df, val_df], ignore_index=True).to_csv(
    os.path.join(ROOT, 'train_labels.csv'), index=False
)
print(f"Saved train_labels.csv with {len(train_df) + len(val_df)} rows")

test_df = build_csv_to_df(ROOT, 'test')
test_df.to_csv(os.path.join(ROOT, 'test_labels.csv'), index=False)
print(f"Saved test_labels.csv with {len(test_df)} rows")

# Verify — check first file actually exists
df = pd.read_csv(os.path.join(ROOT, 'train_labels.csv'))
row = df.iloc[0]
sample_path = os.path.join(ROOT, 'train', str(row['folder']), str(row['image']))
print(f"\nSample path: {sample_path}")
print(f"Exists: {os.path.exists(sample_path)}")



df = pd.read_csv(os.path.join(ROOT, 'train_labels.csv'))
print(f"Before: {len(df)} rows")

# Keep only rows where file actually exists on disk
clean_rows = []
for _, row in df.iterrows():
    path = os.path.join(ROOT, 'train', str(row['folder']), str(row['image']))
    if os.path.exists(path):
        clean_rows.append(row)

clean_df = pd.DataFrame(clean_rows)
clean_df.to_csv(os.path.join(ROOT, 'train_labels.csv'), index=False)
print(f"After: {len(clean_df)} rows")
print(f"Removed: {len(df) - len(clean_df)} rows")

# Verify
print("\nClass distribution:")
print(clean_df['folder'].value_counts())