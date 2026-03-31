import torch

from torch.utils.data import Dataset
from PIL import Image


class CelebDS(Dataset):
    def __init__(self, meta_df, transform, conditional=False):
        self.df = meta_df.reset_index(drop=True)
        self.transform = transform
        self.conditional = conditional

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.loc[idx]
        img = Image.open(r["out_path"]).convert("RGB")
        img = self.transform(img)
        if self.conditional:
            y = torch.tensor(int(r["male"]), dtype=torch.long)
            return img, y
        return img
