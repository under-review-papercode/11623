import json
import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
from dataset.utils import generate_masks

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ROCO(Dataset):
    def __init__(self, ann_file, transform, image_root, patchify_fn, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += [
                ann for ann in pd.read_csv(f).to_dict("records") if os.path.isfile(os.path.join(image_root, ann["name"])) or os.path.islink(os.path.join(image_root, ann["name"]))
            ]

        self.transform = transform
        self.max_words = max_words
        self.image_root = image_root
        self.patchify_fn = patchify_fn

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        caption = ann['caption']
        image_path = os.path.join(self.image_root, ann['name'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if self.patchify_fn:
            mask_ids = generate_masks(image, None, "random", self.patchify_fn)
            return image, caption, mask_ids

        return image, caption
