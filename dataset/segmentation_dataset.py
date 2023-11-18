import numpy as np
import pandas as pd
from segmentation.caption_factory import CaptionFactory
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
from dataset.utils import pre_caption
from dataset.utils import generate_masks

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class segmentation_dataset(Dataset):
    def __init__(self, ann_file, transform, split, mode, patchify_fn, max_words=30):
        self.ann = []
        for f in ann_file:
            ds = pd.read_csv(f)
            ds = ds[ds["SPLIT"] == split]
            self.ann += [ann for ann in ds.to_dict("records")]

        self.patch_strategy = mode
        self.transform = transform
        self.max_words = max_words
        self.factory = CaptionFactory()
        self.patchify_fn = patchify_fn

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        # each entry has {DATASET, PATH, SPLIT, BACKGROUND and ORGANS}
        ann = self.ann[index].copy()

        # ds = ann.pop("DATASET")
        image_path = ann.pop("PATH")
        caption = ann.pop("CAPTION")

        image, mask = np.split(np.load(image_path, allow_pickle=True), 2, axis=0)
        image = np.squeeze(image)
        mask = np.squeeze(mask)

        image, mask = self.transform([image, mask])
        image = image.repeat(3, 1, 1).float()
        mask = mask.repeat(3, 1, 1).int()

        if self.patchify_fn:
            mask_ids = generate_masks(image, mask, self.patch_strategy, self.patchify_fn)
            return image, caption, mask_ids
        return image, caption