import numpy as np
import pandas as pd
from segmentation.caption_factory import CaptionFactory
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
from dataset.question_factory.questions import QuestionFactory, extract_metadata_from_mask, extract_images
import random
from torchvision import transforms
from PIL import Image
from dataset.randaugment import RandomRotate, RandomShiftIntensity, RandomHorizontalFlip, RandomVerticalFlip, Downscale
from dataset.randaugment import ToTensor as SegToTensor
from dataset.randaugment import Normalize as SegNormalize
from vqaTools.vqaExporter import VqaPreprocessor
from dataset.medvqa_dataset import pre_question
from torch.nn import Upsample
from monai.transforms.croppad.array import RandCropByLabelClasses
from monai.utils.misc import fall_back_tuple
from monai.transforms.croppad.array import SpatialCrop
from dataset.question_factory.extract_masks import FINAL_LABELS

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class CustomCropWithLabel(RandCropByLabelClasses):

    def __call__(self, image, label, indices=None):
        if label is None:
            label = self.label
        if label is None:
            raise ValueError("label should be provided.")
        if image is None:
            image = self.image
        try:
            self.randomize(label, indices, image)
        except ValueError as error:
            manual_idx = [np.where(label.flatten() == i)[0] for i in range(len(FINAL_LABELS))]
            self.ratios = [0] + [1] * (len(FINAL_LABELS) - 1)
            self.randomize(label, manual_idx, image)

        if self.centers is not None:
            center = self.centers[0]
            roi_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
            cropper = SpatialCrop(roi_center=tuple(center), roi_size=roi_size)
            return cropper(image), cropper(label)


class SynQuestionDataset(Dataset):
    def __init__(self, ann_file, image_res, split, eos='[SEP]',):

        self.ann = []
        for f in ann_file:
            ds = pd.read_csv(f)
            ds = ds[ds["SPLIT"] == split]
            self.ann += [ann for ann in ds.to_dict("records")]

        weights = {k: 0 for k in ds["DATASET"].unique()}
        for a in self.ann:
            weights[a["DATASET"]] += 1
        self.sampling_weights = {k: round(max(list(weights.values())) / v, 2) for k, v in weights.items()}

        self.transform = transforms.Compose([
                            RandomRotate(execution_probability=0.3, angle_spectrum=15, order=4),
                            RandomShiftIntensity(execution_probability=0.25),
                            RandomHorizontalFlip(execution_probability=0.2),
                            RandomVerticalFlip(execution_probability=0.3),
                            SegToTensor(),
        ])
        self.down = Upsample(size=(image_res, image_res), mode="bilinear")
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.factory = CaptionFactory()
        self.f = QuestionFactory()
        self.eos = eos
        self.preprocess = VqaPreprocessor()

    def __len__(self):
        return len(self.ann)

    def get_samp_weights(self):
        return [self.sampling_weights[sample["DATASET"]] for sample in self.ann]

    def __getitem__(self, index):

        # each entry has {DATASET, PATH, SPLIT, BACKGROUND and ORGANS}
        sample = self.ann[index].copy()

        # plane = sample["PLANE"]
        # organs = []
        # while len(organs) == 0:
        #
        #     image, mask = extract_images(sample)
        #
        #     # CROPPING to SQUARE ROI centered on a label
        #     H, W = image.shape
        #     fix_dim = H if H < W else W
        #     cropper = CustomCropWithLabel(
        #         spatial_size=[fix_dim, fix_dim],
        #         ratios=[0] + [1] * (len(FINAL_LABELS) - 1),
        #         num_classes=len(FINAL_LABELS)
        #     )
        #
        #     image, mask = cropper(image[None], mask[None])
        #
        #     image = image[0]
        #     mask = mask[0]
        #
        #     image, mask = self.transform([image, mask.astype(float)])
        #     image = image.repeat(3, 1, 1).float()
        #     mask = mask.int().numpy()
        #
        #     # METADATA
        #     organs, tumors = extract_metadata_from_mask(mask)

        image, mask = extract_images(sample)
        q, a, is_open, category, area = self.f.sample_question(sample)

        image, _, _ = self.transform([image, mask.astype(float), category])
        image = image.repeat(3, 1, 1).float()

        q = pre_question(q)
        a = self.preprocess.pre_answer(a)

        image = self.down(image[None])[0]  # resize
        image = self.normalize(image)

        return image, q, a, is_open, category, area
