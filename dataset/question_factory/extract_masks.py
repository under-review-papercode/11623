import json
import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from monai.utils import first
import cv2
from monai.transforms import (
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    EnsureChannelFirstd,
    Compose,
)
from monai.data import DataLoader, Dataset
from torch.nn import Upsample
from monai.transforms import SpatialPad
import torch
import hashlib, random
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None

BASE_DATA_DIR = "/ANONYMIZED/med-segmentation/data"
# SAVE_DIR = "/ANONYMIZED/segVQA2.0"
SAVE_DIR = "ANONYMIZED/SegVQA3.0"
SPLEEN_DATASET_DIR = os.path.join(BASE_DATA_DIR, "spleen")
LUNGS_DATASET_DIR = os.path.join(BASE_DATA_DIR, "lungs")
HEART_DATASET_DIR = os.path.join(BASE_DATA_DIR, "heart")
LIVER_DATASET_DIR = os.path.join(BASE_DATA_DIR, "liver")
LIVER_CANCER_DATASET_DIR = os.path.join(BASE_DATA_DIR, "DSD_Liver")  # liver cancer decathlon
LIVER_LITS_DIR = os.path.join(BASE_DATA_DIR, "LITS")
BRAIN_DATASET_DIR = os.path.join(BASE_DATA_DIR, "brain")
KIDNEY_DATASET_DIR = os.path.join(BASE_DATA_DIR, "kidney/kits19/data")
KIDNEY_XRAY_DIR = os.path.join(BASE_DATA_DIR, "xray")

XRAY_ABN_DIR = "/ANONYMIZED/med-segmentation/data/chest-bb"

ABNORMALITIES = {
    "BRAIN_TUMOR": 2, "KIDNEY_TUMOR": 8, "LIVER_TUMOR": 10,
    "ATELECTASIS": 11, "CARDIOMEGALY": 12, "EFFUSION": 13,
    "INFILTRATE": 14, "MASS": 15, "NODULE": 16,
    "PNEUMONIA": 17, "PNEUMOTHORAX": 18,
}
ORGANS = {
    "BRAIN": 1, "SPLEEN": 3, "KIDNEY": 4,
    "BLADDER": 5, "LUNGS": 6, "LIVER": 7, "VENTRICLE": 9}

FINAL_LABELS = {"BACKGROUND": 0, **ORGANS, **ABNORMALITIES}
NP_LABELS = np.asarray(list(FINAL_LABELS.values()))

# foreach of this area/dataset, if the network predicts a label which has zero in this matrix
# , and we have background in our GT for that pixel, then the loss is not computed.
# this encourages the network to explore over areas where we might be missing the labels
# but also penalises the network if it - for instance - explores brains into abdomen
MASKED_BACKPROP = {
    #       BG  B BC  S  K Bl Lu Li KT  H  LC A  C  E  I  M  N  P  Pt
    "M":    [0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "B":    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "XR":   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    "H":    [0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "S":    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "K":    [0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "LC":   [0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
}

######
# DATA EXTRACTION
NUM_SLICES = 20000  # number of slices per dataset (training. -> testing = training * 0.2) -> only for BRAIN! <-
MIN_Q = 5   # suppress below 3%
MAX_Q = 90  # suppress above 97%
DIGITS_IN_FILENAMES = 7  # up to 10M images

RANDOM_SEED = 50
TRAINING_SIZE = 0.9
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def plot_histogram(x, y, title):
    plt.figure(figsize=(24, 10))
    plt.title(f"{title} (value 0 removed)")
    plt.bar(x[1:], y[1:])
    plt.xticks(x[::4], rotation=90)
    plt.show()


def dump(df, res_imgs, res_labels, plane_name, p_id, split, dataset_code, counter):

    th = 100 if plane_name == "axial" else 200
    threshold = np.where(np.sum(res_labels > 0, axis=(1, 2)) > th)[0]
    res_labels = np.take(res_labels, threshold, axis=0).squeeze()
    res_imgs = np.take(res_imgs, threshold, axis=0).squeeze()
    assert res_labels.shape == res_imgs.shape

    for i in range(res_labels.shape[0]):
        presence = np.in1d(NP_LABELS, res_labels[i])
        path = os.path.join(SAVE_DIR, split, f"{dataset_code}{str(counter).zfill(DIGITS_IN_FILENAMES)}.npy")
        df = pd.concat([df, pd.DataFrame([
            {k: v for k, v in zip(
                ("DATASET", *FINAL_LABELS.keys(), "PLANE", "PATIENT_ID", "PATH", "SPLIT"),
                [dataset_code, *presence, plane_name, p_id, path, split.capitalize()]
            )}
        ])], ignore_index=True)
        np.save(path, np.stack((res_imgs[i], res_labels[i])))
        counter = counter + 1
    return df, counter


def load_data(image_path, label_path, transforms):
    check_ds = Dataset(data=[{"image": image_path, "label": label_path}], transform=Compose(transforms))
    data = first(DataLoader(check_ds, batch_size=1))
    return data["image"].squeeze().numpy(), data["label"].squeeze().numpy()


def extract_multi(df):
    """
    We take only the trusted LITS images. We found out that when the multi dataset has the liver labels equal to LITS
    then: C1 => liver, C3 => lungs, C4 => Kidneys
    if liver labels LITS != Multi then
    C1 => bladder, C4 => Bones (!)
    Brain is not  available for this dataset.
    Returns:
    """
    print("Extracting Multilabel LITS dataset")
    counter = 0
    DATASET_CODE = "M"
    volume_shapes = []
    df = df[df["DATASET"] != DATASET_CODE]  # reset entries
    d = os.listdir(LIVER_LITS_DIR)
    filenames = [img for img in d if img.find("volume") != -1]
    splits = {
        "train": filenames[:int(len(filenames) * TRAINING_SIZE)],
        "test": filenames[int(len(filenames) * TRAINING_SIZE) + 1:]
    }
    use_lits = ['22', '20', '18', '19', '17', '21']
    corrupted = ['26', '0', '6', '4', '3', '10', '16', '8', '5', '14', '23', '2', '11', '12', '24', '100', '15', '25',
                 '7', '9', '27', '1', '13']

    for split, filenames in splits.items():
        for filename in tqdm(filenames, total=len(filenames)):
            filename_multi = filename.replace("nii", "nii.gz")
            patient_num = filename_multi.replace("volume-", "").replace(".nii.gz", "")
            p_id = hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()[:8]

            # transformations for this dataset
            transforms = [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="LAS"),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-175,
                    a_max=250,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
            ]

            if patient_num in use_lits:
                data_path = os.path.join(LIVER_LITS_DIR, filename)
                gt_path = os.path.join(LIVER_LITS_DIR, filename.replace("volume", "segmentation"))

                imgs, labels = load_data(data_path, gt_path, transforms)
                converted = np.where(labels == 1, FINAL_LABELS["LIVER"], FINAL_LABELS["BACKGROUND"])
            else:
                data_path = os.path.join(LIVER_DATASET_DIR, filename_multi)
                gt_path = os.path.join(LIVER_DATASET_DIR, filename_multi.replace("volume", "labels"))
                imgs, labels = load_data(data_path, gt_path, transforms)
                converted = np.zeros_like(labels)
                converted[labels == 1] = FINAL_LABELS["LIVER"]
                converted[labels == 2] = FINAL_LABELS["BLADDER"]
                converted[labels == 3] = FINAL_LABELS["LUNGS"]
                converted[labels == 4] = FINAL_LABELS["KIDNEY"]

            converted = converted.astype(int)
            if np.sum(converted == FINAL_LABELS["KIDNEY"]) > np.sum(converted != FINAL_LABELS["KIDNEY"]):
                print(f"currepted kidney volume found: {patient_num}")
                continue
            labels = converted

            # ORIENTATION FIX AFTER MEDICAL CHECK
            imgs, labels = np.swapaxes(imgs, 0, 1), np.swapaxes(labels, 0, 1)  # ensure RAS
            imgs, labels = np.moveaxis(imgs, -1, 0), np.moveaxis(labels, -1, 0)  # ensure Z first
            imgs, labels = np.flip(imgs, 1), np.flip(labels, 1)
            imgs, labels = np.flip(imgs, 0), np.flip(labels, 0)
            if int(patient_num) in range(28, 48):
                imgs, labels = np.flip(imgs, 2), np.flip(labels, 2)

            # DEBUG FOR MEDICAL TEAM!
            # SAVE_THIS = "/ANONYMIZED/results/sega/samples_max"
            # plt.imshow(imgs[imgs.shape[0] // 2], cmap="gray")
            # plt.savefig(os.path.join(SAVE_THIS, f"{filename}_axial.png"), bbox_inches='tight')
            # plt.imshow(imgs[..., imgs.shape[-1] // 2], cmap="gray")
            # plt.savefig(os.path.join(SAVE_THIS, f"{filename}_sagittal.png"), bbox_inches='tight')
            volume_shapes.append(imgs.shape)

            # SAVING AXIAL IMAGES
            df, counter = dump(df, imgs, labels, "axial", p_id, split, DATASET_CODE, counter)
            if int(patient_num) not in (28, 48):
                # SAVING SAGITTAL IMAGES IF TOTAL BODY
                imgs = np.moveaxis(imgs, -1, 0)
                labels = np.moveaxis(labels, -1, 0)
                df, counter = dump(df, imgs, labels, "sagittal", p_id, split, DATASET_CODE, counter)

        print(f"MULTI Dataset. Split {split}. processed {counter} images.")
    print(f"list of unique shapes: {np.unique(volume_shapes)}")
    return df


def extract_spleen(df):
    name = "SPLEEN"
    print(f"Processing {name} dataset")
    DATASET_CODE = "S"
    basedir = SPLEEN_DATASET_DIR
    volume_shapes = []
    df = df[df["DATASET"] != DATASET_CODE]  # reset entries
    with open(os.path.join(basedir, "dataset.json")) as det:
        dataset = json.load(det)["training"]  # no label for test are available in this dataset
    splits = {
        "train": dataset[:int(len(dataset) * TRAINING_SIZE)],
        "test": dataset[int(len(dataset) * TRAINING_SIZE) + 1:]
    }
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
    for split, dataset in splits.items():
        counter = 0
        for i in tqdm(range(len(dataset)), total=len(dataset)):
            p_id = hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()[:8]

            imgs, labels = load_data(
                os.path.join(basedir, dataset[i]["image"]),
                os.path.join(basedir, dataset[i]["label"]),
                transforms
            )
            imgs, labels = np.swapaxes(imgs, 0, 1), np.swapaxes(labels, 0, 1)  # ensure RAS
            imgs, labels = np.moveaxis(imgs, -1, 0), np.moveaxis(labels, -1, 0)  # ensure Z first
            imgs, labels = np.flip(imgs, 1), np.flip(labels, 1)
            imgs, labels = np.flip(imgs, 0), np.flip(labels, 0)

            labels = labels.astype(int)
            labels[labels == 1] = FINAL_LABELS["SPLEEN"]
            volume_shapes.append(imgs.shape)

            # ONLY AXIAL; NO SAGITTAL
            df, counter = dump(df, imgs, labels, "axial", p_id, split, DATASET_CODE, counter)

        print(f"{name} Dataset. Split {split}. processed {counter} images.")
    print(f"list of unique shapes: {np.unique(volume_shapes)}")
    return df


def extract_heart(df):
    name = "HEART"
    DATASET_CODE = "H"
    print(f"Processing {name} dataset")
    basedir = HEART_DATASET_DIR
    volume_shapes = []
    df = df[df["DATASET"] != DATASET_CODE]  # reset entries
    with open(os.path.join(basedir, "dataset.json")) as det:
        dataset = json.load(det)["training"]  # no label for test are available in this dataset
    splits = {
        "train": dataset[:int(len(dataset) * TRAINING_SIZE)],
        "test": dataset[int(len(dataset) * TRAINING_SIZE) + 1:]
    }
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    for split, dataset in splits.items():
        counter = 0
        for i in tqdm(range(len(dataset)), total=len(dataset)):
            p_id = hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()[:8]

            imgs, labels = load_data(
                os.path.join(basedir, dataset[i]["image"]),
                os.path.join(basedir, dataset[i]["label"]),
                transforms
            )
            imgs, labels = np.moveaxis(imgs, 1, 0), np.moveaxis(labels, 1, 0)  # axial, coronal, sagittal
            imgs, labels = np.flip(imgs, 2), np.flip(labels, 2)
            imgs, labels = np.flip(imgs, 0), np.flip(labels, 0)

            labels = labels.astype(int)
            labels[labels == 1] = FINAL_LABELS["VENTRICLE"]
            volume_shapes.append(imgs.shape)

            # SAVING AXIAL IMAGES
            df, counter = dump(df, imgs, labels, "axial", p_id, split, DATASET_CODE, counter)

        print(f"{name} Dataset. Split {split}. processed {counter} images.")
    print(f"list of unique shapes: {np.unique(volume_shapes)}")
    return df


def extract_kidneys(df):
    print("Processing kindeys dataset")
    DATASET_CODE = "K"
    volume_shapes = []
    df = df[df["DATASET"] != DATASET_CODE]  # reset entries
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    with open(os.path.join(KIDNEY_DATASET_DIR, "kits.json")) as det:
        ds = json.load(det)
        splits = {"train": ds[:int(len(ds) * TRAINING_SIZE)], "test": ds[int(len(ds) * TRAINING_SIZE) + 1:]}
        for split, patients in splits.items():
            counter = 0
            for i in tqdm(range(len(patients)), total=len(patients)):
                p_id = hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()[:8]

                imgs, labels = load_data(
                    os.path.join(KIDNEY_DATASET_DIR, ds[i]["case_id"], "imaging.nii.gz"),
                    os.path.join(KIDNEY_DATASET_DIR, ds[i]["case_id"], "segmentation.nii.gz"),
                    transforms
                )
                imgs, labels = np.swapaxes(imgs, 0, 1), np.swapaxes(labels, 0, 1)  # ensure RAS
                imgs, labels = np.moveaxis(imgs, -1, 0), np.moveaxis(labels, -1, 0)  # ensure Z first
                imgs, labels = np.flip(imgs, 1), np.flip(labels, 1)
                imgs, labels = np.flip(imgs, 0), np.flip(labels, 0)

                volume_shapes.append(imgs.shape)
                labels = labels.astype(int)
                converted = np.zeros_like(labels)
                converted[labels == 1] = FINAL_LABELS["KIDNEY"]  # converting to our label set
                converted[labels == 2] = FINAL_LABELS["KIDNEY_TUMOR"]  # converting to our label set
                labels = converted

                # SAVING AXIAL IMAGES
                df, counter = dump(df, imgs, labels, "axial", p_id, split, DATASET_CODE, counter)


        print(f"KIDNEY Dataset. Split {split}. processed {counter} images.")
    print(f"list of unique shapes: {np.unique(volume_shapes)}")
    df = df[~((ds["DATASET"] == "K") & (df["KIDNEY"] == False))]
    return df


def extract_brain(df):
    print("Processing Brain dataset.")
    DATASET_CODE = "B"
    volume_shapes = []
    all = [f for f in os.listdir(os.path.join(BRAIN_DATASET_DIR, "images_structural_unstripped")) if not f.startswith('.')]
    tumoral = [f for f in os.listdir(os.path.join(BRAIN_DATASET_DIR, "images_segm")) if not f.startswith('.')]
    safe = [path for path in all if f"{path}_segm.nii.gz" not in tumoral]
    dataset = tumoral + safe[:len(tumoral)]  # 50%-50%
    random.shuffle(dataset)
    splits = {"train": dataset[:int(len(dataset) * TRAINING_SIZE)], "test": dataset[int(len(dataset) * TRAINING_SIZE) + 1:]}
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]
    for split, ds in splits.items():
        counter = 0
        num_slices = NUM_SLICES // len(ds)
        num_slices = num_slices // 2  # half for T1 and half for T2
        if split == "test":
            num_slices = int(num_slices * (1 - TRAINING_SIZE))
        for folder in tqdm(ds, total=len(ds)):
            folder = folder.replace("_segm.nii.gz", "")
            p_id = hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()[:8]

            for t_type in range(1, 2):
                # pick stripped/unstripped input image
                unstripped_path = os.path.join(BRAIN_DATASET_DIR, "images_structural_unstripped", folder, f"{folder}_T{t_type}_unstripped.nii.gz")
                stripped_path = os.path.join(BRAIN_DATASET_DIR, "images_structural", folder, f"{folder}_T{t_type}.nii.gz")
                seg_path = os.path.join(BRAIN_DATASET_DIR, "images_segm", f"{folder}_segm.nii.gz")

                imgs, _ = load_data(unstripped_path, stripped_path, transforms)
                strip_d, strip_l = load_data(stripped_path, stripped_path, transforms)

                # use_unstripped = np.random.uniform(0, 1) < 0.5
                # imgs = unstrip  # if use_unstripped else strip_d
                volume_shapes.append(imgs.shape)

                if os.path.exists(seg_path):  # tumor + brain
                    _, cancer = load_data(stripped_path, seg_path, transforms)
                    converted = np.zeros_like(strip_l)
                    converted[strip_l > 0] = FINAL_LABELS["BRAIN"]
                    converted[cancer > 0] = FINAL_LABELS["BRAIN_TUMOR"]
                    labels = converted
                else:
                    labels = np.where(strip_l > 0, FINAL_LABELS["BRAIN"], 0)  # just a threshold

                labels = labels.astype(int)

                imgs, labels = np.swapaxes(imgs, 0, 1), np.swapaxes(labels, 0, 1)  # ensure RAS
                imgs, labels = np.moveaxis(imgs, -1, 0), np.moveaxis(labels, -1, 0)  # ensure Z first
                imgs, labels = np.flip(imgs, 1), np.flip(labels, 1)
                imgs, labels = np.flip(imgs, 0), np.flip(labels, 0)

                # SAVING ALL CUTS
                df, counter = dump(df, imgs, labels, "axial", p_id, split, DATASET_CODE, counter)
                df, counter = dump(df, np.moveaxis(imgs, -1, 0), np.moveaxis(labels, -1, 0), "sagittal", p_id, split, DATASET_CODE, counter)
                df, counter = dump(df, np.moveaxis(imgs, -2, 0), np.moveaxis(labels, -2, 0), "coronal", p_id, split, DATASET_CODE, counter)

        print(f"BRAIN Dataset. Split {split}. processed {counter} images.")
    print(f"list of unique shapes: {np.unique(volume_shapes)}")
    return df


def extract_liver_cancer(df):
    name = "LIVER_CANCER"
    DATASET_CODE = "LC"
    print(f"Processing {name} dataset")
    basedir = LIVER_CANCER_DATASET_DIR
    volume_shapes = []
    df = df[df["DATASET"] != DATASET_CODE]  # reset entries
    with open(os.path.join(basedir, "dataset.json")) as det:
        dataset = json.load(det)["training"]  # no label for test are available in this dataset
    splits = {
        "train": dataset[:int(len(dataset) * TRAINING_SIZE)],
        "test": dataset[int(len(dataset) * TRAINING_SIZE) + 1:]
    }
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    for split, dataset in splits.items():
        counter = 0
        for i in tqdm(range(len(dataset)), total=len(dataset)):
            p_id = hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()[:8]

            imgs, labels = load_data(
                os.path.join(basedir, dataset[i]["image"]),
                os.path.join(basedir, dataset[i]["label"]),
                transforms
            )
            imgs, labels = np.swapaxes(imgs, 0, 1), np.swapaxes(labels, 0, 1)
            imgs, labels = np.moveaxis(imgs, -1, 0), np.moveaxis(labels, -1, 0)  # ensure Z first
            imgs, labels = np.flip(imgs, 1), np.flip(labels, 1)
            imgs, labels = np.flip(imgs, 0), np.flip(labels, 0)

            labels = labels.astype(int)
            labels[labels == 2] = FINAL_LABELS["LIVER_TUMOR"]
            labels[labels == 1] = FINAL_LABELS["LIVER"]
            volume_shapes.append(imgs.shape)

            # SAVING AXIAL IMAGES
            df, counter = dump(df, imgs, labels, "axial", p_id, split, DATASET_CODE, counter)
            # if Z is at least one third of the other axis, we have a good sagittal view
            if imgs.shape[0] // imgs.shape[1] < 3:
                # SAVING SAGITTAL IMAGES IF TOTAL BODY
                imgs = np.moveaxis(imgs, -1, 0)
                labels = np.moveaxis(labels, -1, 0)
                df, counter = dump(df, imgs, labels, "sagittal", p_id, split, DATASET_CODE, counter)

        print(f"{name} Dataset. Split {split}. processed {counter} images.")
    print(f"list of unique shapes: {np.unique(volume_shapes)}")
    df = df[~((df["DATASET"] == "LC") & (df["LIVER"] == False))]
    return df


def extract_xray(df):
    """
    Extract Xray from https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345
    and convert bounding boxes to mask if they exist, if a patient condition has no findings then
    a BACKGROUND mask is generated
    """
    print("Processing XRAY images!")
    DATASET_CODE = "XR"
    df = df[df["DATASET"] != DATASET_CODE]  # reset entries

    counter = 0
    xray_img_df = pd.read_csv(os.path.join(XRAY_ABN_DIR, "Data_Entry_2017_v2020.csv")).drop(['Unnamed: 0'], axis=1, errors='ignore')
    xray_mask_df = pd.read_csv(os.path.join(XRAY_ABN_DIR, "BBox_List_2017.csv")).drop(['Unnamed: 0'], axis=1, errors='ignore')

    for idx, xray_entry in tqdm(xray_img_df.iterrows(), total=len(xray_img_df)):
        findings = xray_entry["Finding Labels"]
        plane = xray_entry["View Position"]
        filename = xray_entry["Image Index"]
        bb = xray_mask_df.loc[xray_mask_df["Image Index"] == filename]

        if findings == "No Finding":
            img = cv2.imread(os.path.join(XRAY_ABN_DIR, "images", filename))[..., 0]
            mask = np.full_like(img, FINAL_LABELS["BACKGROUND"])
        elif len(bb) > 0:
            img = cv2.imread(os.path.join(XRAY_ABN_DIR, "images", filename))[..., 0]
            findings = bb.iloc[0]["Finding Label"].upper()
            x, y, w, h = bb.iloc[0][["Bbox [x", "y", "w", "h]"]]
            x, y, w, h = int(x), int(y), int(w), int(h)
            mask = np.full_like(img, FINAL_LABELS["BACKGROUND"])
            mask[y:y + h, x:x + w] = FINAL_LABELS[findings]
        else:
            continue

        split = "train" if np.random.random(1) < 0.8 else "test"
        path = os.path.join(SAVE_DIR, split, f"{DATASET_CODE}{str(counter).zfill(DIGITS_IN_FILENAMES)}.npy")
        p_id = hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()[:8]
        presence = np.in1d(NP_LABELS, mask)
        img = img / 255
        df = pd.concat([df, pd.DataFrame([
            {k: v for k, v in zip(
                ("DATASET", *FINAL_LABELS.keys(), "PLANE", "PATIENT_ID", "PATH", "SPLIT"),
                [DATASET_CODE, *presence, plane, p_id, path, split.capitalize()]
            )}
        ])], ignore_index=True)
        np.save(path, np.stack((img, mask)))
        counter = counter + 1

    print(f"export of XRAY completed! saved a total of {counter} images")
    return df

def extract_metadata(sample):
    excluded = ["dataset", "background", "split", "path", "plane", "patient id"]
    all_organs = [name.lower().replace("_", " ") for name, presence in sample.items() if presence]
    all_organs = [name for name in all_organs if name not in excluded]

    organs = [a for a in all_organs if "tumor" not in a]
    tumors = [a.replace("tumor", "").strip() for a in all_organs if "tumor" in a]
    plane = sample["PLANE"]
    return organs, tumors, plane


def extract_images(sample):
    image_path = sample["PATH"]
    image, mask = np.split(np.load(image_path), 2, axis=0)
    image = np.squeeze(image).astype(float)
    mask = np.squeeze(mask).astype(int)
    return image, mask


def patch_organ_presence(ds_path):
    ds = pd.read_csv(ds_path)
    count_fixed = 0
    for i in tqdm(range(len(ds)), total=len(ds)):
        sample = ds.loc[i]
        organs, tumors, plane = extract_metadata(sample)
        image, mask = extract_images(sample)
        im_labels = np.unique(mask)
        if 1 + len(organs) + len(tumors) != len(im_labels):
            count_fixed += 1
            for l, v in FINAL_LABELS.items():
                sample[l] = v in im_labels
            ds.loc[i] = sample

    print(f"number of fixed images: {count_fixed} over a total of {len(ds)} elements")
    ds.to_csv(ds_path, index=False)
    print("that's all folk!")


def patch_dimension(ds):
    count_fixed = 0
    im_res = 384
    im_down = Upsample(size=(im_res, im_res), mode="bilinear")
    m_down = Upsample(size=(im_res, im_res), mode="nearest")

    for i in tqdm(range(len(ds)), total=len(ds)):
        sample = ds.loc[i]
        image, mask = extract_images(sample)
        if image.shape != mask.shape:
            print(f"bad shape on sample {i} : {sample}")

        H, W = image.shape
        if H == im_res and W == im_res:
            continue

        image = torch.from_numpy(image)[None, None]
        mask = torch.from_numpy(mask)[None, None].float()

        if W > H:
            pad = (W - H) // 2
            image = torch.nn.functional.pad(image, (0, 0, pad, pad), value=image.min())
            mask = torch.nn.functional.pad(mask, (0, 0, pad, pad), value=0)
        elif H > W:
            pad = (H - W) // 2
            image = torch.nn.functional.pad(image, (pad, pad, 0, 0), value=image.min())
            mask = torch.nn.functional.pad(mask, (pad, pad, 0, 0), value=0)

        # rescale
        image = im_down(image)[0, 0].numpy()
        mask = m_down(mask)[0, 0].int().numpy()
        np.save(sample["PATH"], np.stack((image, mask)))

        # checking that label are still correct
        organs, tumors, plane = extract_metadata(sample)
        im_labels = np.unique(mask)
        if 1 + len(organs) + len(tumors) != len(im_labels):
            count_fixed += 1
            for l, v in FINAL_LABELS.items():
                sample[l] = v in im_labels
            ds.loc[i] = sample

    print(f"number of fixed images: {count_fixed} over a total of {len(ds)} elements")
    ds.to_csv(os.path.join(SAVE_DIR, "dataframe.csv"), index=False)
    print("that's all folk!")


def use_ai(imgs, window=(192, 192, 120)):
    from models.segmentor.model.Universal_model import Universal_model
    from monai.inferers import sliding_window_inference
    import torch.nn.functional as F
    from models.segmentor.utils.utils import organ_post_process, threshold_organ, TEMPLATE as T

    model = Universal_model(img_size=window,
                            in_channels=1,
                            out_channels=32,
                            backbone="unet",
                            encoding='word_embedding'
                            )
    from torch.nn import Upsample
    # input_img = Upsample(size=IN_SIZE)(torch.from_numpy(imgs[None, None].copy())
    input_img = torch.from_numpy(imgs.copy()).moveaxis(0, -1)[None, None]
    # input_img = input_img / input_img.max()
    from torchvision.transforms import Normalize
    input_img = Normalize(input_img.mean(), input_img.std())(input_img)
    input_img = input_img.cuda()
    model = model.cuda()
    store_dict = model.state_dict()
    checkpoint = torch.load("/ANONYMIZED/cache/unet.pth")
    load_dict = checkpoint['net']
    for key, value in load_dict.items():
        if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
            name = '.'.join(key.split('.')[1:])
            name = 'backbone.' + name
        else:
            name = '.'.join(key.split('.')[1:])
        store_dict[name] = value
    model.load_state_dict(store_dict)
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred = sliding_window_inference(input_img, window, 1, model, overlap=0.5, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
    pred_hard = threshold_organ(pred_sigmoid)
    organ_list = T["15"]
    pred_hard_post = organ_post_process(
        pred_hard.cpu().numpy(),
        organ_list
    )

    return pred_hard_post, input_img


if __name__ == '__main__':


    # LOADING
    df = pd.read_csv(os.path.join(SAVE_DIR, "dataframe.csv")).drop(['Unnamed: 0'], axis=1, errors='ignore')
    # df = pd.DataFrame(columns=("DATASET", *FINAL_LABELS.keys(), "PLANE", "PATIENT_ID", "PATH", "SPLIT"))  # NEW ONE

    df = extract_xray(df)
    # df.to_csv(os.path.join(SAVE_DIR, "dataframe.csv"), index=False)
    # df = extract_multi(df)  # liver, bladder, lungs, kidneys
    # df.to_csv(os.path.join(SAVE_DIR, "dataframe.csv"), index=False)
    # df = extract_spleen(df)  # spleen
    # df = extract_heart(df)  # heart
    # df.to_csv(os.path.join(SAVE_DIR, "dataframe.csv"), index=False)
    # df = extract_kidneys(df)
    # df.to_csv(os.path.join(SAVE_DIR, "dataframe.csv"), index=False)
    # df = extract_brain(df)
    # df.to_csv(os.path.join(SAVE_DIR, "dataframe.csv"), index=False)
    # df = extract_liver_cancer(df)
    # df.to_csv(os.path.join(SAVE_DIR, "dataframe.csv"), index=False)

    # patch_dimension(df)
    # patch_organ_presence("/ANONYMIZED/segVQA2.0/dataframe.csv")
    # compute_stats(df)

    train = df[df["SPLIT"] == "Train"]
    train.to_csv(os.path.join(SAVE_DIR, "train.csv"))
    test = df[df["SPLIT"] == "Test"]
    test.to_csv(os.path.join(SAVE_DIR, "test.csv"))

