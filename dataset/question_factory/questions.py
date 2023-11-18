import random
import cv2
import numpy as np
import json
import torch
from scipy.ndimage import center_of_mass
from dataset.question_factory.extract_masks import ORGANS, ABNORMALITIES, FINAL_LABELS, MASKED_BACKPROP, SAVE_DIR


# ORGANS = ["brain", "spleen", "kidney", "bladder", "lungs", "liver", "heart"]
# ALL_CLASS = ["brain", "brain tumor", "spleen", "kidney", "bladder", "lungs", "liver", "kidney tumor", "heart"]
CLOSED_PROB = 0.3


def extract_metadata(sample):
    organs = [name.lower() for name, presence in sample.items() if presence and name in ORGANS]
    tumors = [name.lower().replace("_tumor", "") for name, presence in sample.items() if presence and name in ABNORMALITIES]
    plane = sample["PLANE"]
    ds_code = sample["DATASET"]
    return organs, tumors, plane, ds_code


def extract_metadata_from_mask(mask):
    presence = np.in1d(np.asarray(list(ORGANS.values())), mask)
    organs = [o.lower() for ix, o in enumerate(ORGANS) if presence[ix]]

    presence = np.in1d(np.asarray(list(ABNORMALITIES.values())), mask)
    tumors = [o.lower().replace("_tumor", "") for ix, o in enumerate(ABNORMALITIES) if presence[ix]]

    return organs, tumors


def extract_images(sample):
    image_path = sample["PATH"]
    image, mask = np.split(np.load(image_path, allow_pickle=True), 2, axis=0)
    image = np.squeeze(image).astype(float)
    mask = np.squeeze(mask).astype(int)
    return image, mask


def valid_organs(ds_code):
    return [o.lower() for o, label_id in ORGANS.items() if MASKED_BACKPROP[ds_code][label_id]]


def extract_sub_image(mask: np.array, organ_id: int, tumor_id: int):

    target_y, target_x = center_of_mass(mask == tumor_id)
    target_cm = np.asarray([target_x, target_y])
    n_elems, single_labels, single_stats, single_cms = cv2.connectedComponentsWithStats((mask == organ_id).astype(np.uint8))

    mask_dist = np.inf
    closest_mask_id = 0
    for i in range(1, n_elems):
        if single_stats[i, cv2.CC_STAT_AREA] < 100:  # th to avoid minimal mask to be considered
            continue
        dist = np.linalg.norm(single_cms[i] - target_cm)
        if dist < mask_dist:
            mask_dist = dist
            closest_mask_id = i

    x = single_stats[closest_mask_id, cv2.CC_STAT_LEFT]
    y = single_stats[closest_mask_id, cv2.CC_STAT_TOP]
    w = single_stats[closest_mask_id, cv2.CC_STAT_WIDTH]
    h = single_stats[closest_mask_id, cv2.CC_STAT_HEIGHT]

    return mask[y:y + h, x:x + w]


def extract_area(dataset_signature):
    if dataset_signature == "B":
        return random.choice(["head", "brain", 'cranium'])
    elif dataset_signature == "XR" or dataset_signature == "H":
        return random.choice(['chest', 'thorax', 'breast', 'pectus', 'sternum', 'thoracic cavity'])
    else:
        return random.choice(['abdomen', 'stomach', 'venter', 'abdominal cavity'])


def judge_position(mask: np.array, label: str):
    width, height = mask.shape
    target_mask = np.where(mask == label, 1, 0)
    y, x = center_of_mass(target_mask)

    if x < width / 3:
        if y < height / 3:
            return "upper left"
        elif y < 2 * height / 3:
            return "left"
        else:
            return "lower left"
    elif x < 2 * width / 3:
        if y < height / 3:
            return "upper"
        elif y < 2 * height / 3:
            return "center"
        else:
            return "lower"
    else:
        if y < height / 3:
            return "upper right"
        elif y < 2 * height / 3:
            return "right"
        else:
            return "lower right"


def count(mask: np.array, label_id: int, th_=50):
    label_mask = (mask == label_id).astype(np.uint8)
    num_components, _, stats, _ = cv2.connectedComponentsWithStats(label_mask)
    return sum([1 for i in range(1, num_components) if stats[i, cv2.CC_STAT_AREA] > th_])


class QuestionFactory:

    def __init__(self, tamplate_dir="./dataset/question_factory/templates.json"):
        with open(tamplate_dir, "r") as f:
            self.templates = json.load(f)["QUESTIONS"]

    def organ_presence(self, organs: list, ds_code: str, **kwargs):
        """
        accept a list of organs in the image and the dataset code to extract the modality.
        case 1:
            if the modality is Xray a random organ is extracted among the possible ones, assuming heart and lungs are
            always visible, liver should be ignored and every other organ will not be visible.
        case 2:
            we ask to pick among various organs, organs which might be in the image but not in the segmentation mask
            are excluded. the organs in our segmentation mask become the GT
        case 3:
            we ask if an organ is present and the answer is no -> we filter out among the organs, the ones we might
            not have in the segmentation mask but which might be present in the image
        case 4:
            we ask if an organ is present and the answer is yes: we randomly pick one organ from the segmentation mask.

        Args:
            organs: List of available organs from the segmentation mask
            ds_code: Unique code for this dataset

        Returns: question, answers, is_open (always false)

        """

        multi_choice = np.random.random(1) > CLOSED_PROB
        questions = self.templates["ORGANS"]["open" if multi_choice else "close"]
        question = random.choice(questions)

        if multi_choice:
            if ds_code == "XR":
                true_organs = ["heart", "lungs"]
                all_organs = list(set(true_organs + random.sample([o.lower() for o in ORGANS], 3)))
            else:
                true_organs = organs
                all_organs = random.sample(valid_organs(ds_code), min(len(valid_organs(ds_code)), 3))
                all_organs = list(set(all_organs + true_organs))

            return question.format(organs=", ".join(all_organs)), " and ".join(true_organs), False

        else:
            if ds_code == "XR":
                true_organs = ["heart", "lungs"]
                wrong_organs = [o.lower() for o in ORGANS if o.lower() not in true_organs]
            else:
                true_organs = organs
                wrong_organs = [o for o in valid_organs(ds_code) if o not in true_organs]
            is_true = np.random.random(1) > 0.5
            return question.format(
                organ=random.choice(true_organs) if is_true else random.choice(wrong_organs)
            ), "yes" if is_true else "no", False

    def tumor_position(self, organs: list, tumors: list, mask: np.array, plane: str, **kwargs):
        """
        Create positional questions about tumors in the image, following the medical convention
        case 1:
            questions about a target w.r.t. the whole image always contain the word "in the image"
        case 2:
            questions referring to a target within a specific organs are answered w.r.t. the patient perspective and
            the cutting plane used for that image. specifically according to the medical notation:
            - for sagittal viewers -> left is ventral, right is dorsal
            - for coronal -> left is right , right is left
            - for axial -> left is right, right is left, up is ventral, down is dorsal

            the mask is cut to ensure a view of the organ and its tumor. if multiple organs are available in the mask,
            then only the organ with the closest mask to the cancer is considered
        Args:
            organs: list of organs in the image
            tumors: list of abnormalities in the image
            mask: segmentation mask
            plane: cutting plane for the image

        Returns:
        """

        assert len(tumors) > 0, "called tumor position without tumors! check this out!"
        assert len(tumors) == 1, "We should not have more than one cancer/abn type per image."

        # if we have no info on the organ we look at the overall picture
        is_within = np.random.random(1) > .5 and len(organs) > 0
        questions = self.templates["POSITION"]["lesion"]["within" if is_within else "whole"]
        question = random.choice(questions)

        target = tumors[0]
        if target.upper() in ORGANS.keys():
            tumor_id = FINAL_LABELS[f"{target.upper()}_TUMOR"]
        else:
            tumor_id = FINAL_LABELS[f"{target.upper()}"]  # just abnormality (probably XRAY)

        if is_within:
            organ_id = FINAL_LABELS[target.upper()]  # NOT xray
            # special case for kidney, we split the image in two and pick left or right kidney
            if "kidney" in target:
                W_half = mask.shape[0] // 2
                r_count = np.sum(mask[:, :W_half] == tumor_id)
                l_count = np.sum(mask[:, W_half:] == tumor_id)
                if r_count == 0:
                    target = "right kidney"
                    mask = mask[:, :W_half]
                elif l_count == 0:
                    target = "left kidney"
                    mask = mask[:, W_half:]
                else:
                    # if we have both, we pick a random one
                    p = np.random.random(1)
                    if p > .5:
                        target = "right kidney"
                        mask = mask[:, :W_half]
                    else:
                        target = "left kidney"
                        mask = mask[:, W_half:]

            question = question.format(organ=target)
            try:
                mask = extract_sub_image(mask, organ_id, tumor_id)
            except ValueError as v:
                print(f"exeption when cutting mask for positional question: {v}")

        answer = judge_position(mask, tumor_id)

        # fixing to patient perspective if needed
        if is_within:
            if plane == "sagittal":
                answer = answer.replace("left", "anterior")
                answer = answer.replace("right", "posterior")
            else:
                answer = answer.replace("left", "right") if "left" in answer else answer.replace("right", "left")
            if plane == "axial":
                answer = answer.replace("upper", "anterior")
                answer = answer.replace("lower", "posterior")
        return question, answer, True

    def plane(self, plane: str, organs: list, ds_code: str, **kwargs):
        """
        return a question about the cutting plane. usually adding also one of the organs in the image to
        enrich the question
        Args:
            plane: string of the GT plane
            organs: list of organs in the image
            ds_code: for XRay we hardcode the organs
        """
        if ds_code == "XR":
            organs = ["lungs", "heart"]
        is_open = bool(np.random.random(1) > CLOSED_PROB)
        questions = self.templates["PLANE"]["open" if is_open else "close"]
        question = random.choice(questions)
        organ = random.choice(organs)
        if is_open:
            question = question.format(organ=organ)  # more details
            return question, plane, is_open
        else:
            pred = random.choice(["axial", "coronal", "sagittal"])
            question = question.format(organ=organ, plane=pred)
            return question,  ("yes" if pred == plane else "no"), is_open

    def counting(self, mask: np.array, tumors: list, organs: list, plane: str, ds_code: str, **kwargs):
        """
        Creates counting questions. We decided to skip the count for Xray images since we always have 2 lungs and 1 heart.
        case 1: counting available organs in the mask, with a threshold on the number of occurrences for single
                connected components. one more threshold is based on human anatomy for each possible view.
        case 2: counting tumors if present else answer is zero
        """

        is_open = np.random.random(1) > CLOSED_PROB
        count_organs = np.random.random(1) > .5 and len(organs) > 0
        if count_organs:  # COUNT ORGANS!
            organ = random.choice(valid_organs(ds_code))
            if organ not in organs:  # we have not it in the mask
                organ_count = "0"
            else:
                organ_count = count(mask, FINAL_LABELS[organ.upper()])
                if plane in ["axial", "coronal"]:
                    if organ in ["kidney", "lungs"]:
                        organ_count = min(organ_count, 2)
                    else:
                        organ_count = min(organ_count, 1)
                else:
                    organ_count = min(organ_count, 1)

            # OPEN CLOSED QUESTIONS
            if is_open:  # open
                return f"How many {organ} are in the image?", str(organ_count), True
            else:
                pred = random.randint(0, 2)
                return f"{'Is' if pred == 1 else 'Are'} there {pred} {organ} in the image?", "yes" if organ_count == pred else "no", False

        else:  # COUNTING TUMORS!
            questions = self.templates["COUNT"]["open" if is_open else "close"]
            question = random.choice(questions)

            if len(tumors) > 0:
                target = random.choice(tumors)
                n_lesion = count(mask, FINAL_LABELS[f"{target.upper()}_TUMOR"])
            else:
                target = "brain" if ds_code == "B" else random.choice(["liver", "kidney"])
                n_lesion = 0

            if is_open:
                question = question.format(organ=target)
                return question, str(n_lesion), True
            else:
                pred = random.randint(0, 5)
                question = question.format(number=str(pred), organ=target, abnormality="tumor")
                return question, ("yes" if pred == n_lesion else "no"), False

    def modality(self, ds_code: str, **kwargs):
        """
        Creates a question about the modality of the image based on the info we have on the datasets
        """

        MODALITIES = {"H": "MRI", "B": "MRI", "M": "CT", "S": "CT", "K": "CT", "LC": "CT", "XR": "X-Ray"}
        area = extract_area(ds_code)
        is_open = bool(np.random.random(1) > CLOSED_PROB)
        questions = self.templates["MODALITY"]["open" if is_open else "close"]
        question = random.choice(questions)
        modality = MODALITIES[ds_code]

        if is_open:
            return question, modality, is_open
        else:
            pred = random.choice(["MRI", "X-Ray", "CT"])
            question = question.format(modality=pred, area=area)
            return question,  ("yes" if modality == pred else "no"), is_open

    def abnormality(self, organs: list, tumors: list, ds_code:str, **kwargs):

        if ds_code == "XR":
            organs = ["lungs", "heart"]

        if len(tumors) > 0:
            target = tumors[0]
            if target.upper() in ORGANS.keys():
                target = target + " tumor"
        else:
            target = random.choice(organs)

        case = np.argmax(np.random.random(size=3))
        if case == 0:
            questions = random.choice(self.templates["ABN"]["close"]["is_good"])
            question = questions.format(organ=target.replace(" tumor", ""))
            return question, ("yes" if len(tumors) == 0 else "no"), False
        elif case == 1:
            questions = random.choice(self.templates["ABN"]["close"]["is_bad"])
            question = questions.format(organ=target.replace(" tumor", ""))
            return question, ("yes" if len(tumors) > 0 else "no"), False
        else:
            questions = random.choice(self.templates["ABN"]["open"])
            return questions, (target if len(tumors) > 0 else "nothing"), True

    def organ_size(self, mask: np.array, organs: list, ds_code: str, **kwargs):
        """
        Return questions about the size of the organs or asks for the largest organ in the image
        Laveraging the number of pixel in the segmentation mask.

        if Xray the largest organ is lungs, whereas a comparison can be made between heart and lungs
        We can safely ask for the largest organs in the M, B
        We should skip this category for S, LC, and K due to the lack of relevant labels
        Args:
            mask: Segmentation mask
            organs: List of the available organs
            ds_code: Signature of the dataset
        Returns:

        """

        assert ds_code not in ["H", "S", "K", "LC"], f"organ size not supported for this dataset: {ds_code}"

        if ds_code == "XR":
            is_comparison = np.random.random(1) > 0.4
            if is_comparison:
                return "what is bigger heart or lungs?", "Lungs", False
            else:
                return "what is the largest organ in the image?", "Lungs", True

        if len(organs) == 1:
            all_organs = random.sample(valid_organs(ds_code), min(len(valid_organs(ds_code)), 3))
            all_organs = list(set(all_organs + organs))
            return "what is the largest organ in the image between " + ", ".join(all_organs) + "?", organs[0], True

        is_open = np.random.random(1) > CLOSED_PROB
        if is_open:
            lab, sizes = np.unique(mask[mask != 0], return_counts=True)
            res = [k for k, v in FINAL_LABELS.items() if v == lab[sizes.argmax()]][0].lower()
            return "what is the largest organ in the image between " + ", ".join(organs) + "?", res, True
        else:
            o1, o2 = random.sample(organs, 2)
            o1_size = np.sum(mask == FINAL_LABELS[o1.upper()])
            o2_size = np.sum(mask == FINAL_LABELS[o2.upper()])
            if np.random.random(1) > .5:
                q = f"which is bigger {o1} or {o2}?"
                a = o1 if o1_size > o2_size else o2
            else:
                q = f"which is smaller {o1} or {o2}?"
                a = o2 if o1_size > o2_size else o1
            return q, a, False

    def sample_question(self, sample):
        """
        Uniformly samples a question category and generate question/answer
        Handles borderline cases to avoid sampling un-answerable questions
        Args:
            sample: Series from pandas dataframe containing one row of our SEGA dataset

        Returns: triplet of Question, Answer, is_open

        """
        organs, tumors, plane, ds_code = extract_metadata(sample)
        _, mask = extract_images(sample)
        area = extract_area(ds_code)
        args = {"mask": mask, "organs": organs, "tumors": tumors, "plane": plane, "ds_code": ds_code}
        methods = [
            self.organ_presence, # 0
            self.tumor_position, # 1
            self.plane,          # 2
            self.counting,       # 3
            self.modality,       # 4
            self.abnormality,    # 5
            self.organ_size      # 6
        ]
        categories = ["presence", "position", "plane", "counting", "modality", "abnormality", "size"]

        probs = np.random.uniform(size=len(methods))
        # skip tumor position if we have no tumor / abnormality
        if len(tumors) == 0:
            probs[1] = 0.
        # skip counting for xray images
        if ds_code == "XR":
            probs[3] = 0.
        # skip size questions if dataset is heart, spleen and liver cancer
        if ds_code in ["H", "S", "LC", "K"]:
            probs[6] = 0.
        # we have no information about findings in the multi-organ dataset, better not to ask
        if ds_code == "M":
            probs[5] = 0.

        sampled_id = np.argmax(probs)
        return *methods[sampled_id](**args), categories[sampled_id], area


if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    import os
    import shutil
    import wandb
    from matplotlib import pyplot as plt
    ds = pd.read_csv(os.path.join(SAVE_DIR, "dataframe.csv"))
    f = QuestionFactory(tamplate_dir="templates.json")
    destination_folder = '/ANONYMIZED/dr_max_eval'
    os.makedirs(destination_folder, exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "images"), exist_ok=True)
    subsample = []
    methods = [
        f.organ_presence, f.tumor_position, f.plane,
        f.counting, f.modality, f.abnormality, f.organ_size
    ]
    # run = wandb.init(
    #     project="PubMedBLIP",
    #     entity="aiis-chair",
    #     name="samples_for_paper"
    # )
    unique_questions = set()
    unique_answers = set()

    # NUM_SAMPLES = 25
    for ds_code in ds.DATASET.unique():
        # if ds_code == "XR":
        #     columns_to_check = ["ATELECTASIS", "CARDIOMEGALY", "EFFUSION",
        #                         "INFILTRATE", "MASS", "NODULE", "PNEUMONIA", "PNEUMOTHORAX"]
        #
        #     samples = ds[columns_to_check].any(axis=1)
        #     samples = ds[samples].sample(NUM_SAMPLES)
        # else:
        #     samples = ds[ds.DATASET == ds_code].sample(NUM_SAMPLES)
        samples = ds[ds.DATASET == ds_code]
        for i, s in tqdm(samples.iterrows(), total=len(samples)):
            organs, tumors, plane, ds_code = extract_metadata(s)
            basename = os.path.basename(s["PATH"])
            # shutil.copy(s["PATH"], os.path.join(destination_folder, "images", basename))
            img, mask = extract_images(s)
            area = extract_area(ds_code)
            args = {"mask": mask, "organs": organs, "tumors": tumors, "plane": plane, "ds_code": ds_code}

            methods = [
                f.organ_presence, f.plane,
                f.modality, f.abnormality
            ]
            if len(tumors) > 0:
                methods.append(f.tumor_position)
            if ds_code != "XR":
                methods.append(f.counting)
            if ds_code not in ["H", "S", "LC", "K"]:
                methods.append(f.organ_size)
            full_line = ""
            for m in methods:
                q, a, _ = m(**args)
                unique_answers.add(a)
                unique_questions.add(q)

                # subsample.append({"path": basename, "qa": q + " " + a})
                # full_line += f"{q} -> {a}\n"

            # if np.any(mask!=0):
            #     wandb.log({"Segmentation NYC": wandb.Image(
            #         img / img.max(),
            #         masks={
            #             "organs": {
            #                 "mask_data": mask,
            #                 "class_labels": {int(FINAL_LABELS[k]): k for k in FINAL_LABELS.keys()},
            #             }
            #         },
            #         caption=full_line
            #     )})
    # pd.DataFrame(subsample).to_csv(os.path.join(destination_folder, "dataframe.csv"), index=False)
    print(f"unique answers: {len(unique_answers)}, unique questions: {len(unique_questions)}")
    print("that's all folk!")
