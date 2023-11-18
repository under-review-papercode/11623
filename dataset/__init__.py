import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from dataset.medvqa_dataset import vqa_dataset
from dataset.ROCO import ROCO
from dataset.segmentation_dataset import segmentation_dataset
from dataset.question_factory.syntetic_dataset import SynQuestionDataset
from dataset.randaugment import RandomAugment
from dataset.randaugment import RandomRotate, RandomShiftIntensity, RandomHorizontalFlip, RandomVerticalFlip, Downscale
from dataset.randaugment import ToTensor as SegToTensor
from dataset.randaugment import Normalize as SegNormalize

def create_dataset(dataset_name, config, patch_strategy, patch_fn):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    if "vqa_pretrain" in dataset_name:
        vqa_conf = config["pretrain"]["datasets"].get("vqa_pretrain", False)
        train_dataset = SynQuestionDataset(vqa_conf['train_file'], config['image_res'], "Train")
        # using train_file in the next line is not a bug, split happens inside
        test_dataset = SynQuestionDataset(vqa_conf['train_file'], config['image_res'], "Test")
        return train_dataset, test_dataset

    # if "med_pretrain" in dataset_name:
    #
    #     pretrain_transform = transforms.Compose([
    #         transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0), interpolation=Image.BICUBIC),
    #         transforms.RandomHorizontalFlip(),
    #         RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
    #                                               'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    #
    #     roco_cfg = config["pretrain"]["datasets"].get("med_pretrain", False)
    #     train_dataset = ROCO(roco_cfg['train_file'], pretrain_transform, image_root=roco_cfg['image_root'], patchify_fn=patch_fn)
    #     test_dataset = ROCO(roco_cfg['test_file'], pretrain_transform, image_root=roco_cfg['image_root'], patchify_fn=patch_fn)
    #     return train_dataset, test_dataset
    #
    # if "seg_pretrain" in dataset_name:
    #
    #     segmentation_transform = transforms.Compose([
    #         RandomRotate(execution_probability=0.3, angle_spectrum=180, order=4),
    #         RandomShiftIntensity(execution_probability=0.3),
    #         RandomHorizontalFlip(execution_probability=0.3),
    #         RandomVerticalFlip(execution_probability=0.3),
    #         SegToTensor(),
    #         SegNormalize(0.53899017, 0.31196428),
    #         Downscale(config['image_res'])
    #     ])
    #
    #     seg_cfg = config["pretrain"]["datasets"].get("seg_pretrain")
    #     seg_train = segmentation_dataset(
    #         seg_cfg['train_file'],
    #         segmentation_transform,
    #         split="Train",
    #         mode=patch_strategy,
    #         patchify_fn=patch_fn)
    #     seg_test = segmentation_dataset(
    #         seg_cfg['train_file'],
    #         segmentation_transform,
    #         split="Test",
    #         mode=patch_strategy,
    #         patchify_fn=patch_fn
    #     )
    #     return seg_train, seg_test

    if dataset_name in ["rad", "slake", "ovqa"]:

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            # transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        vqa_config = config["vqa"]["datasets"].get(dataset_name)

        train_dataset = vqa_dataset(
            vqa_config['train_file'], train_transform, vqa_config['vqa_root'], split='train'
        )
        test_dataset = vqa_dataset(
            vqa_config['test_file'], test_transform, vqa_config['vqa_root'], split='test'
        )
        return train_dataset, test_dataset

    raise Exception(f"unknown dataset entry: {dataset_name}")


def pretrain_collate_fn(batch):
    im, cap, m = [], [], []
    for image, caption, mask in batch:
        im.append(image)
        cap.append(caption)
        m.append(mask)
    return torch.stack(im, dim=0), torch.stack(cap, dim=0), torch.stack(m, dim=0) if torch.is_tensor(m[0]) else m


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def dataset_factory(config, task):
    assert task in ["Pretrain", "vqa"]

    data = {"train": {}, "test": {}}
    if config.get(task.lower(), False):
        for name in config[task.lower()].get("datasets", []):
            print(f"Loading {name}.")
            train, test = create_dataset(name, config, None, None)
            data["train"][name] = train
            data["test"][name] = test
        return data


def create_loader(data, task, config, workers):
    # converting datasets to loaders
    for split in ["train", "test"]:
        # PRETRAINING DATASET
        if task == "Pretrain":

            if split == "train" and config["pretrain"].get("weighted_sampling", False) and len(data["train"]) == 1:
                print("Using Sampling Weights for the pretraining dataset.")
                samp_weights = list(data["train"].values())[0].get_samp_weights()
                data["train"] = DataLoader(
                    list(data["train"].values())[0],
                    batch_size=config["pretrain"]["batch_size"],
                    num_workers=workers["train"],
                    sampler=WeightedRandomSampler(samp_weights, len(samp_weights)),
                    collate_fn=vqa_collate_fn,
                    drop_last=True,
                )
            else:
                data[split] = DataLoader(
                    ConcatDataset(data[split].values()),
                    batch_size=config["pretrain"]["batch_size"],
                    num_workers=workers[split],
                    shuffle=True,
                    drop_last=True,
                    collate_fn=vqa_collate_fn,
                ) if len(data[split]) > 0 else None
        # VQA DATASET
        else:
            for name in data[split].keys():
                if split == "train" and config["vqa"].get("weighted_sampling", False):
                    samp_weights = data[split][name].get_samp_weights()
                else:
                    samp_weights = [1.] * len(data[split][name])

                bs = config["vqa"]["batch_size"] if split == "train" else config["vqa"]["batch_size"] // 2
                data[split][name] = DataLoader(
                    data[split][name],
                    batch_size=bs,
                    num_workers=workers[split],
                    sampler=WeightedRandomSampler(samp_weights, len(samp_weights)),
                    collate_fn=vqa_collate_fn,
                ) if len(data[split]) > 0 else None
    return data


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, open_q, category, areas = [], [], [], [], [], []
    for image, question, answer, is_open, cat, area in batch:
        image_list.append(image)
        question_list.append(question)
        answer_list.append(answer)
        open_q.append(is_open)
        category.append(cat)
        areas.append(area)
    return torch.stack(image_list, dim=0), question_list, answer_list, open_q, category, areas