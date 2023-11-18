import argparse
import os
import yaml
import datetime
from models.vit import interpolate_pos_embed as ext_pos_emb
from pathlib import Path
from dataset import dataset_factory, create_loader
from vqa import main as vqa, vqa_eval, vqa_tuning, vqa_predict
from models.BLIP.blip_vqa import blip_vqa
from torch.optim import AdamW
from scheduler.scheduler_factory import create_scheduler
import wandb
import utils
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Seg_All.yaml')
    parser.add_argument('--pretrain', action='store_true', help="perform pretraining (default false")
    parser.add_argument('--reload', action='store_true', help="load best.pth (default false)")
    parser.add_argument('--exp_name', required=False)

    parser.add_argument('--debug', action='store_true', help='disable wandb logs. (default false)')
    parser.add_argument('--output_dir', default='/ANONYMIZED/results/MSM/pretrain')
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--gpu', default=0, type=int, help='gpu ids')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--workers', default=4, type=int)

    parser.add_argument('--custom_weights', required=False)
    parser.add_argument('--reset_start_epoch', action='store_true', help="start from epoch 0 (default false)")

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))

    folder_name = "debug"
    args.output_dir = os.path.join(args.output_dir, folder_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Experiment name: {folder_name}")
    print(f"deploying path: {args.output_dir}")

    wandb.Api(timeout=60)
    wandb.init(
        project="PubMedBLIP",
        entity="aiis-chair",
        mode="disabled" if args.debug else "online",
        name=f"BLIP-zeroshot"
    )
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    yaml.dump(args.__dict__, open(os.path.join(args.output_dir, 'args.yaml'), 'w'))

    print("config values: ", config)
    print("args values: ", args)

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    utils.set_seed(args.seed)

    ####
    # DATA LOADING
    vqa_dataset = dataset_factory(config, "vqa")
    vqa_loader = create_loader(vqa_dataset, "vqa", config, workers={"train": args.workers, "test": args.workers})
    pretrain_dataset = dataset_factory(config, "Pretrain")
    pretrain_loader = create_loader(
        pretrain_dataset,
        "Pretrain",
        config,
        workers={"train": args.workers, "test": args.workers}
    )

    ######
    # MODEL/WEIGHTS LOADING
    vqa_model = blip_vqa(
        image_size=config['image_res'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer']
    )
    vqa_model = vqa_model.to(device)


    ##########
    # LOAD WEIGHTS
    checkpoint = torch.load(args.custom_weights, map_location='cpu')
    state_dict = checkpoint['model']
    state_dict['visual_encoder.pos_embed'] = ext_pos_emb(
        state_dict['visual_encoder.pos_embed'],
        vqa_model.visual_encoder
    )
    vqa_model.load_state_dict(state_dict, strict=False)
    score, all_metrics = vqa_predict(
        vqa_model,
        vqa_loader["test"]["slake"],
        0,
        device,
        "Pretrain",
        "SLAKE-Zero-shot"
    )  # ZERO SHOT ON SLAKE
    print(slake_all_metrics)

