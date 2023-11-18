import argparse
import os
import yaml
import datetime
from models.vit import interpolate_pos_embed as ext_pos_emb
from pathlib import Path
from dataset import dataset_factory, create_loader
from models.BLIP.blip_vqa import blip_vqa
import wandb
import utils
import torch
from tqdm import tqdm

ANONYMOUS_CODE
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/BLIP.Q.yaml')

    parser.add_argument('--debug', action='store_true', help='disable wandb logs. (default false)')
    parser.add_argument('--output_dir', default='ANONYMIZED')
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--gpu', default=0, type=int, help='gpu ids')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_port', required=False, help='port for distributed')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config, 'r'))


    wandb.Api(timeout=60)
    run = wandb.init(
        project="PubMedBLIP",
        entity="aiis-chair",
        mode="disabled" if args.debug else "online",
        name=f"Error analysis"
    )

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    utils.set_seed(args.seed)

    WEIGHTS = {
        "sega": "ANONYMIZED",
        "baseline": "ANONYMIZED"
    }
    TASK = "rad"
    logs = {
        "image": [],
        "baseline": [],
        "sega": []
    }

    ####
    # DATA LOADING
    vqa_dataset = dataset_factory(config, "vqa")
    vqa_loader = create_loader(vqa_dataset, "vqa", config, workers={"train": 0, "test": 0})

    ######
    # LOADING BASLINE
    baseline_model = blip_vqa(
        image_size=config['image_res'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer']
    )
    baseline_model = baseline_model.to(device)
    checkpoint = torch.load(os.path.join(WEIGHTS["baseline"], TASK, "best", "best.pth"), map_location='cpu')
    state_dict = checkpoint['model']
    state_dict['visual_encoder.pos_embed'] = ext_pos_emb(state_dict['visual_encoder.pos_embed'], baseline_model.visual_encoder)
    baseline_model.load_state_dict(state_dict, strict=False)
    baseline_model.eval()

    ######
    # LOADING SEGA
    sega_model = blip_vqa(
        image_size=config['image_res'],
        vit=config['vit'],
        vit_grad_ckpt=config['vit_grad_ckpt'],
        vit_ckpt_layer=config['vit_ckpt_layer']
    )
    sega_model = sega_model.to(device)
    checkpoint = torch.load(os.path.join(WEIGHTS["sega"], TASK, "best", "best.pth"), map_location='cpu')
    state_dict = checkpoint['model']
    state_dict['visual_encoder.pos_embed'] = ext_pos_emb(state_dict['visual_encoder.pos_embed'], sega_model.visual_encoder)
    sega_model.load_state_dict(state_dict, strict=False)
    sega_model.eval()

    for task in vqa_loader["train"].keys():
        loader = vqa_loader["test"][task]
        answer_candidates = loader.dataset.get_answerlist()
        k_test = min(len(answer_candidates), 128)
        candidate_tokens = sega_model.tokenizer(answer_candidates, padding='longest', return_tensors='pt').to(device)
        candidate_tokens.input_ids[:, 0] = sega_model.tokenizer.bos_token_id

        with torch.no_grad():
            for i, (image, question, gt_answers, is_open, category, area) in enumerate(tqdm(loader, total=len(loader))):
                image = image.to(device, non_blocking=True)
                # answering as a classification task and computing metrics
                answers, _ = baseline_model(image, question, candidate_tokens, train=False, inference='rank',k_test=k_test)
                answers = answers.int().cpu().detach().numpy()
                baseline_answers = [answer_candidates[a] for a in answers]

                answers, _ = sega_model(image, question, candidate_tokens, train=False, inference='rank',k_test=k_test)
                image = image.cpu().detach().numpy()
                answers = answers.int().cpu().detach().numpy()
                sega_answers = [answer_candidates[a] for a in answers]

                for j in range(len(answers)):
                    if baseline_answers[j] != gt_answers[j]:
                        if baseline_answers[j] != sega_answers[j]:
                            wandb.log({f"{task} differs": wandb.Image(
                                image[j][0] / image[j][0].max(),
                                caption= f"{question[j]}, gt: {gt_answers[j]} \n"
                                         f"baseline: {baseline_answers[j]}, \n"
                                         f"SEGA: {sega_answers[j]}, \n",
                            )})
                        else:
                            wandb.log({f"{task} same": wandb.Image(
                                image[j][0] / image[j][0].max(),
                                caption= f"{question[j]}, gt: {gt_answers[j]} \n"
                                         f"baseline: {baseline_answers[j]}, \n"
                                         f"SEGA: {sega_answers[j]}, \n",
                            )})

    print("That's all folks!")
