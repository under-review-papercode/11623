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
from torch.cuda.amp import GradScaler
# import torch._dynamo
# torch._dynamo.config.verbose=True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Seg_All.yaml')
    parser.add_argument('--pretrain', action='store_true', help="perform pretraining (default false")
    parser.add_argument('--reload', action='store_true', help="load best.pth (default false)")
    parser.add_argument('--exp_name', required=False)

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

    parser.add_argument('--pretrain_weights', default="ANONYMIZED", )
    parser.add_argument('--finetuning_weights', default=False)
    parser.add_argument('--reset_start_epoch', action='store_true', help="start from epoch 0 (default false)")
    parser.add_argument('--tuning_policy', default="best",
                        help="which weights to load for tuning: baseline, last, validation, best, manual")

    args = parser.parse_args()
    assert args.tuning_policy in ["baseline", "last", "validation", "best", "manual"], "choose a proper tuning policy"

    config = yaml.safe_load(open(args.config, 'r'))

    if args.exp_name:
        folder_name = args.exp_name
    elif args.debug:
        folder_name = "debug"
    else:
        folder_name = datetime.date.today().strftime('%Y-%m-%d')

    args.output_dir = os.path.join(args.output_dir, folder_name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Experiment name: {folder_name}")
    print(f"deploying path: {args.output_dir}")

    do_pretrain = config.get("pretrain", False) and args.pretrain
    Path(os.path.join(args.output_dir, "pretrain")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "slake")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "rad")).mkdir(parents=True, exist_ok=True)

    wandb.Api(timeout=60)
    run = wandb.init(
        project="PubMedBLIP",
        entity="aiis-chair",
        mode="disabled" if args.debug else "online",
        name=f"{folder_name}_{args.tuning_policy}Tuning"
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
    if args.pretrain:
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

    if args.pretrain:

        artifact = wandb.Artifact(name="pretraining", type="weights")
        print(f"Loading weights from here for pretraining: {args.pretrain_weights}")
        checkpoint = torch.load(args.pretrain_weights, map_location='cpu')
        state_dict = checkpoint['model']
        start_epoch = checkpoint['epoch'] if checkpoint.get('epoch', 0) and not args.reset_start_epoch else 0
        print(f"starting training from epoch {start_epoch}")
        state_dict['visual_encoder.pos_embed'] = ext_pos_emb(state_dict['visual_encoder.pos_embed'], vqa_model.visual_encoder)
        vqa_model.load_state_dict(state_dict, strict=False)

        # PRE-TRAINING (SYN VQA + ZERO-SHOT)
        print("Starting training on syntetic questions")
        optim_conf = config["vqa"]["optimizer"]
        optim = AdamW(vqa_model.parameters(), lr=optim_conf["lr"], weight_decay=optim_conf["weight_decay"])
        sched, _ = create_scheduler(config["vqa"]["scheduler"], optim)


        epochs = config["pretrain"]['scheduler']['epochs']
        warmup_steps = config["pretrain"]['scheduler']['warmup_epochs']

        scaler = GradScaler()

        # print("Compiling model before starting pre-training stage")
        # vqa_model = torch.compile(vqa_model)

        best_metrics = {"val_loss": torch.inf, "rad": 0, "slake": 0}
        best_metric = 0
        for epoch in range(start_epoch, epochs):

            vqa_tuning(vqa_model, pretrain_loader["train"], optim, epoch, device, sched, "Syn-data", warmup_steps, phase="Pretrain", scaler=scaler)  # train on SYN VQA
            if epoch > 0:
                sched.step(epoch + warmup_steps)

            val_loss = vqa_eval(vqa_model, pretrain_loader["test"], epoch, device, "Syn-data", "Pretrain")  # validate on SYN VQA
            rad_score = vqa_predict(vqa_model, vqa_loader["test"]["rad"], epoch, device, "Pretrain", "RAD-Zero-shot")  # ZERO SHOT ON RAD
            slake_score = vqa_predict(vqa_model, vqa_loader["test"]["slake"], epoch, device, "Pretrain", "SLAKE-Zero-shot")  # ZERO SHOT ON SLAKE

            save_obj = {
                'model': vqa_model.state_dict(),
                'optim': optim.state_dict(),
                'sched': sched.state_dict(),
                'config': config,
                'epoch': epoch,
                'rad_score': rad_score,
                'slake_score': slake_score,
                'val_loss': val_loss
            }

            torch.save(save_obj, os.path.join(args.output_dir, "pretrain", 'last.pth'))
            if slake_score > best_metrics["slake"]:
                best_metrics["slake"] = slake_score
                wandb.log({f"vqa/pretrain/best_slake": slake_score, 'epoch': epoch})
                torch.save(save_obj, os.path.join(args.output_dir, "pretrain", 'best_slake.pth'))
            if rad_score > best_metrics["rad"]:
                best_metrics["rad"] = rad_score
                wandb.log({f"vqa/pretrain/best_rad": rad_score, 'epoch': epoch})
                torch.save(save_obj, os.path.join(args.output_dir, "pretrain", 'best_rad.pth'))
            if val_loss < best_metrics["val_loss"]:
                best_metrics["val_loss"] = val_loss
                wandb.log({f"vqa/pretrain/best_loss": val_loss, 'epoch': epoch})
                torch.save(save_obj, os.path.join(args.output_dir, "pretrain", 'best_val.pth'))

        # saving final weights as artifacts
        artifact.add_file(local_path=os.path.join(args.output_dir, "pretrain", 'last.pth'), name="last")
        artifact.add_file(local_path=os.path.join(args.output_dir, "pretrain", 'best_slake.pth'), name="slake")
        artifact.add_file(local_path=os.path.join(args.output_dir, "pretrain", 'best_rad.pth'), name="rad")
        artifact.add_file(local_path=os.path.join(args.output_dir, "pretrain", 'best_val.pth'), name="validation")
        run.log_artifact(artifact)
    else:
        print("skipping pre-training!")

    # FINE-TUNING
    print("Starting fine-tuning")
    artifact = wandb.Artifact(name="tuning", type="weights")
    for task in vqa_loader["train"].keys():
        Path(os.path.join(args.output_dir, task, args.tuning_policy)).mkdir(parents=True, exist_ok=True)
        weights_path = {
            "baseline": "ANONYMIZED",
            "last": os.path.join(args.output_dir, "pretrain", f"last.pth"),
            "validation": os.path.join(args.output_dir, "pretrain", f"best_val.pth"),
            "best": os.path.join(args.output_dir, "pretrain", f"best_{task}.pth"),
            "manual": args.finetuning_weights
        }[args.tuning_policy]

        print(f"Finetuning {task}, loading weights from {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')

        state_dict = checkpoint['model']
        state_dict['visual_encoder.pos_embed'] = ext_pos_emb(state_dict['visual_encoder.pos_embed'], vqa_model.visual_encoder)
        vqa_model.load_state_dict(state_dict, strict=False)

        optim_conf = config["vqa"]["optimizer"]
        optim = AdamW(vqa_model.parameters(), lr=optim_conf["lr"], weight_decay=optim_conf["weight_decay"])
        sched, _ = create_scheduler(config["vqa"]["scheduler"], optim)

        print(f"Training/Testing on {task}")
        vqa(config, args, device, vqa_loader, task, vqa_model, optim, sched)
        artifact.add_file(
            os.path.join(args.output_dir, task, args.tuning_policy, 'best.pth'),
            name=f"{task}"
        )

    run.log_artifact(artifact)
    print("That's all folks!")
