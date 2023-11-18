import os
import numpy as np
import torch
import wandb
from tqdm import tqdm
import utils
from vqa import vqa_predict


def train_fn(model, data_loader, optim, epoch, device, max_alpha, train=True):

    if train:
        model.train()
        heading = "Train"
    else:
        model.eval()
        heading = "Validation"

    epoch_loss = []
    for i, val in enumerate(tqdm(data_loader, total=len(data_loader), desc=f"{heading} epoch {epoch}")):

        if train:
            optim.zero_grad()

        alpha = max_alpha if epoch > 0 else max_alpha * min(1, i / len(data_loader))

        if len(val) == 3:  # mae active
            image, caption, mask_ids = val
            image = image.to(device, non_blocking=True)
            mask_ids = mask_ids.to(device)
            losses, recon = model(image, caption, mask_ids)
            if i % 100 == 0:
                wandb.log(
                    {f"Pretrain/{heading}-Recon": wandb.Image(recon, caption="reconstruction from MAE"), "epoch": epoch})
        else:
            image, caption = val
            image = image.to(device, non_blocking=True)
            losses = model(image, caption, alpha=alpha)

        loss = sum(losses.values())
        if train:
            loss.backward()
            optim.step()


        for loss_name, loss_value in losses.items():
            wandb.log({f"Pretrain/{heading}/{loss_name}": float(loss_value.item()), "epoch": epoch})
        wandb.log({f"Pretrain/{heading}/tot": float(loss.item()), "epoch": epoch})

        epoch_loss.append(loss.item())
    return np.mean(epoch_loss)


def main(config, args, device, vqa_loader, pretrain_loader, model_t, vqa_model, optim, sched):

    model_without_ddp = model_t
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model_t, device_ids=[args.gpu])
        model_without_ddp = model.module

    epochs = config["pretrain"]['scheduler']['epochs']
    warmup_steps = config["pretrain"]['scheduler']['warmup_epochs']
    best_metric = 0
    for epoch in range(0, epochs):
        train_fn(model_t, pretrain_loader["train"], optim, epoch, device, config["alpha"], train=True)

        if epoch > 0:
            sched.step(epoch + warmup_steps)

        with torch.no_grad():
            train_fn(model_without_ddp, pretrain_loader["test"], optim, epoch, device, max_alpha=config["alpha"], train=False)
        # load weights and do vqa zero shot
        vqa_model.load_state_dict(model_without_ddp.state_dict(), strict=False)
        curr_metric, all_metrics = vqa_predict(vqa_model, list(vqa_loader["test"].items())[0][1], epoch, device, "Pretrain", "Zero-shot")

        save_obj = {
            'model': model_without_ddp.state_dict(),
            'optim': optim.state_dict(),
            'sched': sched.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(args.output_dir, 'pretrain', 'last.pth'))
        if curr_metric < best_metric:
            best_metric = curr_metric
            wandb.log({
                f"Pretrain/Zero-shot/best": all_metrics,
                'epoch': epoch,
            })
            torch.save(save_obj, os.path.join(args.output_dir, 'pretrain', 'best.pth'))