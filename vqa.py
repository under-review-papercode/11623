import os
from torch import autocast
import torch
import wandb
from tqdm import tqdm
import evaluate
from sklearn.metrics import f1_score
import numpy as np
from pathlib import Path


def vqa_tuning(model, loader, optim, epoch, device, scheduler, dataset_name, warmup_steps, scaler=None, phase="Tuning"):

    model.train()
    epoch_loss = []
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    for i, (image, question, gt_answers, _, _, _) in enumerate(tqdm(loader, total=len(loader), desc=f"finetuning on {dataset_name} | epoch {epoch}")):
        optim.zero_grad()
        image = image.to(device, non_blocking=True)
        if scaler:
            with autocast(device_type='cuda', dtype=torch.float16):
                loss = model(image, question, gt_answers, train=True)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, question, gt_answers, train=True)
            loss.backward()
            optim.step()

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

        wandb.log({f"{phase}/{dataset_name}/Train_Loss": float(loss.item()), "epoch": epoch})
        epoch_loss.append(loss.item())
    return np.mean(epoch_loss)


def vqa_eval(model, loader, epoch, device, dataset_name, phase="Tuning"):
    model.eval()
    epoch_loss = []
    with torch.no_grad():
        for i, (image, question, gt_answers, _, _, _) in enumerate(tqdm(loader, total=len(loader), desc=f"validation on {dataset_name} | epoch {epoch}")):
            image = image.to(device, non_blocking=True)
            loss = model(image, question, gt_answers, train=True)
            wandb.log({f"{phase}/{dataset_name}/Val_Loss": float(loss.item()), "epoch": epoch})
            epoch_loss.append(loss.item())
    return np.mean(epoch_loss)


def vqa_predict(model, loader, epoch, device, phase, dataset_name):
    assert phase in ["vqa", "Pretrain", "Zero-shot"]
    assert dataset_name in ["rad", "slake", "Zero-shot", "RAD-Zero-shot", "SLAKE-Zero-shot", "ovqa"]
    # VQA evaluation during pre-training or tuning (task)

    bleu = evaluate.load("bleu")
    # rouge = evaluate.load('rouge')
    accuracy_metric = evaluate.load("accuracy")

    answer_candidates = loader.dataset.get_answerlist()
    k_test = min(len(answer_candidates), 128)
    candidate_tokens = model.tokenizer(answer_candidates, padding='longest', return_tensors='pt').to(device)
    candidate_tokens.input_ids[:, 0] = model.tokenizer.bos_token_id

    model.eval()
    area_metrics = {k: {"pred": [], "gt": []} for k in loader.dataset.get_areas()}
    cat_metrics = {k: {"pred": [], "gt": []} for k in loader.dataset.get_categories()}
    auc_metric = {"pred": [], "gt": []}
    epoch_metric = {
        "Generative": {"pred": [], "gt": []},
        "Open": {"pred": [], "gt": []},
        "Close": {"pred": [], "gt": []},
    }

    with torch.no_grad():
        for i, (image, question, gt_answers, is_open, category, area) in enumerate(tqdm(loader, total=len(loader), desc=f"VQA prediction test: {dataset_name}")):
            image = image.to(device, non_blocking=True)

            # generating answers and compute generative metrics
            answers = model(image, question, train=False, inference='generate')

            epoch_metric["Generative"]["gt"] += gt_answers
            epoch_metric["Generative"]["pred"] += answers

            # answering as a classification task and computing metrics
            answers, probs = model(image, question, candidate_tokens, train=False, inference='rank', k_test=k_test)
            answers = answers.int().cpu().detach().numpy()
            gt_classes = [answer_candidates.index(gt_answer) for gt_answer in gt_answers]

            for j, gt_class in enumerate(gt_classes):
                q_type = "Open" if is_open[j] else "Close"
                epoch_metric[q_type]["gt"].append(gt_class)
                epoch_metric[q_type]["pred"].append(answers[j])
                # probs
                auc_metric["gt"].append(gt_class)
                auc_metric["pred"].append(probs[j].float().cpu().detach().numpy().tolist())
                # area
                area_metrics[area[j]]["gt"].append(gt_class)
                area_metrics[area[j]]["pred"].append(answers[j])
                for tmp in category[j]:
                    cat_metrics[tmp]["gt"].append(gt_class)
                    cat_metrics[tmp]["pred"].append(answers[j])

        ########
        # computing accuracies for the epoch and logging

        mean_accuracy = accuracy_metric.compute(
            references=epoch_metric["Open"]["gt"] + epoch_metric["Close"]["gt"],
            predictions=epoch_metric["Open"]["pred"] + epoch_metric["Close"]["pred"]
        )["accuracy"]

        # Auc metric
        # pred_scores = np.array(auc_metric["pred"])
        # common_classes = set(auc_metric["gt"]) & set(range(pred_scores.shape[1]))
        # # Filter predicted scores to include only common classes
        # filtered_pred_scores = pred_scores[:, list(common_classes)]
        # filtered_pred_scores = softmax(filtered_pred_scores, axis=1)
        # auc_score = roc_auc_score(y_true=auc_metric["gt"], y_score=filtered_pred_scores, multi_class="ovr")
        # wandb.log({
        #     f"{phase}/{dataset_name}/ROC_AUC": round(auc_score, 2),
        #     'epoch': epoch,
        # })

        for modality, values in cat_metrics.items():
            wandb.log({
                f"{phase}/{dataset_name}/{modality}":
                           accuracy_metric.compute(
                               references=values["gt"],
                               predictions=values["pred"]
                           )["accuracy"],
                'epoch': epoch,
           })

        for category, values in area_metrics.items():
            wandb.log({
                f"{phase}/{dataset_name}/{category}":
                           accuracy_metric.compute(
                               references=values["gt"],
                               predictions=values["pred"]
                           )["accuracy"],
                'epoch': epoch,
           })

        wandb.log({
            # CLASSIFICATION MEAN
            f"{phase}/{dataset_name}/Mean": mean_accuracy,
            # CATEGORIES
            f"{phase}/{dataset_name}/Open": accuracy_metric.compute(
                references=epoch_metric["Open"]["gt"],
                predictions=epoch_metric["Open"]["pred"]
            )["accuracy"],
            # CLASSIFICATION CLOSE
            f"{phase}/{dataset_name}/Close": accuracy_metric.compute(
                references=epoch_metric["Close"]["gt"],
                predictions=epoch_metric["Close"]["pred"]
            )["accuracy"],
            # GENERATIVE F1
            f"{phase}/{dataset_name}/F1": f1_score(
                epoch_metric["Generative"]["gt"],
                epoch_metric["Generative"]["pred"],
                average="micro"
            ),
            # GENERATIVE ROUGE
            # f"{phase}/{dataset_name}/ROUGE": rouge.compute(
            #     references=epoch_metric["Generative"]["gt"],
            #     predictions=epoch_metric["Generative"]["pred"],
            # )["rouge1"],
            # GENERATIVE BLEU
            f"{phase}/{dataset_name}/BLEU": bleu.compute(
                references=epoch_metric["Generative"]["gt"],
                predictions=epoch_metric["Generative"]["pred"],
            )["bleu"],
            # f'{phase}/{dataset_name}/Generation': wandb.Table(
            #     columns=['Question', "Predicted", "Groundtruth"],
            #     data=[[question[i], answers[i], gt_answers[i]] for i in range(len(answers))]
            # ),
            # PRED TOKEN DISTRIBUTION
            f'{phase}/{dataset_name}/PredDistribution': wandb.Table(
                columns=['Prediction'], data=[[epoch_metric["Generative"]["pred"]]]
            ),
            # EPOCH
            'epoch': epoch,
        })

    return mean_accuracy


def main(config, args, device, vqa_loader, task, vqa_model, optim, sched):
    best_metric = 0
    epochs = config["vqa"]['scheduler']['epochs']
    warmup_steps = config["vqa"]['scheduler']['warmup_epochs']

    for epoch in range(0, epochs):

        vqa_tuning(vqa_model, vqa_loader["train"][task], optim, epoch, device, sched, task, warmup_steps)
        if epoch > 0:
            sched.step(epoch + warmup_steps)

        vqa_eval(vqa_model, vqa_loader["test"][task], epoch, device, task)
        epoch_score = vqa_predict(vqa_model, vqa_loader["test"][task], epoch, device, "vqa", task)
        save_obj = {
            'model': vqa_model.state_dict(),
            'optim': optim.state_dict(),
            'sched': sched.state_dict(),
            'config': config,
            'epoch': epoch,
            f'{task}_score': epoch_score,
        }

        torch.save(save_obj, os.path.join(args.output_dir, task, args.tuning_policy, 'last.pth'))
        if epoch_score > best_metric:
            best_metric = epoch_score
            wandb.log({
                f"vqa/{task}/best": epoch_score,
                'epoch': epoch,
            })
            torch.save(save_obj, os.path.join(args.output_dir, task, args.tuning_policy, 'best.pth'))