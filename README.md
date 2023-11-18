# SEGA-VQA


This repository contains the code for the paper SegMedVQA: Segmentation-Guided Pre-training for Medical Visual
Under review.

## Usage

### Data Preparation

Download the following datasets and store them in the same folder for pre-processing

| Organ | Dataset                                                                                       |
|-------|-----------------------------------------------------------------------------------------------|
| 🧠    | [Download](https://shorturl.at/erFQ8)                                                         |
| 🫁    | [Dowload Multi-Organ](https://shorturl.at/doGS2), [Download Lungs](https://shorturl.at/jzAOZ) |
| 🫀    | [Download](http://medicaldecathlon.com)                                                       |
| 🩸    | [Download](https://github.com/neheller/kits19)                                                |
| 🩻    | [Download](https://shorturl.at/owGW5)                                                         | 

the preprocessing of the data can then be run with
```
cd dataset/question_factory
python extract_masks.py
```
Make sure to set the right paths in the varibles at the beginning of the file before running the script.
You also need to dowload the SLAKE and RAD dataset for finetuning and testing the model.

##  Tuning on RAD/SLAKE
Make sure to fix all the path for the config file in `configs/BLIP.Q` before running the experiments
Note: pre-training weights has to exist in exp_name folder, then those will be loaded according to the tuning policy.
- best: load Slake and RAD weights that performed the best in zero-shot 
- validation: load the weights with the best validation score on SEGA
- last: load the last weights of pretraining
- base: load BLIP base weights

```
python main.py --config ./configs/BLIP.Q.yaml --exp_name SEGA --distributed --gpu 0 --dist_port 5744  --tuning_policy validation
```
## Pretrain 
Make sure to fix all the path for the config file in `configs/BLIP.Q` before running the experiments
Note: a folder with exp_name will be created in output_dir and used for saving the weights and the logs. 
if the folder already exists, weights and logs will be loaded/over-written!

### SEGA 
```
python main.py --config ./configs/BLIP.Q.yaml  --pretrain --exp_name SEGA --distributed --gpu 0 --dist_port 5744
```

### M2I2 + SEGA
```
python main.py --config ./configs/BLIP.Q.yaml  --pretrain --exp_name name --dist_port 5699 --distributed --gpu 0 --pretrain_weights ./M2I2_weights.pth
```


### Zero Shot
Zero-shot values for the proposed model will be logged during the training on the SynVQA dataset.
Zero-shot with BLIP base weights can be obtained by running:
```
python blip_zeroshot.py
```

### Weights

Please refer to the table to download the final weights whose values are reported in the paper:

| Model            | Weights                                                                                                |
|------------------|--------------------------------------------------------------------------------------------------------|
| BLIP Base        | [Download](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth) |
| Proposed         | [Download]()                                                                                           |                                                                                         |

