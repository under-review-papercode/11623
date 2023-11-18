from models.mae import mae_vit_base_patch16, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM
from utils import get_world_size, is_dist_avail_and_initialized

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist


class M2I2(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 temp=0.07,
                 init_deit=True
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.losses = config["losses"]
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']

        # vision encoder
        self.visual_encoder = mae_vit_base_patch16(img_size=config['image_res'], norm_pix_loss=False)

        if init_deit:
            # MAE
            checkpoint = torch.load(config['vit_mae_pretrain_path'], map_location='cpu')
            print("Load mae pre-trained checkpoint from ", config['vit_mae_pretrain_path'])
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        vision_width = config['vision_width']
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(text_width, 2)

        self.visual_encoder_m = mae_vit_base_patch16(img_size=config['image_res'], norm_pix_loss=False)

        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]

        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, image, text, organs, mask_idx, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        loss_val = {}

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)


        # text encoder
        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True,
                                             mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # get momentum features
        # with torch.no_grad():
        #     self._momentum_update()
        #
        #     image_embeds_m = self.visual_encoder_m(image)
        #     image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
        #     image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
        #     text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask=text.attention_mask,
        #                                              return_dict=True, mode='text')
        #     text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
        #     text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
        #
        #
        #     sim_i2t_m = image_feat_m @ text_feat_all / self.temp
        #     sim_t2i_m = text_feat_m @ image_feat_all / self.temp
        #
        #     sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
        #     sim_targets.fill_diagonal_(1)
        #
        #     sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
        #     sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
        #
        # sim_i2t = image_feat @ text_feat_all / self.temp
        # sim_t2i = text_feat @ image_feat_all / self.temp
        # self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        if "ITA" in self.losses:
            sim_i2t = (image_feat @ text_feat.t) / self.temp
            sim_t2i = (text_feat @ image_feat.t) / self.temp
            labels = torch.arange(image.shape[0], device=image.device).long()
            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * labels, dim=1).mean()
            loss_ita = (loss_i2t + loss_t2i) / 2
            loss_val["ITA"] = loss_ita

        # #================= MLM ========================#
        if "MLM" in self.losses:
            input_ids = text.input_ids.clone()
            labels = input_ids.clone()

            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                          probability_matrix=probability_matrix)

            with torch.no_grad():
                logits_m = self.text_encoder_m(input_ids,
                                               attention_mask=text.attention_mask,
                                               encoder_hidden_states=image_embeds,
                                               encoder_attention_mask=image_atts,
                                               return_dict=True,
                                               return_logits=True,
                                               )
            mlm_output = self.text_encoder(input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,
                                           labels=labels,
                                           soft_labels=F.softmax(logits_m, dim=-1),
                                           alpha=alpha
                                           )
            loss_mlm = mlm_output.loss
            loss_val["MLM"] = loss_mlm

        # #================= MIM ========================#
        if "MIM" in self.losses:
            loss_mim, recon = self.visual_encoder(organs, mask_idx)
            loss_val["MIM"] = loss_mim
            return loss_val, recon

        return loss_val

    def generate_masks(self, img, mode, organs=None):
        """
        Perform per-sample masking of areas marked with organs.
        x: [N, L, D], sequence
        organs: [N, L, D], sequence
        """

        assert mode in ["random", "threshold", "organ"]

        img = self.visual_encoder.patchify(img)
        N, L, D = img.shape  # batch, length, dim

        if mode == "organ":  # our approach both all organs or question organs, depends on the dataloader
            organs = self.visual_encoder.patchify(organs)
            restore_idx = torch.any(organs > 0, dim=2, keepdim=True)
        if mode == "random":  # classic MAE
            noise = torch.rand(N, L, 1, device=img.device)
            return (noise > 0.5).bool()
        if mode == "threshold":  # our new baseline
            # th = torch.quantile(img, 0.1)
            th = 0
            restore_idx = torch.all(img > th, dim=2, keepdim=True)
        return restore_idx


    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        if is_dist_avail_and_initialized():
            image_feats = concat_all_gather(image_feat)
            text_feats = concat_all_gather(text_feat)
        else:
            image_feats = image_feat
            text_feats = text_feat

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
