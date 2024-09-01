import math
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from positional_encodings.torch_encodings import PositionalEncoding1D
from lavis.models.blip2_models.osrt.layers import SlotAttention
from peft import LoraConfig, get_peft_model, TaskType

from lavis.models.taming_transformers.main import instantiate_from_config
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange
# from lavis.models.taming_transformers.scripts.reconstruction_usage_origin import stack_reconstructions, custom_to_pil, preprocess_vqgan, preprocess, download_image

# custom tokens
IMAGE_START = "<image>"
IMAGE_END = "</image>"


def patchify(x, n):
    """
    Rearrange the tensor from shape (b, f, h, w, c) to (b, f, h//n, w//n, c*n*n).
    Args:
        x: Input tensor of shape (b, f, h, w, c).
        n: Patch size.

    Returns:
        Patchified tensor of shape (b, f, h//n, w//n, c*n*n).
    """
    # x = rearrange(x, 'b f (n1 h) (n2 w) c -> b (f h w) (n1 n2 c)', n1=n, n2=n)  # cut high res into multiple low res
    x = rearrange(x, 'b f (h1 n1) (w1 n2) c -> b (f h1 w1) (n1 n2 c)', n1=n, n2=n)
    return x


def unpatchify(x, n, h, w):
    """
    Rearrange the tensor from shape (b, f, h//n, w//n, c*n*n) back to (b, f, h, w, c).
    Args:
        x: Input tensor of shape (b, f, h//n, w//n, c*n*n).
        n: Patch size.
        h: Original height of the tensor.
        w: Original width of the tensor.

    Returns:
        Unpatchified tensor of shape (b, f, h, w, c).
    """
    new_h = h // n
    new_w = w // n 
    x = rearrange(x, 'b (f h1 w1) (n1 n2 c) -> b f (h1 n1) (w1 n2) c', h1=new_h, w1=new_w, n1=n, n2=n)
    # z_q_predict = rearrange(z_q_predict, 'b (f h w) cnn -> b f h w cnn', )

    return x


class SEMlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 gate_layer=nn.Sigmoid,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        # return x + self.gate(x_se)
        return x * self.gate(x_se)


@registry.register_model("blip2_vqa_t5_elm_video_ego_pose")
class Blip2VQAT5ELMVideoEgoPose(Blip2Base):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,

        first_stage_config=None,

    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # self.Qformer, self.query_tokens, self.extra_query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer, self.query_tokens = self.init_Qformer_vqgan(num_query_token, 1408)

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
              
        # self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        self.t5_tokenizer = T5TokenizerFast.from_pretrained('./flan-t5-xl', local_files_only=True)

        self.origin_length = len(self.t5_tokenizer)

        # t5_config = T5Config.from_pretrained(t5_model)
        t5_config = T5Config.from_pretrained('./flan-t5-xl', local_files_only=True)
    
        t5_config.dense_act_fn = "gelu"
        # self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)
        self.t5_model = T5ForConditionalGeneration.from_pretrained('./flan-t5-xl', local_files_only=True, config=t5_config)

        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data

        # self.t5_model.get_output_embeddings().requires_grad_(True)
        # self.t5_model.get_input_embeddings().requires_grad_(True)

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)

        pos_model = PositionalEncoding1D(1408 // 3)
        x = torch.zeros(1, 256, 1408 // 3)
        self.pos_embedding = pos_model(x).squeeze().cuda()

        self.max_txt_len = max_txt_len
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None

        self.memory_bank = {}
        self.num_query_token = num_query_token

        # self.slot_attention = SlotAttention(num_slots=32, input_dim=1408, slot_dim=1408, iters=1,
        #                                     randomize_initial_slots=False)

        # self.vqgan_adapter = nn.Linear(256*16*16, self.t5_model.config.hidden_size)
        # self.vqgan_head = nn.Linear(self.t5_model.config.hidden_size, 16*16*1024, bias=False)

        self.check = True

        # self.init_first_stage_from_ckpt(first_stage_config)
        
        self.n_patch = 2
        self.image_start_chunk = None
        self.image_end_chunk = None

        self.mid_time = 3
        self.end_time = 6


    # @ torch.no_grad()
    def get_tokenized_chunks(self, tokens, inputs):
        assert self.t5_tokenizer is not None, "Error tokenizer is None!"
        chunks = []
        for token in tokens:
            chunk = self.t5_tokenizer(token).input_ids  
            # print(chunk)
            chunk = torch.tensor(chunk[2], dtype=torch.long, device=inputs.device)  # chunk[1:3]
            chunk = self.t5_model.encoder.embed_tokens(chunk)
            chunk = chunk.unsqueeze(0).expand(inputs.shape[0], -1, -1)
            chunks.append(chunk)
        return chunks   


    def pre_fusion(self, image_tokens, pre_ego_tokens):
        """Applies pre-fusion method to incorporate prior ego tokens."""
        assert self.se_layer is not None, "Semantic extraction layer must be initialized."
        image_tokens = self.se_layer(image_tokens, pre_ego_tokens)
        return image_tokens


    def split_occ_and_pose(self, inputs, input_length, pose_length, chunk_length, split=False):
        """Splits the inputs into occupancy and pose components."""
        total_length = input_length + pose_length + 2 * chunk_length
        inputs = torch.split(inputs, total_length, dim=1)

        if split:
            static_occ_list, dynamic_occ_list, ego_list = [], [], []
            for inp in inputs:
                static_occ_list.append(inp[:, :input_length//2])
                dynamic_occ_list.append(inp[:, input_length//2: input_length])
                ego_list.append(inp[:, input_length:input_length+pose_length])
            return torch.cat(static_occ_list, dim=1), torch.cat(dynamic_occ_list, dim=1), torch.cat(ego_list, dim=1)
        else:
            occ_list, ego_list = [], []
            for inp in inputs:
                occ_list.append(inp[:, :input_length])
                ego_list.append(inp[:, input_length:input_length+pose_length])
            return torch.cat(occ_list, dim=1), torch.cat(ego_list, dim=1)
        

    def register_vqgan(self, first_stage_config, ego_type="pre_fusion"):
        self.init_first_stage_from_ckpt(first_stage_config)

        self.ego_type = ego_type

        self.vqgan_adapter = nn.Linear(256*self.n_patch*self.n_patch, self.t5_model.config.hidden_size)
        self.vqgan_head = nn.Linear(self.t5_model.config.hidden_size, self.n_patch*self.n_patch*1024)

        self.lm_head = nn.Linear(self.t5_model.config.hidden_size, 2)
        self.ego_adapter = nn.Linear(2, self.t5_model.config.hidden_size)

        if ego_type == 'pre_fusion':
            # self.se_layer = SELayer(self.base_channel)
            self.se_layer = SEMlp(2, self.t5_model.config.hidden_size, self.t5_model.config.hidden_size)
    

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model


    def find_adj(self, B, adj_id, scene_id, image_embeds):
        
        batch_adj_img = []
        for i in range(B):
            adj_img_embeds = []
            for single_adj_id in adj_id[i]:
                # print(single_adj_id, self.memory_bank.keys())
                # single_adj_id = scene_id[i]
                if single_adj_id in self.memory_bank:
                    adj_img_embeds.append(self.memory_bank[single_adj_id])
            if len(adj_img_embeds) == len(adj_id[i]):
                adj_imgs = torch.stack(adj_img_embeds, dim=0)
                # print(adj_imgs.shape)
                batch_adj_img.append(adj_imgs)


    def get_2d_sincos_pos_embed(self, embed_dim, size_h, size_w):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(size_h, dtype=np.float32)
        grid_w = np.arange(size_w, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)
       

    def calc_loss_image(self, occ_logits, occ_labels):
        occ_logits = rearrange(occ_logits, 'b fhw (nn c) -> b fhw nn c', nn=self.n_patch*self.n_patch)
        # assert occ_logits.shape[-1] == self.occ_size, f"Error, occ logits size {occ_logits.shape} and vocab size {self.occ_size} mismatch!"

        loss_fct = CrossEntropyLoss()
        # occ_logits = occ_logits.contiguous().view(-1, self.occ_size)
        occ_logits = occ_logits.contiguous().view(-1, occ_logits.shape[-1])
        occ_labels = occ_labels.contiguous().view(-1)
        occ_labels = occ_labels.to(occ_logits.device)
        return loss_fct(occ_logits, occ_labels)
    
    def calc_loss_ego(self, ego_logits, ego_labels, ego_labels_mask):
        loss_mse = MSELoss()
        ego_labels = ego_labels.to(ego_logits.device).contiguous().float()
        
        bool_ego_labels_mask = ego_labels_mask.bool()
        ego_logits_valid = ego_logits[bool_ego_labels_mask]
        ego_labels_valid = ego_labels[bool_ego_labels_mask]

        return loss_mse(ego_logits_valid, ego_labels_valid)


    def process_input_split(self, hidden_img_feats, text_tokens):
        assert hidden_img_feats.shape[1] == text_tokens.shape[1]

        patchified_inputs = []
        for f in range(hidden_img_feats.shape[1]):
            patchified_input = patchify(hidden_img_feats[:, f:f+1], self.n_patch)
            patchified_length = patchified_input.shape[1]
            patchified_input = self.vqgan_adapter(patchified_input)

            if self.ego_type == 'place_holder':
                ego_tokens = torch.zeros_like(self.image_end_chunk) - 100
            elif self.ego_type == 'direct_apply':
                ego_tokens = self.ego_adapter(text_tokens[:, f:f+1])
            elif self.ego_type == 'pre_fusion':
                ego_tokens = self.ego_adapter(text_tokens[:, f:f+1])
                pre_ego_tokens = text_tokens[:, f-1:f] if f > 0 else torch.zeros_like(text_tokens[:, f:f+1]) - 100.
                patchified_input = self.pre_fusion(patchified_input, pre_ego_tokens)
            else:
                raise NotImplementedError

            # if self.training and f >= self.mid_time:
            # if f >= self.mid_time:
            # if f > self.mid_time and self.random_mask > 0 and self.is_training:

                # patchified_input = torch.zeros_like(patchified_input) - 100  # -100: None
                # ego_tokens = torch.zeros_like(ego_tokens) - 100

                # patchified_input = self.t5_model.decoder.embed_tokens(torch.zeros_like(patchified_input[:, :, 0], dtype=torch.long))  # -100: None
                # ego_tokens = self.t5_model.decoder.embed_tokens(torch.zeros_like(ego_tokens[:, :, 0], dtype=torch.long))

            patchified_input = torch.cat((self.image_start_chunk, patchified_input, ego_tokens, self.image_end_chunk), dim=1)
            patchified_inputs.append(patchified_input)

        inputs_embeds = torch.stack(patchified_inputs, dim=1)

        # if self.first_print:
        #     print('input shape:', inputs_embeds.shape)
        #     self.first_print = False

        return inputs_embeds, patchified_length


    def forward(self, samples):
        img_feats = samples["images"]
        img_feats = img_feats.permute(0, 1, 4, 2, 3).contiguous()
        B, F, D_in, H, W = img_feats.shape

        # get chunks
        if self.image_start_chunk is None or self.image_end_chunk is None:
            chunks = self.get_tokenized_chunks([IMAGE_START, IMAGE_END], img_feats)
            self.image_start_chunk = chunks[0]
            self.image_end_chunk = chunks[1]

        img_feats = img_feats.view(B*F, D_in, H, W)

        # _, z_indices = self.encode_to_z(img_feats)
        quant_z, z_indices = self.encode_to_z(img_feats)
        _, D, _, _ = quant_z.shape

        targets = z_indices.view(quant_z.shape[0], quant_z.shape[2], quant_z.shape[3]).contiguous()
        text_tokens = samples["gt_ego_poses"]
        text_tokens_mask = samples["gt_ego_poses_mask"]

        # hidden_img_feats = self.first_stage_model.encoder(img_feats)

        # for name, param in self.first_stage_model.named_parameters():
        #     print(f'Parameter name: {name}, Requires gradient: {param.requires_grad}')


        # if self.check:
        #     print(samples["questions"])
        #     print(samples["answers"])
        #     self.check = False
        device = samples["vfeats"].device
        vfeats = samples["vfeats"]

        #########################
        vfeats = vfeats.squeeze(1)
        #########################

        B = vfeats.shape[0]
        device = vfeats.device
        
        # with self.maybe_autocast():
            # if vfeats.dim() == 4:
            #     image_embeds = self.ln_vision(self.visual_encoder(vfeats))
            #     image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) # [2, 128]
            #     query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            #     query_output = self.Qformer.bert(
            #             query_embeds=query_tokens,
            #             encoder_hidden_states=image_embeds,
            #             encoder_attention_mask=image_atts,
            #             return_dict=True,)

            #     t5_query = query_output.last_hidden_state
            # else:
            #     if "timesteps" in samples:
            #         timesteps = samples["timesteps"].to(device)
            #     else:
            #         Time = vfeats.shape[1]
            #         timesteps = 0
            #     tmp_list = []
            #     index = 0
            #     for adj_img in vfeats[:,:-1,...].split(1, dim=1):
            #         adj_img = adj_img.squeeze(1)
            #         image_embeds = self.ln_vision(self.visual_encoder(adj_img))
            #         image_embeds = self.slot_attention(image_embeds)
            #         tmp_list.append(torch.unsqueeze(image_embeds, dim=1))
                
            #     image_embeds = self.ln_vision(self.visual_encoder(vfeats[:,-1,...]))
            #     image_embeds = self.slot_attention(image_embeds)
            #     tmp_list.append(torch.unsqueeze(image_embeds, dim=1))
                
            #     tmp_query = torch.cat(tmp_list, dim=1)
            #     tmp_query = tmp_query.reshape(tmp_query.shape[0], -1, tmp_query.shape[-1])
            #     image_atts = torch.ones(tmp_query.size()[:-1], dtype=torch.long).to(device) # [2, 128]

            #     query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
            #     query_tokens = query_tokens.expand(tmp_query.shape[0], -1, -1)
            #     query_output = self.Qformer.bert(
            #             query_embeds=query_tokens,
            #             encoder_hidden_states=tmp_query,
            #             encoder_attention_mask=image_atts,
            #             return_dict=True,)

            #     t5_query = query_output.last_hidden_state

        # inputs_t5 = self.t5_proj(t5_query)
        # inputs_t5 = self.vqgan_adapter(h.view(B, 1, -1))
        hidden_img_feats = quant_z.permute(0, 2, 3, 1).contiguous()
        hidden_img_feats = hidden_img_feats.view(B, F, *hidden_img_feats.shape[1:])

        patchified_inputs, patchified_length = self.process_input_split(hidden_img_feats, text_tokens)

        inputs_t5 = patchified_inputs[:, :self.mid_time]
        inputs_t5 = inputs_t5.view(B, -1, inputs_t5.shape[-1])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
        
        # answers = samples["answers"]
        
        # text_input = samples["questions"]
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # input_tokens = self.t5_tokenizer( # 8 x 17
            #     text_input,
            #     padding="longest",
            #     truncation=True,
            #     max_length=300,
            #     return_tensors="pt",
            # ).to(device)
            # output_tokens = self.t5_tokenizer( # 8 x 17
            #     answers,
            #     padding="longest",
            #     truncation=True,
            #     max_length=300,
            #     return_tensors="pt",
            # ).to(device)

            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []

            for b, _ in enumerate(range(B)):
                # batch_input_tokens_input_ids += [input_tokens.input_ids[b]]
                # batch_input_tokens_atts += [input_tokens.attention_mask[b]]
                batch_atts_t5 += [atts_t5[b]]
                batch_inputs_t5 += [inputs_t5[b]]

            # batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            # batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

            # encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)
            encoder_atts = batch_atts_t5

            # targets = output_tokens.input_ids.masked_fill(
            #     output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            # )
            
            # inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            # inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)

            inputs_embeds = batch_inputs_t5

            # outputs = self.t5_model(
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=encoder_atts,
            #     # decoder_attention_mask=output_tokens.attention_mask,
            #     return_dict=True,
            #     labels=targets,
            # )
            # loss = outputs.loss

            # targets_hold_place = torch.ones_like(targets)
            # sequence_output = self.t5_model(
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=encoder_atts,
            #     # decoder_attention_mask=output_tokens.attention_mask,
            #     return_dict=True,
            #     labels=targets_hold_place,
                
            #     vqgan=True,
            
            # )

            # decoder_inputs_embeds_first_pad = self.t5_model.decoder.embed_tokens(torch.tensor([0], device=img_feats.device))
            # decoder_inputs_embeds_first_pad = decoder_inputs_embeds_first_pad[None, :, :].expand([B, -1, -1])
            # quant_z = quant_z.permute(0, 2, 3, 1).contiguous().view(B, -1, 256)
            # embeds_quant_z = self.vqgan_adapter(quant_z)
            decoder_inputs_embeds_quant_z = patchified_inputs[:, self.mid_time:]
            decoder_inputs_embeds_quant_z = decoder_inputs_embeds_quant_z.view(B, -1, decoder_inputs_embeds_quant_z.shape[-1])
            # decoder_inputs_embeds_quant_z = torch.cat([decoder_inputs_embeds_first_pad, decoder_inputs_embeds_quant_z[:, :-1]], dim=1)
            sequence_output = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                # decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                decoder_inputs_embeds=decoder_inputs_embeds_quant_z,
                                
                vqgan=True,
            
            )

            sequence_output_logits, ego_logits = self.split_occ_and_pose(
                sequence_output, patchified_length, 1, self.image_start_chunk.shape[1], split=False)

            # labels = targets.view(B, F, -1)[:, self.mid_time:].contiguous()

            # label patchify
            targets = rearrange(
                targets, 
                '(b f) (h1 n1) (w1 n2) -> b f (h1 w1) (n1 n2)', 
                b=B, n1=self.n_patch, n2=self.n_patch
            )
            labels = targets[:, self.mid_time:].contiguous()
            labels_text_tokens = text_tokens[:, self.mid_time:].contiguous()
            labels_text_tokens_mask = text_tokens_mask[:, self.mid_time:].contiguous()

            sequence_output_logits = self.vqgan_head(sequence_output_logits)
            ego_logits = self.lm_head(ego_logits)

            loss_image = self.calc_loss_image(sequence_output_logits, labels)
            loss_ego = self.calc_loss_ego(ego_logits, labels_text_tokens, labels_text_tokens_mask)
            loss = loss_image + loss_ego * 0.5

            # sequence_output_logits = rearrange(sequence_output_logits, 'b fhw (nn c) -> b fhw nn c', nn=self.n_patch*self.n_patch)
            # lm_logits = sequence_output_logits.contiguous()

            # # index = lm_logits.detach().argmax(dim=-1) 

            # loss = None
            # if labels is not None:
            #     loss_fct = CrossEntropyLoss(ignore_index=-100, reduction=reduction)
            #     loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            #     if reduction == "none":
            #         loss = loss.view(lm_logits.size(0), -1).sum(1)

            return {"loss": loss}

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        # indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        device = samples["vfeats"].device
        records, vfeats, vfeat_lens = samples["records"], samples["vfeats"], samples["vfeat_lens"]
        word_ids, char_ids, s_labels = samples["word_ids"], samples["char_ids"], samples["s_labels"]
        e_labels, h_labels = samples["e_labels"], samples["h_labels"]

        B = s_labels.shape[0]
        device = vfeats.device
        with self.maybe_autocast():

            image_embeds = self.ln_vision(self.visual_encoder(vfeats))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) # [2, 128]

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,)

            t5_query = query_output.last_hidden_state

        inputs_t5 = self.t5_proj(t5_query)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
        
        text_input = samples["questions"]
        with torch.cuda.amp.autocast(dtype=torch.float32):
            input_tokens = self.t5_tokenizer( # 8 x 17
                text_input,
                padding="longest",
                truncation=True,
                max_length=300,
                return_tensors="pt",
            ).to(device)

            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []

            for b, _ in enumerate(range(B)):
                batch_input_tokens_input_ids += [input_tokens.input_ids[b]]
                batch_input_tokens_atts += [input_tokens.attention_mask[b]]
                batch_atts_t5 += [atts_t5[b]]
                batch_inputs_t5 += [inputs_t5[b]]

            batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

            encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)
            
            inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text
    

    def post_process(self, z_q, un_flat=False):
        # inputs: occ -> encode
        # labels: occ -> encode -> min_indices
        # logits: predicted occ
        # new logits: logits -> argmax -> 1 index -> codebook -> 2 occ code -> 3 occ adapter
        # z_q = rearrange(z_q, 'b (h w) (f c) -> b f h w c', f=f, h=50, w=50)
        # z_q = z_q[:, -1:].clone().detach().argmax(dim=-1)  # 1, 

        # z_q = z_q.detach().argmax(dim=-1)  # 1, 
        # z_q = self.occ_vae.vqvae.get_codebook_entry(z_q, shape=None)   # 2
        # if not un_flat:
        #     z_q = rearrange(z_q, 'b f h w c -> (b f) c h w')

        z_q = z_q.detach().argmax(dim=-1)  # 1, 
        z_q = self.first_stage_model.quantize.get_codebook_entry(
            z_q, shape=None)
        if not un_flat:
            z_q = rearrange(z_q, 'b f h w c -> (b f) c h w')

        return z_q    

    def get_pred(self, logits, dynamic=None):
        z_q_predict = unpatchify(logits, self.n_patch, self.n_patch, self.n_patch)
        if dynamic is None:
            z_q_predict = self.post_process(z_q_predict, un_flat=True)
        else:
            z_q_predict = self.post_process_split_dynamic(z_q_predict, un_flat=True, dynamic=dynamic)
        z_q_predict = patchify(z_q_predict, self.n_patch)
        z_q_predict = self.vqgan_adapter(z_q_predict)  # cnn -> 4096
        return z_q_predict
    
    # def get_pred_output(self, occ_logits, latent_shape, shape1, shape2, dynamic=None):
    #     z_q = unpatchify(occ_logits, self.n_patch, latent_shape[0], latent_shape[1])
    #     if dynamic is None:
    #         z_q = self.post_process(z_q, un_flat=False)
    #         z_q = self.decode_occ(z_q, shape1, shape2)
    #     else:
    #         z_q = self.post_process_split_dynamic(z_q, un_flat=False, dynamic=dynamic)
    #         z_q = self.decode_occ_split_dynamic(z_q, shape1, shape2, dynamic=dynamic)
    #     return z_q
    
    def get_pred_output(self, occ_logits, latent_shape):
        z_q = unpatchify(occ_logits, self.n_patch, latent_shape[0], latent_shape[1])
        z_q = self.post_process(z_q, un_flat=False)
        z_q = self.first_stage_model.decode(z_q)

        return z_q


    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs,
    ):

        img_feats = samples["images"]
        img_feats = img_feats.permute(0, 1, 4, 2, 3).contiguous()
        B, F, D_in, H, W = img_feats.shape

        # get chunks
        if self.image_start_chunk is None or self.image_end_chunk is None:
            chunks = self.get_tokenized_chunks([IMAGE_START, IMAGE_END], img_feats)
            self.image_start_chunk = chunks[0]
            self.image_end_chunk = chunks[1]

        img_feats = img_feats.view(B*F, D_in, H, W)

        # _, z_indices = self.encode_to_z(img_feats)
        quant_z, z_indices = self.encode_to_z(img_feats)
        _, D, _, _ = quant_z.shape

        targets = z_indices.view(quant_z.shape[0], quant_z.shape[2], quant_z.shape[3]).contiguous()
        text_tokens = samples["gt_ego_poses"]
        text_tokens_mask = samples["gt_ego_poses_mask"]

        # h = self.first_stage_model.encoder(img_feats)

        # if self.check:
        #     print(samples["questions"][0])
        #     print(samples["answers"][0])
        #     self.check = False
            
        device = samples["vfeats"].device
        vfeats = samples["vfeats"]

        #########################
        vfeats = vfeats.squeeze(1)
        #########################

        B = vfeats.shape[0]
        device = vfeats.device
        
        # with self.maybe_autocast():
        #     if vfeats.dim() == 4:
        #         image_embeds = self.ln_vision(self.visual_encoder(vfeats))
        #         image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) # [2, 128]
        #         query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        #         query_output = self.Qformer.bert(
        #                 query_embeds=query_tokens,
        #                 encoder_hidden_states=image_embeds,
        #                 encoder_attention_mask=image_atts,
        #                 return_dict=True,)

        #         t5_query = query_output.last_hidden_state
        #     else:
        #         if "timesteps" in samples:
        #             timesteps = samples["timesteps"].to(device)
        #         else:
        #             Time = vfeats.shape[1]
        #             timesteps = 0
        #         tmp_list = []
        #         index = 0
        #         for adj_img in vfeats[:,:-1,...].split(1, dim=1):
        #             adj_img = adj_img.squeeze(1)
        #             image_embeds = self.ln_vision(self.visual_encoder(adj_img))
        #             image_embeds = self.slot_attention(image_embeds)
        #             tmp_list.append(torch.unsqueeze(image_embeds, dim=1))
                
        #         image_embeds = self.ln_vision(self.visual_encoder(vfeats[:,-1,...]))
        #         image_embeds = self.slot_attention(image_embeds)
        #         tmp_list.append(torch.unsqueeze(image_embeds, dim=1))
                
        #         tmp_query = torch.cat(tmp_list, dim=1)
        #         tmp_query = tmp_query.reshape(tmp_query.shape[0], -1, tmp_query.shape[-1])
        #         image_atts = torch.ones(tmp_query.size()[:-1], dtype=torch.long).to(device) # [2, 128]

        #         query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
        #         query_tokens = query_tokens.expand(tmp_query.shape[0], -1, -1)
        #         query_output = self.Qformer.bert(
        #                 query_embeds=query_tokens,
        #                 encoder_hidden_states=tmp_query,
        #                 encoder_attention_mask=image_atts,
        #                 return_dict=True,)

        #         t5_query = query_output.last_hidden_state

        # inputs_t5 = self.t5_proj(t5_query)
        # inputs_t5 = self.vqgan_adapter(h.view(B, 1, -1))
        hidden_img_feats = quant_z.permute(0, 2, 3, 1).contiguous()
        hidden_img_feats = hidden_img_feats.view(B, F, *hidden_img_feats.shape[1:])
        latent_shape = hidden_img_feats.shape[2:4]  # get original latent h and w
        hidden_img_feats = hidden_img_feats[:, :self.mid_time]
        
        patchified_inputs, patchified_length = self.process_input_split(hidden_img_feats, text_tokens[:, :self.mid_time])

        inputs_t5 = patchified_inputs[:, :self.mid_time]
        inputs_t5 = inputs_t5.view(B, -1, inputs_t5.shape[-1])
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
        
        # text_input = samples["questions"]
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # input_tokens = self.t5_tokenizer( # 8 x 17
            #     text_input,
            #     padding="longest",
            #     truncation=True,
            #     max_length=300,
            #     return_tensors="pt",
            # ).to(device)

            batch_input_tokens_input_ids = []
            batch_input_tokens_atts = []
            batch_atts_t5 = []
            batch_inputs_t5 = []

            for b, _ in enumerate(range(B)):
                # batch_input_tokens_input_ids += [input_tokens.input_ids[b]]
                # batch_input_tokens_atts += [input_tokens.attention_mask[b]]
                batch_atts_t5 += [atts_t5[b]]
                batch_inputs_t5 += [inputs_t5[b]]

            # batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
            # batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
            batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
            batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

            # encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)
            encoder_atts = batch_atts_t5

            # inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
            # inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)
            inputs_embeds = batch_inputs_t5

            # outputs = self.t5_model.generate(
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=encoder_atts,
            #     do_sample=False,
            #     num_beams=num_beams,
            #     max_new_tokens=max_len,
            #     min_length=1,
            #     length_penalty=-1,
            # )
            # output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)


            # decoder_inputs_embeds_first_pad = self.t5_model.decoder.embed_tokens(torch.tensor([0], device=h.device))
            # decoder_inputs_embeds_first_pad = decoder_inputs_embeds_first_pad[None, :, :].expand([B, -1, -1])
            # decoder_inputs_embeds_quant_z = decoder_inputs_embeds_first_pad
            # inference
            ego_pre = text_tokens[:, self.mid_time-1:self.mid_time]
            for cur_time in range(self.mid_time, self.end_time):
                if cur_time == self.mid_time:
                    decoder_inputs_embeds_quant_z = self.image_start_chunk
                else:
                    decoder_inputs_embeds_quant_z = torch.cat((decoder_inputs_embeds_quant_z, self.image_start_chunk), dim=1)  # add new start
                
                for _ in range(patchified_length):
                    sequence_output = self.t5_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=encoder_atts,
                        # decoder_attention_mask=output_tokens.attention_mask,
                        return_dict=True,
                        decoder_inputs_embeds=decoder_inputs_embeds_quant_z,
                                        
                        vqgan=True,
                    
                    )

                    lm_logits = self.vqgan_head(sequence_output[:, -1:])

                    # index = lm_logits.detach().argmax(dim=-1)  # 1, 
                    # cur_quant_z = self.first_stage_model.quantize.get_codebook_entry(
                    #     index.reshape(-1), shape=None)
                    # cur_quant_z = cur_quant_z[:, None, :]
                    # quant_z_list.append(cur_quant_z)
                    # cur_quant_z_embeds = self.vqgan_adapter(cur_quant_z)

                    cur_quant_z_embeds = self.get_pred(lm_logits, dynamic=None)
                    if self.ego_type == 'pre_fusion':
                        cur_quant_z_embeds = self.pre_fusion(cur_quant_z_embeds, pre_ego_tokens=ego_pre)
                    decoder_inputs_embeds_quant_z = torch.cat([decoder_inputs_embeds_quant_z, cur_quant_z_embeds], dim=1)

                if self.ego_type == 'place_holder':
                    ego_tokens = torch.zeros_like(self.image_end_chunk) - 100
                else:  # generate ego
                    sequence_output = self.t5_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=encoder_atts,
                        # decoder_attention_mask=output_tokens.attention_mask,
                        return_dict=True,
                        decoder_inputs_embeds=decoder_inputs_embeds_quant_z,
                                        
                        vqgan=True,
                    
                    )
                    ego_pre = self.lm_head(sequence_output[:, -1:])
                    ego_tokens = self.ego_adapter(ego_pre)
                decoder_inputs_embeds_quant_z = torch.cat((decoder_inputs_embeds_quant_z, ego_tokens, self.image_end_chunk), dim=1)  # add new end

        sequence_output = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            # decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            decoder_inputs_embeds=decoder_inputs_embeds_quant_z,
                            
            vqgan=True,
        
        )

        sequence_output_logits, ego_logits = self.split_occ_and_pose(
            sequence_output, patchified_length, 1, self.image_start_chunk.shape[1], split=False)

        sequence_output_logits = self.vqgan_head(sequence_output_logits)
        ego_logits = self.lm_head(ego_logits)

        z_q = self.get_pred_output(sequence_output_logits, latent_shape).detach()
        reconstructed_img = z_q

        # quant_z = torch.cat(quant_z_list, dim=1)
        # quant_z = quant_z.permute(0, 2, 1).contiguous().view(B, 256, 16, 16)
        # reconstructed_img = self.first_stage_model.decode(quant_z)

        # img_feats = samples["images"]
        # img_feats = img_feats.permute(0, 3, 1, 2)

        # _, z_indices = self.encode_to_z(img_feats)
        # targets = z_indices
        # quant_z = self.first_stage_model.quantize.get_codebook_entry(
        #     targets.reshape(-1), shape=(B, 24, 24, 256))
        # reconstructed_img = self.first_stage_model.decode(quant_z)

        # quant_z, z_indices = self.encode_to_z(img_feats)
        # targets = z_indices
        # reconstructed_img = self.first_stage_model.decode(quant_z)        

        titles_gt_past=["GT past 1.0s", "GT past 0.5s", "GT past 0.0s"]
        titles_vqgan_past=["VQGAN past 1.0s", "VQGAN past 0.5s", "GT past 0.0s"]
        titles_gt_future=["GT future 0.5s", "GT future 1.0s", "GT future 1.5s"]
        titles_vqgan_future=["VQGAN future 0.5s", "VQGAN future 1.0s", "GT future 1.5s"]
        titles_llm_future=["LLM future 0.5s", "LLM future 1.0s", "LLM future 1.5s"]


        input_img = samples['images'][0]
        path = samples['path_images']
        size = 256

        x_vqgan_list = []
        for cur_path in path[0]:
            cur_x_vqgan = preprocess(download_image(cur_path), target_image_size=size, map_dalle=False)
            cur_x_vqgan = cur_x_vqgan.to(img_feats.device)
            x_vqgan_list.append(cur_x_vqgan)
        x_vqgan = torch.stack(x_vqgan_list)

        # quant_z, z_indices = self.encode_to_z(x_vqgan)
        # targets = z_indices
        # reconstructed_img_vqgan = self.first_stage_model.decode(quant_z)

        img_gt_past = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0][0])), custom_to_pil(preprocess_vqgan(x_vqgan[1][0])), 
                                    custom_to_pil(preprocess_vqgan(x_vqgan[2][0])), titles=titles_gt_past)

        # img_vqgan_past = stack_reconstructions(custom_to_pil(reconstructed_img_vqgan[0]), custom_to_pil(reconstructed_img_vqgan[1]), 
        #                             custom_to_pil(reconstructed_img_vqgan[2]), titles=titles_vqgan_past)
        
        img_gt_future = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[3][0])), custom_to_pil(preprocess_vqgan(x_vqgan[4][0])), 
                                    custom_to_pil(preprocess_vqgan(x_vqgan[5][0])), titles=titles_gt_future)

        # img_vqgan_future = stack_reconstructions(custom_to_pil(reconstructed_img_vqgan[3]), custom_to_pil(reconstructed_img_vqgan[4]), 
        #                             custom_to_pil(reconstructed_img_vqgan[5]), titles=titles_vqgan_future)

        img_llm_future = stack_reconstructions(custom_to_pil(reconstructed_img[0]), custom_to_pil(reconstructed_img[1]), 
                                    custom_to_pil(reconstructed_img[2]), titles=titles_llm_future)

        # torch.set_printoptions(sci_mode=False)


        # if self._apply_lemmatizer:
        #     output_text_new = self._lemmatize(output_text)
        #     output_text = output_text_new
        # return output_text

        return reconstructed_img
        

    # def predict_answers(
    #     self,
    #     samples,
    #     num_beams=5,
    #     inference_method="generate",
    #     max_len=10,
    #     min_len=1,
    #     num_ans_candidates=128,
    #     answer_list=None,
    #     prompt="",
    #     length_penalty=-1,
    #     **kwargs,
    # ):

    #     img_feats = samples["images"]
    #     img_feats = img_feats.permute(0, 1, 4, 2, 3).contiguous()
    #     B, F, D_in, H, W = img_feats.shape

    #     # get chunks
    #     if self.image_start_chunk is None or self.image_end_chunk is None:
    #         chunks = self.get_tokenized_chunks([IMAGE_START, IMAGE_END], img_feats)
    #         self.image_start_chunk = chunks[0]
    #         self.image_end_chunk = chunks[1]

    #     img_feats = img_feats.view(B*F, D_in, H, W)

    #     # _, z_indices = self.encode_to_z(img_feats)
    #     quant_z, z_indices = self.encode_to_z(img_feats)
    #     _, D, _, _ = quant_z.shape

    #     targets = z_indices.view(quant_z.shape[0], quant_z.shape[2], quant_z.shape[3]).contiguous()
    #     text_tokens = samples["gt_ego_poses"]
    #     text_tokens_mask = samples["gt_ego_poses_mask"]

    #     # h = self.first_stage_model.encoder(img_feats)

    #     # if self.check:
    #     #     print(samples["questions"][0])
    #     #     print(samples["answers"][0])
    #     #     self.check = False
            
    #     device = samples["vfeats"].device
    #     vfeats = samples["vfeats"]

    #     #########################
    #     vfeats = vfeats.squeeze(1)
    #     #########################

    #     B = vfeats.shape[0]
    #     device = vfeats.device
        
    #     # with self.maybe_autocast():
    #     #     if vfeats.dim() == 4:
    #     #         image_embeds = self.ln_vision(self.visual_encoder(vfeats))
    #     #         image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device) # [2, 128]
    #     #         query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    #     #         query_output = self.Qformer.bert(
    #     #                 query_embeds=query_tokens,
    #     #                 encoder_hidden_states=image_embeds,
    #     #                 encoder_attention_mask=image_atts,
    #     #                 return_dict=True,)

    #     #         t5_query = query_output.last_hidden_state
    #     #     else:
    #     #         if "timesteps" in samples:
    #     #             timesteps = samples["timesteps"].to(device)
    #     #         else:
    #     #             Time = vfeats.shape[1]
    #     #             timesteps = 0
    #     #         tmp_list = []
    #     #         index = 0
    #     #         for adj_img in vfeats[:,:-1,...].split(1, dim=1):
    #     #             adj_img = adj_img.squeeze(1)
    #     #             image_embeds = self.ln_vision(self.visual_encoder(adj_img))
    #     #             image_embeds = self.slot_attention(image_embeds)
    #     #             tmp_list.append(torch.unsqueeze(image_embeds, dim=1))
                
    #     #         image_embeds = self.ln_vision(self.visual_encoder(vfeats[:,-1,...]))
    #     #         image_embeds = self.slot_attention(image_embeds)
    #     #         tmp_list.append(torch.unsqueeze(image_embeds, dim=1))
                
    #     #         tmp_query = torch.cat(tmp_list, dim=1)
    #     #         tmp_query = tmp_query.reshape(tmp_query.shape[0], -1, tmp_query.shape[-1])
    #     #         image_atts = torch.ones(tmp_query.size()[:-1], dtype=torch.long).to(device) # [2, 128]

    #     #         query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
    #     #         query_tokens = query_tokens.expand(tmp_query.shape[0], -1, -1)
    #     #         query_output = self.Qformer.bert(
    #     #                 query_embeds=query_tokens,
    #     #                 encoder_hidden_states=tmp_query,
    #     #                 encoder_attention_mask=image_atts,
    #     #                 return_dict=True,)

    #     #         t5_query = query_output.last_hidden_state

    #     # inputs_t5 = self.t5_proj(t5_query)
    #     # inputs_t5 = self.vqgan_adapter(h.view(B, 1, -1))
    #     hidden_img_feats = quant_z.permute(0, 2, 3, 1).contiguous()
    #     hidden_img_feats = hidden_img_feats.view(B, F, *hidden_img_feats.shape[1:])
    #     latent_shape = hidden_img_feats.shape[2:4]  # get original latent h and w
        
    #     patchified_inputs, patchified_length = self.process_input_split(hidden_img_feats, text_tokens)

    #     inputs_t5 = patchified_inputs[:, :self.mid_time]
    #     inputs_t5 = inputs_t5.view(B, -1, inputs_t5.shape[-1])
    #     atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)
        
    #     # text_input = samples["questions"]
    #     with torch.cuda.amp.autocast(dtype=torch.float32):
    #         # input_tokens = self.t5_tokenizer( # 8 x 17
    #         #     text_input,
    #         #     padding="longest",
    #         #     truncation=True,
    #         #     max_length=300,
    #         #     return_tensors="pt",
    #         # ).to(device)

    #         batch_input_tokens_input_ids = []
    #         batch_input_tokens_atts = []
    #         batch_atts_t5 = []
    #         batch_inputs_t5 = []

    #         for b, _ in enumerate(range(B)):
    #             # batch_input_tokens_input_ids += [input_tokens.input_ids[b]]
    #             # batch_input_tokens_atts += [input_tokens.attention_mask[b]]
    #             batch_atts_t5 += [atts_t5[b]]
    #             batch_inputs_t5 += [inputs_t5[b]]

    #         # batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
    #         # batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
    #         batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
    #         batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

    #         # encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)
    #         encoder_atts = batch_atts_t5

    #         # inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
    #         # inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)
    #         inputs_embeds = batch_inputs_t5

    #         # outputs = self.t5_model.generate(
    #         #     inputs_embeds=inputs_embeds,
    #         #     attention_mask=encoder_atts,
    #         #     do_sample=False,
    #         #     num_beams=num_beams,
    #         #     max_new_tokens=max_len,
    #         #     min_length=1,
    #         #     length_penalty=-1,
    #         # )
    #         # output_text = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)


    #         # decoder_inputs_embeds_first_pad = self.t5_model.decoder.embed_tokens(torch.tensor([0], device=h.device))
    #         # decoder_inputs_embeds_first_pad = decoder_inputs_embeds_first_pad[None, :, :].expand([B, -1, -1])
    #         # decoder_inputs_embeds_quant_z = decoder_inputs_embeds_first_pad
    #         # quant_z_list = []
    #         # for cur_time in range(self.mid_time, self.end_time):
    #         #     if cur_time == self.mid_time:
    #         #         decoder_inputs_embeds_quant_z = self.image_start_chunk
    #         #     else:
    #         #         decoder_inputs_embeds_quant_z = torch.cat((decoder_inputs_embeds_quant_z, self.image_start_chunk), dim=1)  # add new start
                
    #         #     for _ in range(patchified_length):
    #         #         sequence_output = self.t5_model(
    #         #             inputs_embeds=inputs_embeds,
    #         #             attention_mask=encoder_atts,
    #         #             # decoder_attention_mask=output_tokens.attention_mask,
    #         #             return_dict=True,
    #         #             decoder_inputs_embeds=decoder_inputs_embeds_quant_z,
                                        
    #         #             vqgan=True,
                    
    #         #         )

    #         #         lm_logits = self.vqgan_head(sequence_output[:, -1:])

    #         #         # index = lm_logits.detach().argmax(dim=-1)  # 1, 
    #         #         # cur_quant_z = self.first_stage_model.quantize.get_codebook_entry(
    #         #         #     index.reshape(-1), shape=None)
    #         #         # cur_quant_z = cur_quant_z[:, None, :]
    #         #         # quant_z_list.append(cur_quant_z)
    #         #         # cur_quant_z_embeds = self.vqgan_adapter(cur_quant_z)

    #         #         cur_quant_z_embeds = self.get_pred(lm_logits, dynamic=None)
    #         #         decoder_inputs_embeds_quant_z = torch.cat([decoder_inputs_embeds_quant_z, cur_quant_z_embeds], dim=1)

    #         #     decoder_inputs_embeds_quant_z = torch.cat((decoder_inputs_embeds_quant_z, self.image_end_chunk), dim=1)  # add new end

    #     decoder_inputs_embeds_quant_z = patchified_inputs[:, self.mid_time:]
    #     decoder_inputs_embeds_quant_z = decoder_inputs_embeds_quant_z.view(B, -1, decoder_inputs_embeds_quant_z.shape[-1])

    #     sequence_output = self.t5_model(
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=encoder_atts,
    #         # decoder_attention_mask=output_tokens.attention_mask,
    #         return_dict=True,
    #         decoder_inputs_embeds=decoder_inputs_embeds_quant_z,
                            
    #         vqgan=True,
        
    #     )

    #     sequence_output_logits, ego_logits = self.split_occ_and_pose(
    #         sequence_output, patchified_length, 1, self.image_start_chunk.shape[1], split=False)

    #     sequence_output_logits = self.vqgan_head(sequence_output_logits)
    #     ego_logits = self.lm_head(ego_logits)

    #     z_q = self.get_pred_output(sequence_output_logits, latent_shape).detach()
    #     reconstructed_img = z_q

    #     # quant_z = torch.cat(quant_z_list, dim=1)
    #     # quant_z = quant_z.permute(0, 2, 1).contiguous().view(B, 256, 16, 16)
    #     # reconstructed_img = self.first_stage_model.decode(quant_z)

    #     # img_feats = samples["images"]
    #     # img_feats = img_feats.permute(0, 3, 1, 2)

    #     # _, z_indices = self.encode_to_z(img_feats)
    #     # targets = z_indices
    #     # quant_z = self.first_stage_model.quantize.get_codebook_entry(
    #     #     targets.reshape(-1), shape=(B, 24, 24, 256))
    #     # reconstructed_img = self.first_stage_model.decode(quant_z)

    #     # quant_z, z_indices = self.encode_to_z(img_feats)
    #     # targets = z_indices
    #     # reconstructed_img = self.first_stage_model.decode(quant_z)        

    #     titles=["Input", "VQGAN (f16, 1024)"]
    #     input_img = samples['images'][0]
    #     path = samples['path_images']
    #     size = 256

    #     x_vqgan_list = []
    #     for cur_path in path[0]:
    #         cur_x_vqgan = preprocess(download_image(cur_path), target_image_size=size, map_dalle=False)
    #         cur_x_vqgan = cur_x_vqgan.to(img_feats.device)
    #         x_vqgan_list.append(cur_x_vqgan)
    #     x_vqgan = torch.stack(x_vqgan_list)

    #     # quant_z, z_indices = self.encode_to_z(x_vqgan)
    #     # targets = z_indices
    #     # reconstructed_img = self.first_stage_model.decode(quant_z)

    #     img_0 = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[3][0])), 
    #                                 custom_to_pil(reconstructed_img[0]), titles=titles)

    #     img_1 = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[4][0])), 
    #                                 custom_to_pil(reconstructed_img[1]), titles=titles)
        
    #     img_2 = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[5][0])), 
    #                                 custom_to_pil(reconstructed_img[2]), titles=titles)

    #     # img_3 = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[3][0])), 
    #     #                             custom_to_pil(reconstructed_img[3]), titles=titles)



    #     # if self._apply_lemmatizer:
    #     #     output_text_new = self._lemmatize(output_text)
    #     #     output_text = output_text_new
    #     # return output_text

    #     return reconstructed_img


    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        first_stage_config = cfg.get("first_stage_config")

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,

            first_stage_config=first_stage_config,

        )
        model.load_checkpoint_from_config(cfg)

        # Lora
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # for param in model.slot_attention.parameters():
        #     param.requires_grad = True

        # model.extra_query_tokens.requires_grad = True

        model.register_vqgan(first_stage_config)
        model.first_stage_model.requires_grad_(False)

        return model