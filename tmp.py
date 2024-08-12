    def forward(self, samples):
        img_feats = samples["images"]
        img_feats = img_feats.permute(0, 3, 1, 2)
        # _, z_indices = self.encode_to_z(img_feats)
        quant_z, z_indices = self.encode_to_z(img_feats)

        targets = z_indices

        h = self.first_stage_model.encoder(img_feats)

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
        h = h.permute(0, 2, 3, 1).contiguous()
        inputs_t5 = self.vqgan_adapter(h.view(B, -1, 256))
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

            decoder_inputs_embeds_first_pad = self.t5_model.decoder.embed_tokens(torch.tensor([0], device=h.device))
            decoder_inputs_embeds_first_pad = decoder_inputs_embeds_first_pad[None, :, :].expand([B, -1, -1])
            quant_z = quant_z.permute(0, 2, 3, 1).contiguous().view(B, -1, 256)
            embeds_quant_z = self.vqgan_adapter(quant_z)
            decoder_inputs_embeds_quant_z = torch.cat([decoder_inputs_embeds_first_pad, embeds_quant_z[:, :-1]], dim=1)
            sequence_output = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                # decoder_attention_mask=output_tokens.attention_mask,
                return_dict=True,
                decoder_inputs_embeds=decoder_inputs_embeds_quant_z,
                                
                vqgan=True,
            
            )

            labels = targets
            reduction = "mean"
            lm_logits = self.vqgan_head(sequence_output)
            lm_logits = lm_logits.view(B, 16*16, 1024)

            # index = lm_logits.detach().argmax(dim=-1) 

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100, reduction=reduction)
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                if reduction == "none":
                    loss = loss.view(lm_logits.size(0), -1).sum(1)

            return {"loss": loss}