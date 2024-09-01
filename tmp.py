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
        
        patchified_inputs, patchified_length = self.process_input_split(hidden_img_feats, text_tokens)

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
            # quant_z_list = []
            # for cur_time in range(self.mid_time, self.end_time):
            #     if cur_time == self.mid_time:
            #         decoder_inputs_embeds_quant_z = self.image_start_chunk
            #     else:
            #         decoder_inputs_embeds_quant_z = torch.cat((decoder_inputs_embeds_quant_z, self.image_start_chunk), dim=1)  # add new start
                
            #     for _ in range(patchified_length):
            #         sequence_output = self.t5_model(
            #             inputs_embeds=inputs_embeds,
            #             attention_mask=encoder_atts,
            #             # decoder_attention_mask=output_tokens.attention_mask,
            #             return_dict=True,
            #             decoder_inputs_embeds=decoder_inputs_embeds_quant_z,
                                        
            #             vqgan=True,
                    
            #         )

            #         lm_logits = self.vqgan_head(sequence_output[:, -1:])

            #         # index = lm_logits.detach().argmax(dim=-1)  # 1, 
            #         # cur_quant_z = self.first_stage_model.quantize.get_codebook_entry(
            #         #     index.reshape(-1), shape=None)
            #         # cur_quant_z = cur_quant_z[:, None, :]
            #         # quant_z_list.append(cur_quant_z)
            #         # cur_quant_z_embeds = self.vqgan_adapter(cur_quant_z)

            #         cur_quant_z_embeds = self.get_pred(lm_logits, dynamic=None)
            #         decoder_inputs_embeds_quant_z = torch.cat([decoder_inputs_embeds_quant_z, cur_quant_z_embeds], dim=1)

            #     decoder_inputs_embeds_quant_z = torch.cat((decoder_inputs_embeds_quant_z, self.image_end_chunk), dim=1)  # add new end

        decoder_inputs_embeds_quant_z = patchified_inputs[:, self.mid_time:]
        decoder_inputs_embeds_quant_z = decoder_inputs_embeds_quant_z.view(B, -1, decoder_inputs_embeds_quant_z.shape[-1])

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
        # ego_logits = self.lm_head(ego_logits)

        loss_image = self.calc_loss_image(sequence_output_logits, labels)

        6.6035
        3.6061

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