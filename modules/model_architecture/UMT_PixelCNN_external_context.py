from modules.model_architecture.common import RobertaPreTrainedModel, RobertaModel, ImageDecoder, RobertaSelfEncoder,RobertaCrossEncoder, LOSS_TI
import torch
from torch import nn
from modules.model_architecture.torchcrf import CRF

class UMT_PixelCNN(RobertaPreTrainedModel):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """
    def __init__(self, config, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels_=2, auxnum_labels=2):
        super(UMT_PixelCNN, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = RobertaModel(config)
        #self.trans_matrix = torch.zeros(num_labels, auxnum_labels)
        self.image_decoder = ImageDecoder(nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                                 resnet_nonlinearity='concat_elu', input_channels=3)
        self.self_attention = RobertaSelfEncoder(config)
        self.self_attention_v2 = RobertaSelfEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        self.vismap2text_v2 = nn.Linear(2048, config.hidden_size)
        self.txt2img_attention = RobertaCrossEncoder(config, layer_num1)


        
        self.text_dense_cl = nn.Linear(config.hidden_size, config.hidden_size)
        self.text_ouput_cl = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.image_dense_cl = nn.Linear(2048, config.hidden_size)
        self.image_output_cl = nn.Linear(config.hidden_size, config.hidden_size)

        self.img2txt_attention = RobertaCrossEncoder(config, layer_num2)
        self.txt2txt_attention = RobertaCrossEncoder(config, layer_num3)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        ### self.self_attention = BertLastSelfAttention(config)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels_)
        self.aux_classifier = nn.Linear(config.hidden_size, auxnum_labels)

        self.crf = CRF(num_labels_, batch_first=True)
        self.aux_crf = CRF(auxnum_labels, batch_first=True)

        initial_coefficients = torch.tensor([1.0, 0.05, 0.5, 0.005, 1.0, 0.05, 0.5])
        # Add a small epsilon to prevent log(0) if any coefficient was exactly 0
        epsilon = 1e-8
        initial_coefficients = initial_coefficients + epsilon
        # Convert coefficients to logits (log because of softmax)
        # Adding a constant (like mean) is optional for initialization but can keep values smaller
        initial_logits = torch.log(initial_coefficients) # + torch.log(initial_coefficients).mean() # Optional centering
        # Create the learnable parameter
        self.loss_weights = nn.Parameter(initial_logits)

        self.init_weights()

    def text_toimage_loss(self,text_h1, image_h1, temp):
        batch_size = text_h1.shape[0]
        loss = 0
        for i in range(batch_size):
            up = torch.exp(
                (torch.matmul(text_h1[i], image_h1[i]) / (torch.norm(text_h1[i]) * torch.norm(image_h1[i]))) / temp
            )

            down = torch.sum(
                torch.exp((torch.sum(text_h1[i] * image_h1, dim=-1) / (
                            torch.norm(text_h1[i]) * torch.norm(image_h1, dim=1))) / temp), dim=-1)

            loss += -torch.log(up / down)

        return loss

    def image_totext_loss(self,text_h1, image_h1, temp):
        batch_size = text_h1.shape[0]
        loss = 0
        for i in range(batch_size):
            up = torch.exp(
                (
                        torch.matmul(image_h1[i], text_h1[i]) / (torch.norm(image_h1[i]) * torch.norm(text_h1[i]))
                ) / temp
            )

            down = torch.sum(
                torch.exp((torch.sum(image_h1[i] * text_h1, dim=-1) / (
                            torch.norm(image_h1[i]) * torch.norm(text_h1, dim=1))) / temp), dim=-1)

            loss += -torch.log(up / down)

        return loss

    def total_loss(self,text_h1, image_h1, temp, temp_lamb):
        # lamb = 0.5
        lamb = temp_lamb
        batch_size = text_h1.shape[0]
        loss = (1 / batch_size) * (
                    lamb * self.text_toimage_loss(text_h1, image_h1, temp) + (1 - lamb) * self.image_totext_loss(text_h1, image_h1, temp))
        # print("total_loss:",loss)
        return loss

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(self, input_ids_external, segment_ids_external, input_mask_external, added_attention_mask_external, visual_embeds_mean, visual_embeds_att, trans_matrix, added_attention_mask_origin ,input_ids_origin = None, segment_ids_origin = None, input_mask_origin = None, 
                image_decode=None, alpha=None, temp=None, temp_lamb=None, labels_external=None, auxlabels_external=None, labels_origin=None, auxlabels_origin=None):

        # Get the emission scores from the BiLSTM
        features_external = self.roberta(input_ids_external, token_type_ids=segment_ids_external, attention_mask=input_mask_external)  # batch_size * seq_len * hidden_size
        sequence_output_external = features_external["last_hidden_state"]
        pooler_output_external = features_external["pooler_output"]
        sequence_output_external = self.dropout(sequence_output_external)
        pooler_output_external = self.linear(pooler_output_external)

        extended_txt_mask_external = input_mask_external.unsqueeze(1).unsqueeze(2)
        extended_txt_mask_external = extended_txt_mask_external.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask_external = (1.0 - extended_txt_mask_external) * -10000.0
        aux_addon_sequence_encoder_external = self.self_attention(sequence_output_external, extended_txt_mask_external)

        aux_addon_sequence_output_external = aux_addon_sequence_encoder_external[-1]
        aux_addon_sequence_output_external = aux_addon_sequence_output_external[0]
        aux_bert_feats_external = self.aux_classifier(aux_addon_sequence_output_external)
        #######aux_bert_feats = self.aux_classifier(sequence_output)
        trans_bert_feats_external = torch.matmul(aux_bert_feats_external, trans_matrix.float())

        main_addon_sequence_encoder_external = self.self_attention_v2(sequence_output_external, extended_txt_mask_external)
        main_addon_sequence_output_external = main_addon_sequence_encoder_external[-1]
        main_addon_sequence_output_external = main_addon_sequence_output_external[0]
        vis_embed_map_external = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
        converted_vis_embed_map_external = self.vismap2text(vis_embed_map_external)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask_external = added_attention_mask_external[:,:49]
        extended_img_mask_external = img_mask_external.unsqueeze(1).unsqueeze(2)
        extended_img_mask_external = extended_img_mask_external.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_img_mask_external = (1.0 - extended_img_mask_external) * -10000.0

        cross_encoder_external = self.txt2img_attention(main_addon_sequence_output_external, converted_vis_embed_map_external, extended_img_mask_external)
        cross_output_layer_external = cross_encoder_external[-1]  # self.batch_size * text_len * hidden_dim

        # apply img2txt attention mechanism to obtain multimodal-based text representations
        converted_vis_embed_map_v2_external = self.vismap2text_v2(vis_embed_map_external)  # self.batch_size, 49, hidden_dim

        cross_txt_encoder_external = self.img2txt_attention(converted_vis_embed_map_v2_external, main_addon_sequence_output_external, extended_txt_mask_external)
        cross_txt_output_layer_external = cross_txt_encoder_external[-1]  # self.batch_size * 49 * hidden_dim
        cross_final_txt_encoder_external = self.txt2txt_attention(main_addon_sequence_output_external, cross_txt_output_layer_external, extended_img_mask_external)
        ##cross_final_txt_encoder = self.txt2txt_attention(aux_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        cross_final_txt_layer_external = cross_final_txt_encoder_external[-1]  # self.batch_size * text_len * hidden_dim
        #cross_final_txt_layer = torch.add(cross_final_txt_layer, sequence_output)

        # visual gate
        merge_representation_external = torch.cat((cross_final_txt_layer_external, cross_output_layer_external), dim=-1)
        gate_value_external = torch.sigmoid(self.gate(merge_representation_external))  # batch_size, text_len, hidden_dim
        gated_converted_att_vis_embed_external = torch.mul(gate_value_external, cross_output_layer_external)
        # reverse_gate_value = torch.neg(gate_value).add(1)
        # gated_converted_att_vis_embed = torch.add(torch.mul(reverse_gate_value, cross_final_txt_layer),
                                        # torch.mul(gate_value, cross_output_layer))

        # direct concatenation
        # gated_converted_att_vis_embed = self.dropout(gated_converted_att_vis_embed)
        final_output_external = torch.cat((cross_final_txt_layer_external, gated_converted_att_vis_embed_external), dim=-1)
        ###### final_output = self.dropout(final_output)
        #middle_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)
        #final_output = torch.cat((sequence_output, middle_output), dim=-1)

        ###### addon_sequence_output = self.self_attention(final_output, extended_txt_mask)
        bert_feats_external = self.classifier(final_output_external)
        final_bert_feats_external = torch.add(torch.mul(bert_feats_external, alpha),torch.mul(trans_bert_feats_external, 1-alpha))

        # suggested by Hongjie
        #bert_feats = F.log_softmax(bert_feats, dim=-1)

        if labels_external is not None:
            # orgin context
            features_origin = self.roberta(input_ids_origin, token_type_ids=segment_ids_origin, attention_mask=input_mask_origin)  # batch_size * seq_len * hidden_size
            sequence_output_origin = features_origin["last_hidden_state"]
            pooler_output_origin = features_origin["pooler_output"]
            sequence_output_origin = self.dropout(sequence_output_origin)
            pooler_output_origin = self.linear(pooler_output_origin)
            extended_txt_mask_origin = input_mask_origin.unsqueeze(1).unsqueeze(2)
            extended_txt_mask_origin = extended_txt_mask_origin.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_txt_mask_origin = (1.0 - extended_txt_mask_origin) * -10000.0
            aux_addon_sequence_encoder_origin = self.self_attention(sequence_output_origin, extended_txt_mask_origin)
            aux_addon_sequence_output_origin = aux_addon_sequence_encoder_origin[-1]
            aux_addon_sequence_output_origin = aux_addon_sequence_output_origin[0]
            aux_bert_feats_origin = self.aux_classifier(aux_addon_sequence_output_origin)
            #######aux_bert_feats = self.aux_classifier(sequence_output)
            trans_bert_feats_origin = torch.matmul(aux_bert_feats_origin, trans_matrix.float())
            main_addon_sequence_encoder_origin = self.self_attention_v2(sequence_output_origin, extended_txt_mask_origin)
            main_addon_sequence_output_origin = main_addon_sequence_encoder_origin[-1]
            main_addon_sequence_output_origin = main_addon_sequence_output_origin[0]
            vis_embed_map_origin = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
            converted_vis_embed_map_origin = self.vismap2text(vis_embed_map_origin)  # self.batch_size, 49, hidden_dim
            # apply txt2img attention mechanism to obtain image-based text representations
            img_mask_origin = added_attention_mask_origin[:,:49]
            extended_img_mask_origin = img_mask_origin.unsqueeze(1).unsqueeze(2)
            extended_img_mask_origin = extended_img_mask_origin.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_img_mask_origin = (1.0 - extended_img_mask_origin) * -10000.0
            cross_encoder_origin = self.txt2img_attention(main_addon_sequence_output_origin, converted_vis_embed_map_origin, extended_img_mask_origin)
            cross_output_layer_origin = cross_encoder_origin[-1]  # self.batch_size * text_len * hidden_dim
            # apply img2txt attention mechanism to obtain multimodal-based text representations
            converted_vis_embed_map_v2_origin = self.vismap2text_v2(vis_embed_map_origin)  # self.batch_size, 49, hidden_dim
            cross_txt_encoder_origin = self.img2txt_attention(converted_vis_embed_map_v2_origin, main_addon_sequence_output_origin, extended_txt_mask_origin)
            cross_txt_output_layer_origin = cross_txt_encoder_origin[-1]  # self.batch_size * 49 * hidden_dim
            cross_final_txt_encoder_origin = self.txt2txt_attention(main_addon_sequence_output_origin, cross_txt_output_layer_origin, extended_img_mask_origin)
            cross_final_txt_layer_origin = cross_final_txt_encoder_origin[-1]  # self.batch_size * text_len * hidden_dim
            # visual gate
            merge_representation_origin = torch.cat((cross_final_txt_layer_origin, cross_output_layer_origin), dim=-1)
            gate_value_origin = torch.sigmoid(self.gate(merge_representation_origin))  # batch_size, text_len, hidden_dim
            gated_converted_att_vis_embed_origin = torch.mul(gate_value_origin, cross_output_layer_origin)
            # direct concatenation
            # gated_converted_att_vis_embed = self.dropout(gated_converted_att_vis_embed)
            final_output_origin = torch.cat((cross_final_txt_layer_origin, gated_converted_att_vis_embed_origin), dim=-1)
            ###### final_output = self.dropout(final_output)
            #middle_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)
            #final_output = torch.cat((sequence_output, middle_output), dim=-1)
            ###### addon_sequence_output = self.self_attention(final_output, extended_txt_mask)
            bert_feats_origin = self.classifier(final_output_origin)
            final_bert_feats_origin = torch.add(torch.mul(bert_feats_origin, alpha),torch.mul(trans_bert_feats_origin, 1-alpha))
            
            
            # --- Tính toán các thành phần Loss (giữ nguyên) ---
            # Loss origin 1
            aux_loss_origin = - self.aux_crf(aux_bert_feats_origin, auxlabels_origin, mask=input_mask_origin.byte(), reduction='mean')
            # Loss origin 2
            main_loss_origin = - self.crf(final_bert_feats_origin, labels_origin, mask=input_mask_origin.byte(), reduction='mean')
            # Loss origin 3
            text_output_cl_origin = self.text_ouput_cl(self.relu(self.text_dense_cl(pooler_output_origin)))
            image_ouput_cl_origin = self.image_output_cl(self.relu(self.image_dense_cl(visual_embeds_mean)))
            cl_loss_origin = self.total_loss(text_output_cl_origin, image_ouput_cl_origin, temp, temp_lamb)
            
            # Loss external 1
            aux_loss_external = - self.aux_crf(aux_bert_feats_external, auxlabels_external, mask=input_mask_external.byte(), reduction='mean')
            # Loss external 2
            main_loss_external = - self.crf(final_bert_feats_external, labels_external, mask=input_mask_external.byte(), reduction='mean')
            # Loss external 3
            image_generate = self.image_decoder(x=image_decode, h=pooler_output_external)
            loss_ti = LOSS_TI(image_decode, image_generate)
            # Loss external 4
            text_output_cl_external = self.text_ouput_cl(self.relu(self.text_dense_cl(pooler_output_external)))
            image_ouput_cl_external = self.image_output_cl(self.relu(self.image_dense_cl(visual_embeds_mean)))
            cl_loss_external = self.total_loss(text_output_cl_external, image_ouput_cl_external, temp, temp_lamb)


            normalized_weights = F.softmax(self.loss_weights, dim=0) # Shape: [7]
            loss_components = torch.stack([
                main_loss_external,
                cl_loss_external,  
                aux_loss_external, 
                loss_ti,           
                main_loss_origin,  
                cl_loss_origin,    
                aux_loss_origin    
            ])

            weighted_losses = normalized_weights * loss_components
            loss = torch.sum(weighted_losses)

            # with torch.no_grad():
            #     print(f"Các trọng số Loss (Softmax): {normalized_weights.detach().cpu().numpy()}")

            return loss
        else:
            pred_tags = self.crf.decode(final_bert_feats_external, mask=input_mask_external.byte())
            return pred_tags



if __name__ == "__main__": 
    from modules.model_architecture.helper import reinit_custom_modules
    model = UMT_PixelCNN.from_pretrained('vinai/phobert-base-v2',cache_dir='cache', layer_num1=1, layer_num2=1, layer_num3=1, num_labels_=13, auxnum_labels=7)
    # model.to('cuda')
    reinit_custom_modules(model)
    # Check for NaN values
    aaa =[]
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            aaa.append(f"NaN found in {name}")
        if torch.isinf(param).any():
            aaa.append(f"Inf found in {name}")

    if aaa:
        print(aaa)