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

        self.init_weights()

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(self, input_ids, segment_ids, input_mask, added_attention_mask, visual_embeds_mean, visual_embeds_att, trans_matrix, 
                image_decode=None, alpha=None, beta=None, theta=None, sigma=None, temp=None, temp_lamb=None, labels=None, auxlabels=None):

        # Get the emission scores from the BiLSTM
        features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)  # batch_size * seq_len * hidden_size
        sequence_output = features["last_hidden_state"]
        pooler_output = features["pooler_output"]
        sequence_output = self.dropout(sequence_output)
        pooler_output = self.linear(pooler_output)


        extended_txt_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_txt_mask = extended_txt_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_txt_mask = (1.0 - extended_txt_mask) * -10000.0
        aux_addon_sequence_encoder = self.self_attention(sequence_output, extended_txt_mask)

        aux_addon_sequence_output = aux_addon_sequence_encoder[-1]
        aux_addon_sequence_output = aux_addon_sequence_output[0]
        aux_bert_feats = self.aux_classifier(aux_addon_sequence_output)
        #######aux_bert_feats = self.aux_classifier(sequence_output)
        trans_bert_feats = torch.matmul(aux_bert_feats, trans_matrix.float())

        main_addon_sequence_encoder = self.self_attention_v2(sequence_output, extended_txt_mask)
        main_addon_sequence_output = main_addon_sequence_encoder[-1]
        main_addon_sequence_output = main_addon_sequence_output[0]
        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:,:49]
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        cross_encoder = self.txt2img_attention(main_addon_sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]  # self.batch_size * text_len * hidden_dim

        # apply img2txt attention mechanism to obtain multimodal-based text representations
        converted_vis_embed_map_v2 = self.vismap2text_v2(vis_embed_map)  # self.batch_size, 49, hidden_dim

        cross_txt_encoder = self.img2txt_attention(converted_vis_embed_map_v2, main_addon_sequence_output, extended_txt_mask)
        cross_txt_output_layer = cross_txt_encoder[-1]  # self.batch_size * 49 * hidden_dim
        cross_final_txt_encoder = self.txt2txt_attention(main_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        ##cross_final_txt_encoder = self.txt2txt_attention(aux_addon_sequence_output, cross_txt_output_layer, extended_img_mask)
        cross_final_txt_layer = cross_final_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim
        #cross_final_txt_layer = torch.add(cross_final_txt_layer, sequence_output)

        # visual gate
        merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        gated_converted_att_vis_embed = torch.mul(gate_value, cross_output_layer)
        # reverse_gate_value = torch.neg(gate_value).add(1)
        # gated_converted_att_vis_embed = torch.add(torch.mul(reverse_gate_value, cross_final_txt_layer),
                                        # torch.mul(gate_value, cross_output_layer))

        # direct concatenation
        # gated_converted_att_vis_embed = self.dropout(gated_converted_att_vis_embed)
        final_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)
        ###### final_output = self.dropout(final_output)
        #middle_output = torch.cat((cross_final_txt_layer, gated_converted_att_vis_embed), dim=-1)
        #final_output = torch.cat((sequence_output, middle_output), dim=-1)

        ###### addon_sequence_output = self.self_attention(final_output, extended_txt_mask)
        bert_feats = self.classifier(final_output)


        final_bert_feats = torch.add(torch.mul(bert_feats, alpha),torch.mul(trans_bert_feats, 1-alpha))

        # suggested by Hongjie
        #bert_feats = F.log_softmax(bert_feats, dim=-1)

        if labels is not None:

            # Loss 1
            aux_loss = - self.aux_crf(aux_bert_feats, auxlabels, mask=input_mask.byte(), reduction='mean')
            # Loss 2
            main_loss = - self.crf(final_bert_feats, labels, mask=input_mask.byte(), reduction='mean')
            # Loss 3
            image_generate = self.image_decoder(x=image_decode, h=pooler_output)
            loss_ti = LOSS_TI(image_decode, image_generate)

            # print(f"aux_loss: {aux_loss}, main_loss: {main_loss}, loss_ti: {loss_ti}, cl_loss: {cl_loss}")
            loss = main_loss + beta*aux_loss + loss_ti*sigma 
            return loss
        else:
            pred_tags = self.crf.decode(final_bert_feats, mask=input_mask.byte())
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