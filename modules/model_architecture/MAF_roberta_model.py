from transformers import RobertaModel

import copy
from transformers.models.roberta.modeling_roberta import RobertaIntermediate,RobertaOutput,RobertaSelfOutput,RobertaPreTrainedModel
from transformers import BertConfig
from torch import nn as nn
from torchcrf import CRF
import torch
import math
# ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class RobertaCoAttention(nn.Module):
    def __init__(self, config):
        super(RobertaCoAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):

        mixed_query_layer = self.query(s1_hidden_states)
        mixed_key_layer = self.key(s2_hidden_states)
        mixed_value_layer = self.value(s2_hidden_states)



        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)



        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + s2_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class RobertaCrossAttention(nn.Module):
    def __init__(self, config):
        super(RobertaCrossAttention, self).__init__()
        self.self = RobertaCoAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, s1_input_tensor, s2_input_tensor, s2_attention_mask):
        s1_cross_output = self.self(s1_input_tensor, s2_input_tensor, s2_attention_mask)
        attention_output = self.output(s1_cross_output, s1_input_tensor)
        return attention_output


class RobertaCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(RobertaCrossAttentionLayer, self).__init__()
        self.attention = RobertaCrossAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask):
        attention_output = self.attention(s1_hidden_states, s2_hidden_states, s2_attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class RobertaCrossEncoder(nn.Module):
    def __init__(self, config, layer_num):
        super(RobertaCrossEncoder, self).__init__()
        layer = RobertaCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_num)])

    def forward(self, s1_hidden_states, s2_hidden_states, s2_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            s1_hidden_states = layer_module(s1_hidden_states, s2_hidden_states, s2_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(s1_hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(s1_hidden_states)
        return all_encoder_layers

class MTCCMRobertaForMMTokenClassificationCRF(RobertaPreTrainedModel):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """
    def __init__(self, config, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels=2, auxnum_labels=2):
        super(MTCCMRobertaForMMTokenClassificationCRF, self).__init__(config)
        self.num_labels = num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        self.txt2img_attention = RobertaCrossEncoder(config, layer_num1)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.sigmoid = nn.Sigmoid()

        # 用于对比学习训练的头
        self.text_dense_cl = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.text_ouput_cl = nn.Linear(config.hidden_size, config.hidden_size)

        self.image_dense_cl = nn.Linear(2048, config.hidden_size)
        self.image_output_cl = nn.Linear(config.hidden_size, config.hidden_size)

        # 用于计算门单元
        self.Gate_text = nn.Linear(config.hidden_size,config.hidden_size)
        self.Gate_image = nn.Linear(config.hidden_size,config.hidden_size)

        
        self.init_weights()

    def text_toimage_loss(self,text_h1, image_h1, temp):
        # temp = 0.1
        batch_size = text_h1.shape[0]
        # text_h1_copy=text_h1
        # image_h1_copy=image_h1
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
        # temp = 0.1
        batch_size = text_h1.shape[0]
        # text_h1_copy=text_h1
        # image_h1_copy=image_h1
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
    def forward(self, input_ids, segment_ids, input_mask, added_attention_mask, visual_embeds_mean, visual_embeds_att, trans_matrix,temp=None,
                temp_lamb=None,labels=None, auxlabels=None):
        # 获得文本表示
        features= self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask) # batch_size * seq_len * hidden_size
        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)
        sequence_output_pooler = features["pooler_output"]

        # 获取图像的表示，分为49个区域，每个区域用2048维度的向量表示
        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)  # self.batch_size, 49, 2048
        # 将图像的维度映射到与bert维度相同，方便进行注意力计算
        converted_vis_embed_map = self.vismap2text(vis_embed_map)  # self.batch_size, 49, hidden_dim

        # '''
        # apply txt2img attention mechanism to obtain image-based text representations
        img_mask = added_attention_mask[:,:49]  # batch_size * 49
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2) # batch_size * 1 * 1 * 49
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        cross_encoder = self.txt2img_attention(sequence_output, converted_vis_embed_map, extended_img_mask)
        # 获取text-aware image 表示
        cross_output_layer = cross_encoder[-1]  # self.batch_size * text_len * hidden_dim

        # 计算门单元
        Gate = self.sigmoid((self.Gate_text(sequence_output) + self.Gate_image(cross_output_layer)))


        # 获得通过门单元的图像的表示
        gated_converted_att_vis_embed = Gate * cross_output_layer


        final_output = torch.cat((sequence_output, gated_converted_att_vis_embed), dim=-1) # batch_size * seq_len * 2(hidden_size)
        bert_feats = self.classifier(final_output)  # batch_size * seq_len * 13
        final_bert_feats = bert_feats

        if labels is not None:
            # 计算对比学习的损失
            text_output_cl = self.text_ouput_cl(self.relu(self.text_dense_cl(sequence_output_pooler)))
            image_ouput_cl = self.image_output_cl(self.relu(self.image_dense_cl(visual_embeds_mean)))
            cl_loss = self.total_loss(text_output_cl, image_ouput_cl, temp, temp_lamb)
            print(final_bert_feats.shape)
            print(labels.shape)
            main_loss = - self.crf(final_bert_feats, labels, mask=input_mask.byte(), reduction='mean')
            alpha = 0.88
            loss =  alpha * main_loss + (1 - alpha) * cl_loss

            return loss
        else:
            pred_tags = self.crf.decode(final_bert_feats, mask=input_mask.byte())
            return pred_tags


if __name__ == "__main__":
    model = MTCCMRobertaForMMTokenClassificationCRF.from_pretrained('vinai/phobert-base-v2',cache_dir='cache')
    print(model)