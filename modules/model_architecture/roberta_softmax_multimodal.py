from transformers import RobertaModel

import copy
from transformers.models.roberta.modeling_roberta import RobertaIntermediate,RobertaOutput,RobertaSelfOutput,RobertaPreTrainedModel
from transformers import BertConfig
from torch import nn as nn
from torchcrf import CRF
import torch
import math
import json
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

import torch
import torch.nn.functional as F  # For softmax

class RobertaSoftmaxMultimodal(RobertaPreTrainedModel):
    """Coupled Cross-Modal Attention BERT model for token-level classification with a softmax layer on top.
    """
    def __init__(self, config, layer_num1=1, layer_num2=1, layer_num3=1, num_labels_=2, auxnum_labels=2):
        super(RobertaSoftmaxMultimodal, self).__init__(config)
        self.num_labels = num_labels_
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vismap2text = nn.Linear(2048, config.hidden_size)
        self.txt2img_attention = RobertaCrossEncoder(config, layer_num1)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels_)

        self.init_weights()

    def forward(self, input_ids, segment_ids, input_mask, added_attention_mask, visual_embeds_mean, visual_embeds_att, labels=None):
        features = self.roberta(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        sequence_output = features["last_hidden_state"]
        sequence_output = self.dropout(sequence_output)

        vis_embed_map = visual_embeds_att.view(-1, 2048, 49).permute(0, 2, 1)
        converted_vis_embed_map = self.vismap2text(vis_embed_map)

        img_mask = added_attention_mask[:, :49]
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        cross_encoder = self.txt2img_attention(sequence_output, converted_vis_embed_map, extended_img_mask)
        cross_output_layer = cross_encoder[-1]

        final_output = torch.cat((sequence_output, cross_output_layer), dim=-1)
        logits = self.classifier(final_output)  # Now logits is a batch_size * seq_len * num_labels tensor

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if input_mask is not None:
                active_loss = input_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            # Apply softmax to logits to get pred_tags
            pred_tags = torch.nn.functional.softmax(logits,dim=2)
            return pred_tags


if __name__ == "__main__":
    model = RobertaSoftmaxMultimodal.from_pretrained('vinai/phobert-base-v2')
    print(model)