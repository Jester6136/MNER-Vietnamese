from transformers import RobertaPreTrainedModel,RobertaModel
from torch import nn
import torch
from torchcrf import CRF
import torch.nn.functional as F
log_soft = F.log_softmax

class Roberta_CRF(RobertaPreTrainedModel):
    def __init__(self, config):
        super(Roberta_CRF, self).__init__(config, layer_num1=1, layer_num2=1, layer_num3=1,  num_labels_=2, auxnum_labels=2)
        self.num_labels = num_labels_
        self.robert = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
        self.crf = CRF(self.num_labels, batch_first=True)    
    
    def forward(self, input_ids, attn_masks, labels=None):  # dont confuse this with _forward_alg above.
        outputs = self.robert(input_ids, attn_masks)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emission = self.classifier(sequence_output)        
        attn_masks = attn_masks.type(torch.uint8)
        if labels is not None:
            loss = -self.crf(log_soft(emission, 2), labels, mask=attn_masks, reduction='mean')
            return loss
        else:
            prediction = self.crf.decode(emission, mask=attn_masks)
            return prediction
