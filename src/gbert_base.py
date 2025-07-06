from transformers import AutoTokenizer, AutoModel
from transformers import PreTrainedModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn


class GbertTagModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = AutoModel.from_pretrained("deepset/gbert-large", config=config)

        self.use_lstm = False
        self.use_mean_pool = False
        self.use_multisample_drpout = True

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)


        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size = config.hidden_size,
                hidden_size = config.hidden_size // 2,
                num_layers= 2,
                batch_first=True,
                bidirectional=True,
                dropout = config.hidden_dropout_prob
            )
            self._init_lstm(self.lstm)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.post_init()
    
    def _init_lstm(self, lstm_layer):
        for name, param in lstm_layer.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask = attention_mask,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state

        if self.use_lstm:
            self.lstm.flatten_parameters()
            sequence_output, _ = self.lstm(sequence_output)
        
        if self.use_mean_pool:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            mean_embeddings = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
            sequence_output = mean_embeddings
        
        else:
            if self.use_multisample_drpout:
                output = self.classifier(self.dropout1(sequence_output))
                output += self.classifier(self.dropout2(sequence_output))
                output += self.classifier(self.dropout3(sequence_output))
                output += self.classifier(self.dropout4(sequence_output))
                output += self.classifier(self.dropout5(sequence_output))
                logits = output / 5.0
            else:
                sequence_output = self.dropout(sequence_output)
                logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
        
