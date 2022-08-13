from transformers import AutoModel, AutoConfig
import torch.nn as nn

class M2Model(nn.Module):
    def __init__(self, model_type):
        super(M2Model, self).__init__()
        config = AutoConfig.from_pretrained(model_type)
        self.pretrained_model = AutoModel.from_pretrained(model_type, config=config)
        self.cls_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, token_type_ids, **kwargs):
        # hidden [bsize,seqlength,worddim] cls [bsize,worddim]

        model_output = self.pretrained_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)

        return self.cls_layer(model_output.pooler_output)