import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class M1Model(nn.Module):
    """The model that can classify columns and SQL command operator.

    Attributes:
        cond_conn_op_decoder: the classification layer of condition connect operator.
            Ouput size is (batch_size, 3), represent  ['', 'AND', 'OR']

        agg_deocder: the classification layer of the function apply on column.
            Ouput size is (batch_size, column_counts, 7)
            SQL syntax: agg (column_name)
            agg is in ['', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM']

        cond_op_decoder: the classification layer of the column operator.
            Output size is (batch_size, column_count, 5)
            SQL syntax: column_name operator value
            For example: weight > 50
            operator is in ['>', '<', '=', '!=', '']
    """
    def __init__(self, pretrained_model_name):
        super(M1Model, self).__init__()

        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.bert_model = AutoModel.from_pretrained(
            pretrained_model_name, config=config)

        self.cond_conn_op_decoder = nn.Linear(config.hidden_size, 3)
        self.agg_deocder = nn.Linear(config.hidden_size, 7)
        self.cond_op_decoder = nn.Linear(config.hidden_size, 5)

    def get_agg_hiddens(self, hiddens, header_idx):
        """Get hidden states of columns
        Args:
            hiddens: shape = (batch_size, seq_length, word_dimention)
            header_idx: shape = (batch_size, headers_idx)
        """
        arr = []
        batch_size = hiddens.shape[0]
        for b_idx in range(batch_size):
            s = hiddens[b_idx][header_idx[b_idx]]
            arr.append(s)
        return torch.stack(arr, dim=0)

    def forward(self, input_ids, attention_mask, token_type_ids, header_idx, **kwargs):
        
        hiddens, cls = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)

        cond_conn_op = self.cond_conn_op_decoder(cls)
    
        # shape = (batch_size, columns_count, hidden_dim)
        header_hiddens = self.get_agg_hiddens(hiddens, header_idx) 
        # shape = (batch_size, columns_count, 7)
        agg = self.agg_deocder(header_hiddens)
        # shape = (batch_size, columns_count, 5)
        cond_op = self.cond_op_decoder(header_hiddens)

        return cond_conn_op, cond_op, agg