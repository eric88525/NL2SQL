
from select import select
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from difflib import SequenceMatcher


class N2Sm1(nn.Module):
    """The model that can classify columns's attribute and SQL command operator.

    Attributes:
        cond_conn_op_decoder: the classification layer of condition connect operator.
            Ouput size is (batch_size, 3), represent  ['', 'AND', 'OR']

        agg_deocder: the classification layer of the function apply on column.
            For example: Max 'score'
            Ouput size is (batch_size, column_counts, 7), represent ['', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM']

        cond_op_decoder: the classification layer of the column operator.
            For example: 'name' = 'eric'
            Output size is (batch_size, column_count, 5), represent ['>', '<', '=', '!=', '']
    """
    def __init__(self, pretrained_model_name):
        super(N2Sm1, self).__init__()

        config = AutoConfig.from_pretrained(pretrained_model_name)
        self.bert_model = AutoModel.from_pretrained(
            pretrained_model_name, config=config)

        self.cond_conn_op_decoder = nn.Linear(config.hidden_size, 3)
        self.agg_deocder = nn.Linear(config.hidden_size, 7)
        self.cond_op_decoder = nn.Linear(config.hidden_size, 5)

    def get_agg_hiddens(self, hiddens, header_ids):
        # header_ids [bsize,headers_idx]
        # hiddens [bsize,seqlength,worddim]
        arr = []
        for b_idx in range(0, hiddens.shape[0]):
            s = torch.stack([hiddens[b_idx][i]
                            for i in header_ids[b_idx]], dim=0)
            arr.append(s)
        return torch.stack(arr, dim=0)

    def forward(self, input_ids, attention_mask, token_type_ids, header_ids, **kwargs):
        # hidden [bsize,seqlength,worddim] cls [bsize,worddim]
        hiddens, cls = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)

        cond_conn_op = self.cond_conn_op_decoder(cls)

        header_hiddens = self.get_agg_hiddens(hiddens, header_ids)

        agg = self.agg_deocder(header_hiddens)
        cond_op = self.cond_op_decoder(header_hiddens)

        return cond_conn_op, cond_op, agg

class N2Sm2(nn.Module):
    def __init__(self, pretrained_model_name):
        super(N2Sm2, self).__init__()

        config = AutoConfig.from_pretrained(pretrained_model_name)

        self.bert_model = AutoModel.from_pretrained(
            pretrained_model_name, config=config)

        self.decoder = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        # hidden [bsize,seqlength,worddim] cls [bsize,worddim]
        hiddens, cls = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        return self.decoder(cls)


class NL2SQL():
    """The class that combine model 1 and model 2 to generate SQL commmand"""
    def __init__(self, config):

        self.device = config['device']

        self.m1_tokenizer = AutoTokenizer.from_pretrained(
            config['m1_pretrained_model_name'])
        self.m2_tokenizer = AutoTokenizer.from_pretrained(
            config['m2_pretrained_model_name'])

        self.model_1 = N2Sm1(config['m1_pretrained_model_name'])
        self.model_1.load_state_dict(torch.load(
            config['m1_path'], map_location=torch.device('cpu')))

        self.model_2 = N2Sm2(config['m2_pretrained_model_name'])
        self.model_2.load_state_dict(torch.load(
            config['m2_path'], map_location=torch.device('cpu')))

        self.analyze = config['analyze']

    def get_m1_output(self, question, headers):
        """Get model1 output

        Returns:
            A dict mapping to model 1 output. For example:
            {
                'agg': array([6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
                'cond': array([4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4]),
                'conn_op': array(1)
            }
        """
        all_tokens = self.m1_tokenizer.tokenize(question)
        col_type_token_dict = {'text': '[unused11]', 'real': '[unused12]'}

        for h, t in zip(headers[0], headers[1]):
            tokens = ['[SEP]', col_type_token_dict[t]] + \
                self.m1_tokenizer.tokenize(h)
            all_tokens = all_tokens + tokens

        # get the header token place
        header_ids = []

        for i in range(len(all_tokens)):
            if all_tokens[i] == col_type_token_dict['text'] or all_tokens[i] == col_type_token_dict['real']:
                header_ids.append(i+1)

        plus = self.m1_tokenizer.encode_plus(
            all_tokens, is_split_into_words=True)

        input_ids = torch.tensor(plus['input_ids']).unsqueeze(0).to(self.device)
        token_type_ids = torch.tensor(plus['token_type_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(plus['attention_mask']).unsqueeze(0).to(self.device)
        header_ids = torch.tensor(header_ids).unsqueeze(0).to(self.device)

        cond_conn_op_pred, conds_ops_pred, agg_pred = self.model_1(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, header_ids=header_ids)

        result = {}
        result['conn_op'] = torch.argmax(
            cond_conn_op_pred.squeeze(), dim=-1).to('cpu').numpy()
        result['agg'] = torch.argmax(
            agg_pred.squeeze(), dim=-1).to('cpu').numpy()
        result['cond'] = torch.argmax(
            conds_ops_pred.squeeze(), dim=-1).to('cpu').numpy()

        return result

    def get_m2_output(self, question, cond):
        ids = self.m2_tokenizer.encode_plus(question, cond)
        for k in ids.keys():
            ids[k] = torch.tensor(ids[k]).to(self.device)
        input_ids, token_type_ids, attention_mask = ids['input_ids'].unsqueeze(0).to(
            self.device), ids['token_type_ids'].unsqueeze(0).to(self.device), ids['attention_mask'].unsqueeze(0).to(self.device)
        pred = self.model_2(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids).squeeze().item()
        return pred

    def get_sql(self, data, m1, table, table_name):

        conn_map = ['', 'AND', 'OR']
        agg_map = ['', 'AVG', 'MAX', 'MIN', 'COUNT', 'SUM']
        cond_map = ['>', '<', '=', '!=', '']
        agg, cond, conn_op = m1['agg'], m1['cond'], conn_map[m1['conn_op']]
        headers = data['headers']

        pre = ''
        column = ''

        print('agg result')
        print(f"\tagg: {agg}")
        print(f"\tcond: {cond}")
        print(f"\tconn_op: {conn_op}")

        not_select = True
        for i in agg:
            if i != 6:
                not_select = False
                break

        if not_select:
            str_sim = np.zeros(len(agg))
            for i, h in enumerate(headers[0]):
                str_sim[i] = SequenceMatcher(None, data["question"], h).ratio()
            agg[np.argmax(str_sim)] = 0
            print(f"agg not select! after fix  = {agg} by {str_sim.tolist()}")

        # SELECT
        for col, val in enumerate(agg):
            if val != 6:
                if data['headers'][1][col] == 'text':
                    column += pre + f"(`{headers[0][col]}`)"
                else:
                    column += pre + f"{agg_map[val]}(`{headers[0][col]}`)"
                pre = ' ,'

        print(f"SELECT = {column}\n\n")

        # where condition
        condition_str = ''

        for col, val in enumerate(cond):
            if val != 4:
                if data['headers'][1][col] == 'text':
                    values_list = set([r[col]
                                      for r in table])  # value from table
                elif data['headers'][1][col] == 'real':
                    values_list = extract_values_from_text(
                        data['question'])  # value from question
                    if self.analyze:
                        print(values_list, 'number from question')

                possible_cond = []

                for v in values_list:
                    # format like apple > 10
                    if data['headers'][1][col] == 'text':
                        cond = f"`{headers[0][col]}` {cond_map[val]} \"{str(v)}\""
                    else:
                        cond = f"`{headers[0][col]}` {cond_map[val]} {str(v)}"
                    p = self.get_m2_output(data['question'], cond)
                    possible_cond.append([cond, p])

                if len(possible_cond) == 0:
                    continue

                print('possible_cond')
                print(possible_cond, "\n\n")

                # add condition text
                possible_cond = sorted(
                    possible_cond, key=lambda x: x[1], reverse=True)

                # if all cond has low p, pick first
                if possible_cond[0][-1] < 0.4:
                    possible_cond[0][-1] = 1

                if len(possible_cond) > 1 and conn_op == '':
                    conn_op = 'AND'

                for _cond, p in possible_cond:
                    if p < 0.4:
                        continue
                    if len(condition_str):
                        condition_str += f"{conn_op} {_cond}"
                    else:
                        condition_str += f"{_cond}"

        print(f"WHERE = {condition_str}")
        result = f'SELECT {column} FROM `{table_name}`'

        if condition_str != '':
            result += f" WHERE {condition_str}"

        print(result)
        return result

    def go(self, data):
        """Convert input query to SQL command
        Args:
            A dict have table_id, question, headers and table.
            For example:
            {
                table_id: '4d258a053aaa11e994c3f40f24344a08',
                question: '搜房网和人人网的周涨跌幅是多少',
                headers: [['股票名稱', '周漲跌幅'], ['text', 'real']],
                table: [['搜房网', 10], ['人人网', 50], ['長榮', 10], ...]
            }
        """
        m1 = self.get_m1_output(data["question"], data["headers"])
        result = self.get_sql(data, m1, data["table"], data["table_name"])
        return result
