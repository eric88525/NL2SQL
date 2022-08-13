
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from difflib import SequenceMatcher

from model.m1_model import M1Model
from model.m2_model import M2Model


class NL2SQL():
    """The class combine model 1 and model 2 to generate SQL commmand"""
    def __init__(self, config):

        self.device = config['device']

        self.m1_tokenizer = AutoTokenizer.from_pretrained(
            config['m1_pretrained_model_name'])
        self.m2_tokenizer = AutoTokenizer.from_pretrained(
            config['m2_pretrained_model_name'])

        self.model_1 = M1Model(config['m1_pretrained_model_name'])
        self.model_1.load_state_dict(torch.load(
            config['m1_path'], map_location=torch.device('cpu')))

        self.model_2 = M2Model(config['m2_pretrained_model_name'])
        self.model_2.load_state_dict(torch.load(
            config['m2_path'], map_location=torch.device('cpu')))
        
        self.special_token_map = {'text': '[unused11]', 'real': '[unused12]'}
        self.analyze = config['analyze']

    def get_m1_output(self, question, headers):
        """Get model1 output
        Returns:
            AggModel output in dict data type.
            The length of 'agg' and 'cond' is equal to column counts
            For example:
            {
                'agg': array([6, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]),
                'cond': array([4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4]),
                'conn_op': array(1)
            }
        """
        all_tokens = self.m1_tokenizer.tokenize(question)

        # Append [[SEP], [unused11], text_type_column, [SEP], [unused12], real_type_column] to all_tokens
        # The result is:
        # all_tokens + [[SEP], [unused11], 'name', [SEP], [unused12], 'ege'...]
        for col_name, col_type in zip(headers[0], headers[1]):
            tokens = ['[SEP]', self.special_token_map[col_type]] + \
                self.tokenizer.tokenize(col_name)
            all_tokens.extend(tokens)

        # Get header token index
        header_idx = []
        for i, token in enumerate(all_tokens):
            if token == self.special_token_map['text'] or token == self.special_token_map['real']:
                header_idx.append(i+1) # +1 due to we'll add [SEP] token in first index
    
        plus = self.m1_tokenizer.encode_plus(
            all_tokens, is_split_into_words=True, return_tensors='pt')

        for k in plus.keys():
            plus[k] = plus[k].to(self.device)
        plus['header_idx'] = torch.tensor(header_idx).unsqueeze(0).to(self.device)

        cond_conn_op_pred, conds_ops_pred, agg_pred = self.model_1(**plus)

        result = {}
        result['conn_op'] = torch.argmax(
            cond_conn_op_pred.squeeze(), dim=-1).to('cpu').numpy()
        result['agg'] = torch.argmax(
            agg_pred.squeeze(), dim=-1).to('cpu').numpy()
        result['cond'] = torch.argmax(
            conds_ops_pred.squeeze(), dim=-1).to('cpu').numpy()

        return result

    def get_m2_output(self, question, cond):

        plus = self.m2_tokenizer.encode_plus(question, cond, return_tensor='pt')
        for k in plus.keys():
            plus[k] = plus[k].to(self.device)

        pred = self.model_2(**plus).squeeze().item()
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
