import json
import random
import numpy as np
from transformers import AutoTokenizer
import torch

class M1Dataset():
    """model 1 dataset
    
    Attributes:
        table_map: A map mapping table id to it's columns info.
        datas: 
    """
    def __init__(self, table_path, data_path):

        self.table_map = self.get_table(table_path)
        self.datas = self.get_datas(self.table_map, data_path)

    def get_table(self, path):
        """Get table's columns info
        Returns:
            A dict mapping table_id to columns info.
            The columns info is a list contains two item.
            The first item is columns name, second is columns data type.
            For example:
            [
                ['name', 'age', 'gender'],
                ['varchar', 'float', 'varchar']
            ]
        """
        id_header = {}
        
        with open(path, 'r') as json_file:
            json_list = list(json_file)

        for json_str in json_list:
            result = json.loads(json_str)
            id_header[result['id']] = [result['header'], result['types']]

        return id_header

    def get_datas(self, id_header, path):
        """Read dataset file and return datas
        Returns:
            A list contains all row datas.
            Data is map type, for example:
            {
                "question": "长沙2011年平均每天成交量是3.17，那么近一周的成交量是多少",
                "table_id": "69cc8c0c334311e98692542696d6e445",
                "sql": {"agg": [0], "cond_conn_op": 1, 
                "sel": [5], 
                "conds": [[1, 2, "3.17"], [0, 2, "长沙"]]}}
            }
        """
        all_datas = []
        with open(path, 'r') as json_file:
            json_list = list(json_file)

        for row in json_list:
            row = json.loads(row)
            data = dict(row)
            data["header"] = id_header[row['table_id']]
            all_datas.append(data)

        return all_datas
    
class BatchSampler():
    """The sampler can create batch from datas and move to device"""
    def __init__(self, data, model_type, device):

        self.data = sorted(data, key=lambda x: len(x['header'][0]))
        # minimum column counts
        self.min_header_count = len(self.data[0]['header'][0])  
        # maximun column counts
        self.max_header_count = len(self.data[-1]['header'][0])  
        # group datas according to column counts
        self.data_groups = self.group_data(self.data)  
        # [unused1] and [unused2] represent text and real special token
        self.tokenizer = AutoTokenizer.from_pretrained(model_type, additional_special_tokens = ['[unused11]', '[unused12]'])
        # the token to represent text type or real type
        self.special_token_map = {'text': '[unused11]', 'real': '[unused12]'}

        self.device = device
    
    def group_data(self, datas):
        # A list has n groups
        # n = self.max_header_count - self.min_header_count + 1
        grouped_result = [] 
        
        prev_header_len = self.min_header_count
        prev_idx = 0
        
        for i, data_ in enumerate(self.data):
            header_len = len(data_['header'][0])
            if header_len != prev_header_len:
                grouped_result.append(datas[prev_idx:i])
                prev_header_len = header_len
                prev_idx = i

        grouped_result.append(datas[prev_idx:])
        group_item_counts = np.array([len(x) for x in grouped_result])

        # the probabily of the group be selected 
        self.group_prob = group_item_counts / np.sum(group_item_counts)  

        return grouped_result

    def encode(self, data):
        """Encode one data
        Returns:
            A dict has multiple keys:
                'input_ids': tokenized text ids
                'token_type_ids': tokenized text type ids
                'attention_mask': attention mask for skip attention padding tokens
                'agg': the function apply to column
                'cond_conn_op': the column operator 
                'conds_ops': the condition connect operator
                'header_idx': 
        """
        question, headers, sql = data['question'], data['header'], data['sql']

        # tokenize question to tokens (without special token)
        # for example: 
        # 'i have a pen' => ['i', 'have', 'a', 'pen']
        all_tokens = self.tokenizer.tokenize(question)

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
    
        column_counts = len(headers[0])

        # Create label like
        # [6, 6, 6, 5 ,6] 
        # 6 = NO_OP reoresenting not select this column
        # 0~5 representing the column should be selected and apply what kind of funciton 
        # ['', AVG, MAX, MIN, COUNT, SUM, NO_OP]
        agg = [6 for _ in range(column_counts)] 
        for sel_, agg_ in zip(sql['sel'], sql['agg']):
            agg[sel_] = agg_

        # encode tokens
        plus = self.tokenizer.encode_plus(all_tokens, is_split_into_words=True, max_length=280, padding='max_length', truncation=True)

        conds_ops = [4 for _ in range(column_counts)]

        for i in range(len(sql['conds'])):
            conds = sql['conds'][i]  # col type value
            conds_ops[conds[0]] = conds[1]

        for k, v in plus.items():
            plus[k] = torch.tensor(v)

        plus['agg'] = torch.tensor(agg)
        plus['cond_conn_op'] = torch.tensor([sql['cond_conn_op']])
        plus['conds_ops'] = torch.tensor(conds_ops)
        plus['header_idx'] = torch.tensor(header_idx)

        return plus

    def select_random_group(self):
        """Return a group index
        
        The more data a group have, the higher probability it will be selected

        Returns
            int index in range(len(self.group_prob))
        """
        return np.random.choice(len(self.group_prob), p=self.group_prob)

    def list_to_batch(self, data):
        result = {}
        for k in data[0].keys():
            result[k] = torch.stack([i[k] for i in data]).to(self.device)

        result['cond_conn_op'] = result['cond_conn_op'].squeeze()

        return result

    def get_batch(self, batch_size, encode=True):
        """Get a batch"""
        # select one group datas
        one_group_data = self.data_groups[self.select_random_group()]
        # select k data from group
        k_datas = random.choices(one_group_data, k=batch_size)
        
        if encode:
            return self.list_to_batch([self.encode(x) for x in k_datas])
        else:
            return k_datas