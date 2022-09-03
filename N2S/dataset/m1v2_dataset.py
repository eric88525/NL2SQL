from base64 import encode
from email import header
import json
import random
import numpy as np
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset


class M1Dataset(Dataset):
    """model 1 dataset
    Attributes:
        table_map: A map mapping table id to it's columns info.
        datas: A list contains all row datas.
            Data is map type, for example:
            {
                "question": "长沙2011年平均每天成交量是3.17,那么近一周的成交量是多少",
                "table_id": "69cc8c0c334311e98692542696d6e445",
                "sql": {"agg": [0], "cond_conn_op": 1,
                "sel": [5],
                "conds": [[1, 2, "3.17"], [0, 2, "长沙"]]}}
                "header":  [['name', 'age', 'gender'], ['varchar', 'float', 'varchar']]
            }
    """

    def __init__(self, table_path, data_path, tokenizer_name_or_path):
        """Init M1Dataset with table file path and data file path"""
        self.table_map = self.get_table(table_path)
        self.datas = self.get_datas(self.table_map, data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, additional_special_tokens=['[unused11]', '[unused12]'])

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
                "question": "长沙2011年平均每天成交量是3.17,那么近一周的成交量是多少",
                "table_id": "69cc8c0c334311e98692542696d6e445",
                "sql": {"agg": [0], "cond_conn_op": 1,
                "sel": [5],
                "conds": [[1, 2, "3.17"], [0, 2, "长沙"]]}}
                "header":  [['name', 'age', 'gender'], ['varchar', 'float', 'varchar']]
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

    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return len(self.datas)

    @staticmethod
    def collate_fn(batch, tokenizer):
        """
        Arguments:
            batch: A dictionary list, length is batch size. For example:
            [{
                "question": "长沙2011年平均每天成交量是3.17,那么近一周的成交量是多少",
                "table_id": "69cc8c0c334311e98692542696d6e445",
                "sql": { "agg": [0],
                         "cond_conn_op": 1,
                         "sel": [5],
                         "conds": [[1, 2, "3.17"], [0, 2, "长沙"]] # [column index, column operator, value]
                        },
                "header":  [['name', 'age', 'gender'], ['varchar', 'float', 'varchar']]
            }, ...]
        Returns:
            A dict has multiple keys:
                'input_ids': tokenized text ids
                'token_type_ids': tokenized text type ids
                'attention_mask': attention mask for skip attention padding tokens
                'agg': the function apply to column
                'cond_conn_op':the condition connect operator
                'conds_ops': the column operator
                'header_idx':  the header index in input_ids
        """
        batch_tokens = []
        special_token_map = {'text': '[unused11]', 'real': '[unused12]'}
        special_token_id = [tokenizer.convert_tokens_to_ids('[unused11]'),
                            tokenizer.convert_tokens_to_ids('[unused12]')]

        for data in batch:
            header_name = data['header'][0]
            header_type = data['header'][1]

            question_tokens = tokenizer.tokenize(data['question'])

            for col_name, col_type in zip(header_name, header_type):
                col_tokens = ['[SEP]', special_token_map[col_type]] + \
                    tokenizer.tokenize(col_name)
                question_tokens.extend(col_tokens)

            batch_tokens.append(question_tokens)

        # A dict has 3 keys:
        # input_ids: tensor, shape = (batch_size, sequence_len)
        # token_type_ids: tensor, shape = (batch_size, sequence_len)
        # attention_mask: tensor, shape = (batch_size, sequence_len)
        batch_encode = tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_tokens,
                                                   padding=True,
                                                   return_tensors='pt',
                                                   is_split_into_words=True,
                                                   max_length=200,
                                                   truncation=True
                                                   )
        valid_header = torch.zeros_like(batch_encode['input_ids'])

        for special_token_id_ in special_token_id:
            valid_header[batch_encode['input_ids'] == special_token_id_] = 1

        # agg and conds_ops
        # shape = (batch, columns token in batch)
        batch_agg = []
        batch_conds_ops = []

        for batch_idx, data in enumerate(batch):
            sql = data["sql"]
            agg = torch.zeros(valid_header[batch_idx].sum().item(), dtype=torch.long).fill_(6)
            conds_ops = torch.full_like(agg, 4, dtype=torch.long)

            for sel_, agg_ in zip(sql['sel'], sql['agg']):
                if sel_ < len(agg):
                    agg[sel_] = agg_

            for col_idx, col_op, _ in sql["conds"]:
                if col_idx < len(conds_ops):
                    conds_ops[col_idx] = col_op

            batch_agg.append(agg)
            batch_conds_ops.append(conds_ops)

        # cond_conn_op shape=(batch size, 1)
        batch_cond_conn_op = torch.LongTensor(
            [x["sql"]["cond_conn_op"] for x in batch])
        batch_encode['cond_conn_op'] = batch_cond_conn_op


        batch_encode['agg'] = torch.hstack(batch_agg)
        batch_encode['conds_ops'] = torch.hstack(batch_conds_ops)
        batch_encode['header_idx'] = valid_header

        return batch_encode
