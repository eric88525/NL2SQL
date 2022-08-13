
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from collections import namedtuple
from .utils import *

pair = namedtuple('pair', ['question', 'col_op_val', 'label'])


class M2Dataset(Dataset):
    def __init__(self, table_path, data_path, tokenizer_name_or_path):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        self.table_map = self.get_table(table_path)
        row_datas = self.read_datas(self.table_map, data_path)

        self.datas = self.make_pairs(row_datas, self.table_map)

        # every data's label
        self.labels = [i[-1] for i in self.datas]

    def get_table(self, path):
        """Get table's columns info

        Returns:
            A dict mapping table_id to columns_info.
            The columns_info is a namedtuple containing two keys header and types
            The first key is columns name, second is column's data type.
            For example:
                columns_info (
                    header = ['name', 'age', 'gender'],
                    types = ['varchar', 'float', 'varchar']
                )
        """
        id_header = {}

        with open(path, 'r') as json_file:
            json_list = list(json_file)

        table = namedtuple('table', ['header', 'types', 'rows'])

        for row in json_list:
            result = json.loads(row)
            id_header[result['id']] = table(
                header=result['header'], types=result['types'], rows=result['rows'])

        return id_header

    def read_datas(self, id_header, path):
        """Read dataset file and return row datas
        Returns:
            A list contains all datas.
            The data is dict type, for example:
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

    def make_pairs(self, datas, table_map):
        """ Split one positive label to many negative pairs

        If column is text type, for example: 'name = tom'
        create more sample like:
            ['name = eric', 'name = tim',...]

        If column is real type, extract number from query and make pairs
        For example: query = '长沙2011年平均每天成交量是3.17，那么近一周的成交量是多少'
        create more sample like:
            ['成交量 = 2011', '成交量 = 3.17', ....]
        """
        all_pairs = []
        operator_list = ['>', '<', '=', '!=']

        for data in datas:
            # the condition like col=val that has seen
            seen_pair = set()

            # namedtuple('columns_info', ['header', 'types', 'rows'])
            table = table_map[data['table_id']]
            conds = data['sql']['conds']

            # Add positive datas
            for col, op, val in conds:
                # column + operator + value, For example:
                # 'name=Tim' or 'weight>20'
                col_op_val = f"{table.header[col]}{operator_list[op]}{val}"

                if len(col_op_val) > 270:
                    continue

                seen_pair.add(col_op_val)
                all_pairs.append(
                    pair(question=data["question"], col_op_val=col_op_val, label=1))

            # Add negative datas
            # positive:negative = 1:2
            for col, op, val in conds:

                max_neg = 2
                
                if table.types[col] == "real":
                    # extract number value from question
                    candidate_list = extract_values_from_text(data['question'])
                else:
                    candidate_list = (row[col] for row in table.rows)

                for possible_val in candidate_list:
                    # column + operator + value, For example:
                    # 'name=Tim' or 'weight>20'
                    col_op_val = f"{table.header[col]}{operator_list[op]}{possible_val}"

                    if len(col_op_val) > 270 or col_op_val in seen_pair:
                        continue

                    seen_pair.add(col_op_val)
                    pair_ = pair(question=data["question"], col_op_val=col_op_val, label=0)
                    all_pairs.append(pair_)
                    max_neg -=1

                    if not max_neg:
                        break

        return all_pairs

    def __getitem__(self, idx):
        return self.datas[idx]
        
    def __len__(self):
        return len(self.datas)

    @staticmethod
    def collate_fn(batch, tokenizer):
        """ A staticmethod, be used at torch.utils.data.DataLoader collect_fn argument
        Arguments:
            batch: A batch size length list, each item is a namedtuple.
            For example:
                [ pair(question='17年6月11号那天生...', col_op_val='生产日期/批号=2017-09-23', label=1),
                  pair(question='哪些公司周涨跌幅...', col_op_val='收盘价（元）<1', label=1) ]
            tokenizer:
                Tokenizer
        Returns:
            A dict contain 4 keys.
            3 for bert model, 1 for label
        """

        fn_batch = tokenizer.batch_encode_plus( batch_text_or_text_pairs=[[x.question, x.col_op_val] for x in batch], padding=True, return_tensors ='pt')
        fn_batch['label'] = torch.FloatTensor([[x.label] for x in batch])

        return fn_batch