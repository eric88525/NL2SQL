from N2S.sql_model import *
from .dbSerivce import DBService
import re


class ModelService():
    def __init__(self, model_config, db_config):
        """The service interact with ai model"""

        self.dbService = DBService(db_config)
        self.type_dict = {
            'varchar': 'text',
            'float': 'real'
        }
        self.model = SqlModel(model_config)

    def get_sql(self, question: str, table_name: str):
        """Given question and table name, return correspond SQL command"""

        # replace some text to let model works better
        question = re.sub('<', '小于', question)
        question = re.sub('>', '大于', question)
        question = re.sub('=', '等于', question)

        print(f"The user question is {question}")

        # Fetch table from database
        table = self.dbService.get_table(table_name)
        # get headers from database
        headers = self.dbService.get_headers_info(table_name)
        # convert header datatype to token type
        headers[1] = [self.type_dict[i] for i in headers[1]]
        data = {
            'question': question,
            'headers': headers,
            'table':   table,
            'table_name': table_name
        }
        # get SQL command from model
        result = self.model.data_to_sql(data)

        return result
