from N2S.model import *
from service.dbSerivce import DBService
import re
class ModelService():
    def __init__(self,model_config,db_config):
        self.dbService = DBService(db_config)
        self.type_dict = {
                'varchar':'text',
                'float':'real'
        }
        self.model = NL2SQL(model_config)

    def get_sql(self,question,table_name):

      
        question = re.sub('<','小於',question)
        question = re.sub('>','大於',question)
        question = re.sub('=','等於',question)
        table = self.dbService.get_table(table_name)
        headers = self.dbService.get_headers(table_name)
        headers[1] = [ self.type_dict[i] for i in headers[1]] 
        data = {
            'question': question,
            'headers': headers,
            'table':   table,
            'table_name': table_name
        }
        result = self.model.go(data)

        return result