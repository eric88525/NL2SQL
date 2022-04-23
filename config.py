import torch
import os
base_url = os.getcwd()
dbConfig = {
        'host' : '127.0.0.1',
        'user' : "ericzone",
        'password': "ericzone",
        'database' : "nl2sql",
        'table_map':"namemap"
}

modelConfig = { 'pretrained_model_name': "voidful/albert_chinese_base" , 
          'device':torch.device('cpu'),
          'm1_path':f'{base_url}/saved_models/M1.pt',
          'm2_path': f'{base_url}/saved_models/M2.pt',
          'analyze':True}