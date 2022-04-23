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

modelConfig = { 'model_type': f"{base_url}/nlpmodel/chinese_wwm_pytorch/pytorch_model.bin" , 
          'config_path': f'{base_url}/nlpmodel/chinese_wwm_pytorch/bert_config.json',
          'vocab_path': f'{base_url}/nlpmodel/chinese_wwm_pytorch/vocab.txt',
          'device':torch.device('cpu'),
          'm1_path':f'{base_url}/nlpmodel/saved_models/M1.pt',
          'm2_path': f'{base_url}/nlpmodel/saved_models/M2.pt','analyze':True}