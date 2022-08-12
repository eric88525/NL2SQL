import torch
import os
base_url = os.getcwd()

dbConfig = {
        'host' : '127.0.0.1',
        'user' : "ericzone",
        'password': "ericzone",
        'database' : "nl2sql",
}

modelConfig = { 
        'm1_pretrained_model_name': "hfl/chinese-roberta-wwm-ext", 
        'm2_pretrained_model_name': "voidful/albert_chinese_tiny",
        'device':torch.device('cpu'),
        'm1_path':f'{base_url}/saved_models/roberta_v1',
        'm2_path': f'{base_url}/saved_models/tiny_v2',
        'analyze':True
        }