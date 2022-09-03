import torch
import os
base_url = os.getcwd()

dbConfig = {
        'host' : 'db',
        'user' : "n2suser",
        'password': "n2spassword",
        'database' : "nl2sql",
}

modelConfig = {
        'm1_pretrained_model_name': "hfl/chinese-roberta-wwm-ext",
        'm1_tokenizer_name_or_path': "hfl/chinese-roberta-wwm-ext",
        'm2_pretrained_model_name': "voidful/albert_chinese_large",
        'm2_tokenizer_name_or_path': "voidful/albert_chinese_large",
        'device':torch.device('cpu'),
        'm1_model_path':f'{base_url}/N2S/saved_models/m1_model',
        'm2_model_path': f'{base_url}/N2S/saved_models/m2_model',
        'analyze':True
}