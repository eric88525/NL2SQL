import flask
from flask import jsonify
from transformers import BertTokenizer,BertModel,BertConfig
import torch

import mysql.connector
import json
from N2S.model import NL2SQL

app = flask.Flask(__name__ )
app.config["DEBUG"] = True
app.config["JSON_AS_ASCII"] = False
base_url = 'C:/Users/User/Documents/7.Flask/nl2sqlDemo/'

model_config = { 'model_type': f"{base_url}/nlpmodel/chinese_wwm_pytorch/pytorch_model.bin" , 
          'config_path': f'{base_url}/nlpmodel/chinese_wwm_pytorch/bert_config.json',
          'vocab_path': f'{base_url}/nlpmodel/chinese_wwm_pytorch/vocab.txt',
          'device':torch.device('cpu'),
          'm1_path':f'{base_url}/nlpmodel/saved_models/M1.pt',
          'm2_path': f'{base_url}/nlpmodel/saved_models/M2.pt','analyze':False}
        
bertconfig = BertConfig.from_json_file(model_config['config_path'])



@app.route("/")
def hello():
    return "123"

@app.route("/tables")
def get_talbes():
    d = {'a':123,'b':456}

    return jsonify(d)

if __name__ == '__main__':
    app.run()