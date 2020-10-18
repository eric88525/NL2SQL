import flask
from flask import jsonify,request
from transformers import BertTokenizer,BertModel,BertConfig
import torch
import json
from N2S.model import NL2SQL
from service.dbSerivce import DBService
from service.modelService import ModelService

app = flask.Flask(__name__ )
app.config["DEBUG"] = True
app.config["JSON_AS_ASCII"] = False

base_url = 'C:/Users/User/Documents/7.Flask/nl2sqlDemo/'

dbConfig = {
        'host' : '127.0.0.1',
        'user' : "ericzone",
        'password': "ericzone",
        'database' : "nl2sql",
        'table_map':"namemap"
}

model_config = { 'model_type': f"{base_url}/nlpmodel/chinese_wwm_pytorch/pytorch_model.bin" , 
          'config_path': f'{base_url}/nlpmodel/chinese_wwm_pytorch/bert_config.json',
          'vocab_path': f'{base_url}/nlpmodel/chinese_wwm_pytorch/vocab.txt',
          'device':torch.device('cpu'),
          'm1_path':f'{base_url}/nlpmodel/saved_models/M1.pt',
          'm2_path': f'{base_url}/nlpmodel/saved_models/M2.pt','analyze':False}

dbService = DBService(dbConfig)   
modelService = ModelService(model_config,dbConfig)

@app.route("/")
def index():
    return "123"


@app.route("/tablelist")
def get_table_list():
    result = dbService.get_table_list()
    return jsonify(result)

@app.route("/tables",methods=["POST"])
def get_talbes():
    d = {'a':123,'b':456}
    return jsonify(d)

@app.route("/sql",methods=["POST"])
def get_sql():
    request_data = request.get_json()
    result = modelService.get_sql(request_data["question"],request_data["table_name"])
    return jsonify(result)

if __name__ == '__main__':
    app.run()