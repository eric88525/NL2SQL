import flask
from flask import jsonify,request,render_template
import json
from service.dbSerivce import DBService
from service.modelService import ModelService
from config import model_config,dbConfig

app = flask.Flask(__name__ )
app.config["DEBUG"] = True
app.config["JSON_AS_ASCII"] = False


dbService = DBService(dbConfig)   
modelService = ModelService(model_config,dbConfig)

@app.route("/")
def index():
    
    return "123"


@app.route("/api/tablelist",methods=["GET"])
def get_table_list():
    result = dbService.get_table_list()
    return jsonify(result)

@app.route("/api/table",methods=["POST"])
def get_talbes():
    request_data = request.get_json()
    result = dbService.get_table(request_data["table_name"])
    return jsonify(result)

@app.route("/api/headers",methods=["POST"])
def get_headers():
    request_data = request.get_json()
    result = dbService.get_headers(request_data["table_name"])
    return jsonify(result)

@app.route("/api/sql",methods=["POST"])
def get_sql():
    request_data = request.get_json()
    result = modelService.get_sql(request_data["question"],request_data["table_name"])
    return jsonify(result)

if __name__ == '__main__':
    app.run()