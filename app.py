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
#modelService = ModelService(model_config,dbConfig)

@app.route("/")
def index():
    
    return render_template("index.html")


@app.route("/api/tablelist",methods=["GET"])
def get_table_list():
    table_list = dbService.get_table_list()
    result = [ {"Text": row[:15] ,"Value":row  }for row in table_list ]
    return jsonify(result)

@app.route("/api/table",methods=["POST"])
def get_talbes():
    request_data = request.get_json()
    columns = dbService.get_headers(request_data["table_name"])[0]
    rows  = dbService.get_table(request_data["table_name"])
    data = []
    for r in rows:
        temp = {}
        for i,col in enumerate(columns):
            temp[col] = r[i]
        data.append(temp)
    return jsonify( { "columns":columns,"data":data  }  )

@app.route("/api/headers",methods=["POST"])
def get_headers():
    request_data = request.get_json()
    result = dbService.get_headers(request_data["table_name"])
    return jsonify(result)

"""
@app.route("/api/sql",methods=["POST"])
def get_sql():
    request_data = request.get_json()
    result = modelService.get_sql(request_data["question"],request_data["table_name"])
    return jsonify(result)
"""
if __name__ == '__main__':
    app.run()