import flask
from flask import jsonify,request,render_template
import json
from service.dbSerivce import DBService
from service.modelService import ModelService
from config import modelConfig,dbConfig
from opencc import OpenCC

app = flask.Flask(__name__ )
app.config["DEBUG"] = True
app.config["JSON_AS_ASCII"] = False


testing = False

dbService = DBService(dbConfig)
modelService = ModelService(modelConfig,dbConfig)

tw2s = OpenCC('tw2s')
s2t = OpenCC('s2t')

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tablelist",methods=["GET"])
def get_table_list():
    table_list = dbService.get_table_list()
    result = [ {"Text": row[:15] ,"Value":row  }for row in table_list ]


    if testing:
        table_list = ['11111111111111111','22222222222222222222','3333333333333333333']
        result = [ {"Text": row[:3] ,"Value" : row  }  for row in table_list ]

    return jsonify(result)

@app.route("/api/table",methods=["POST"])
def get_talbe():

    request_data = request.get_json()
    #print(request_data , 'req')
    columns = dbService.get_headers(request_data["table_name"])[0]
    rows  = dbService.get_table(request_data["table_name"])
    data = []
    for r in rows:
        temp = {}
        for i,col in enumerate(columns):
            temp[col] = s2t.convert(str(r[i]))
        data.append(temp)

    columns = [ {"Index":  i,"Name":  s2t.convert(i)} for i in columns   ]
    data =  { "columns":columns,"datas":data  }
    #print(data)
    return jsonify( data )

@app.route("/api/headers",methods=["POST"])
def get_headers():
    request_data = request.get_json()
    result = dbService.get_headers(request_data["table_name"])

    if testing:
        result = [ ('姓名' , '學號' , '身高' , '體重' , '居住地') ,\
             ('varchar','varchar','float','float','varchar') ]

    return jsonify(result)


# question -> sql
@app.route("/api/sql",methods=["POST"])
def get_sql():
    try:
        request_data = request.get_json()
        print(request_data)
        question =tw2s.convert(request_data["question"] )
        result = modelService.get_sql( question   ,request_data["table_name"])
        result = s2t.convert(result)
        return jsonify(result)
    except:
        return jsonify("抱歉 此問題無法正確轉換")


# execute sql command and return table
@app.route("/api/runsql",methods=["POST"])
def run_sql():
    request_data = request.get_json()
    print(request_data)
    sql =tw2s.convert(str(request_data["sql"] )).replace('\n','')

    # 字串比對 抽sql col
    columns = sql[6:sql.find("FROM")].replace(')','').replace('(','').replace('`','').replace(' ','').split(',')
    # 產生 index & name pair 

    print(f"sql {sql}")
    # db result
    data = []
    rows = dbService.exe_sql( sql )
    print(f"rows {rows}")
    for r in rows:
        temp = {}
        for i,col in enumerate(columns):
            temp[col] = s2t.convert(str(r[i]))
        data.append(temp)

    columns = [ {"Index":  i,"Name":  s2t.convert(i)} for i in columns   ]

    print(columns)
    print(data)

    result =  { "columns":columns,"datas":data  }
    #data =  { "columns":columns,"datas":data  }
    return jsonify(result)

if __name__ == '__main__':
    app.run()

# Table_43af76a31d7111e98e78f40f24344a08
# Table_43b06b7d1d7111e989d6f40f24344a08


#  SELECT (`办公电话`) ,(`邮箱`) FROM `Table_43b06b7d1d7111e989d6f40f24344a08` WHERE `姓名` = "杨涛" 