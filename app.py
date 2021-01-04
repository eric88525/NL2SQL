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
    print(request_data , 'req')
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
    print(data)
    return jsonify( data )

    '''
    data = {
    "columns": [
        {
            "Index": "城市",
            "Name": "城市"
        },
        {
            "Index": "本期",
            "Name": "本期"
        },
        {
            "Index": "上期",
            "Name": "上期"
        },
        {
            "Index": "去年同期",
            "Name": "去年同期"
        },
        {
            "Index": "11年H2",
            "Name": "11年H2"
        },
        {
            "Index": "11年H1",
            "Name": "11年H1"
        },
        {
            "Index": "10年H2",
            "Name": "10年H2"
        },
        {
            "Index": "本期比上期",
            "Name": "本期比上期"
        },
        {
            "Index": "本期比去年同期",
            "Name": "本期比去年同期"
        }
    ],
    "datas": [
        {
            "10年H2": 20.8,
            "11年H1": 16.1,
            "11年H2": 14.2,
            "上期": 0,
            "去年同期": 0,
            "城市": "北京",
            "本期": 0,
            "本期比上期": "—",
            "本期比去年同期": "—"
        },
        {
            "10年H2": 33.6,
            "11年H1": 26.6,
            "11年H2": 5.3,
            "上期": 9.6,
            "去年同期": 51,
            "城市": "上海",
            "本期": 0,
            "本期比上期": "—",
            "本期比去年同期": "—"
        },
        {
            "10年H2": 10.5,
            "11年H1": 8.2,
            "11年H2": 5.7,
            "上期": 402,
            "去年同期": 0,
            "城市": "广州",
            "本期": 0,
            "本期比上期": "—",
            "本期比去年同期": "None"
        },
        {
            "10年H2": 30.7,
            "11年H1": 0,
            "11年H2": 4.8,
            "上期": 0,
            "去年同期": 0,
            "城市": "深圳",
            "本期": 0,
            "本期比上期": "None",
            "本期比去年同期": "None"
        },
        {
            "10年H2": 28.7,
            "11年H1": 37.5,
            "11年H2": 5.6,
            "上期": 0.1,
            "去年同期": 54.8,
            "城市": "杭州",
            "本期": 1.5,
            "本期比上期": "+",
            "本期比去年同期": "—"
        },
        {
            "10年H2": 24.9,
            "11年H1": 17.7,
            "11年H2": 7.1,
            "上期": 82.3,
            "去年同期": 21.2,
            "城市": "一线平均",
            "本期": 0.3,
            "本期比上期": "—",
            "本期比去年同期": "—"
        },
        {
            "10年H2": 12.5,
            "11年H1": 12.7,
            "11年H2": 0.5,
            "上期": 0,
            "去年同期": 13.6,
            "城市": "天津",
            "本期": 3.1,
            "本期比上期": "+",
            "本期比去年同期": "—"
        },
        {
            "10年H2": 26.1,
            "11年H1": 8.5,
            "11年H2": 12.9,
            "上期": 0.6,
            "去年同期": 12.4,
            "城市": "重庆",
            "本期": 2.7,
            "本期比上期": "+",
            "本期比去年同期": "—"
        },
        {
            "10年H2": 37,
            "11年H1": 10.5,
            "11年H2": 23,
            "上期": 15.9,
            "去年同期": 31.7,
            "城市": "成都",
            "本期": 15.5,
            "本期比上期": "—",
            "本期比去年同期": "—"
        },
        {
            "10年H2": 37.3,
            "11年H1": 11.2,
            "11年H2": 2.1,
            "上期": 0,
            "去年同期": 9.8,
            "城市": "南京",
            "本期": 0,
            "本期比上期": "+",
            "本期比去年同期": "—"
        },
        {
            "10年H2": 17.3,
            "11年H1": 12.8,
            "11年H2": 2.5,
            "上期": 2.9,
            "去年同期": 0.2,
            "城市": "沈阳",
            "本期": 3,
            "本期比上期": "+",
            "本期比去年同期": "+"
        },
        {
            "10年H2": 2.2,
            "11年H1": 0.1,
            "11年H2": 0.7,
            "上期": 0,
            "去年同期": 0,
            "城市": "大连",
            "本期": 0,
            "本期比上期": "NAN",
            "本期比去年同期": "NAN"
        },
        {
            "10年H2": 31.6,
            "11年H1": 7.6,
            "11年H2": 6.9,
            "上期": 0,
            "去年同期": 26.3,
            "城市": "青岛",
            "本期": 2.4,
            "本期比上期": "+",
            "本期比去年同期": "—"
        },
        {
            "10年H2": 9,
            "11年H1": 13.8,
            "11年H2": 3.4,
            "上期": 0.2,
            "去年同期": 0.1,
            "城市": "苏州",
            "本期": 0.9,
            "本期比上期": "+",
            "本期比去年同期": "+"
        },
        {
            "10年H2": 30.8,
            "11年H1": 24.2,
            "11年H2": 0.9,
            "上期": 10.4,
            "去年同期": 0,
            "城市": "宁波",
            "本期": 2.8,
            "本期比上期": "—",
            "本期比去年同期": "+"
        },
        {
            "10年H2": 11.7,
            "11年H1": 2.8,
            "11年H2": 7.3,
            "上期": 0,
            "去年同期": 0,
            "城市": "厦门",
            "本期": 0,
            "本期比上期": "NAN",
            "本期比去年同期": "NAN"
        },
        {
            "10年H2": 20.4,
            "11年H1": 10.3,
            "11年H2": 5.5,
            "上期": 2.7,
            "去年同期": 9.4,
            "城市": "二线平均",
            "本期": 3,
            "本期比上期": "+",
            "本期比去年同期": "—"
        },
        {
            "10年H2": 21.8,
            "11年H1": 12.6,
            "11年H2": 6,
            "上期": 27.6,
            "去年同期": 13.3,
            "城市": "整体平均",
            "本期": 2.1,
            "本期比上期": "—",
            "本期比去年同期": "—"
        }
    ]
    }
    
    return jsonify(data)

    '''


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