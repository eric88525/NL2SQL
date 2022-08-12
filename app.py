from flask import jsonify, request, render_template
from flask import Flask
from service.dbSerivce import DBService
from service.modelService import ModelService
from config import modelConfig, dbConfig
from opencc import OpenCC

app = Flask(__name__)
app.config["DEBUG"] = True
app.config["JSON_AS_ASCII"] = False

testing = False

dbService = DBService(dbConfig)
modelService = ModelService(modelConfig, dbConfig)

tw2s = OpenCC('tw2s')
s2t = OpenCC('s2t')


@app.route("/")
def index():
    """The main page"""
    return render_template("index.html")

@app.route("/api/tablelist", methods=["GET"])
def get_table_list():
    """Get the table name in the database"""
    table_list = dbService.get_table_list()
    result = [{"Text": row[:15], "Value":row} for row in table_list]

    return jsonify(result)

@app.route("/api/table", methods=["POST"])
def get_talbe():
    """Fetch target table from database"""
    request_data = request.get_json()
    columns = dbService.get_headers_info(request_data["table_name"])[0]
    rows = dbService.get_table(request_data["table_name"])
    datas = []

    for r in rows:
        temp = {}
        for i, col in enumerate(columns):
            temp[col] = s2t.convert(str(r[i]))
        datas.append(temp)

    columns = [{"Index":  i, "Name":  s2t.convert(i)} for i in columns]
    table = {"columns": columns, "datas": datas}

    return jsonify(table)

@app.route("/api/headers", methods=["POST"])
def get_headers_info():
    request_data = request.get_json()
    result = dbService.get_headers_info(request_data["table_name"])
    return jsonify(result)

@app.route("/api/sql", methods=["POST"])
def get_sql():
    """Convert query to SQL command"""
    request_data = request.get_json()
    question = tw2s.convert(request_data["question"])
    result = modelService.get_sql(question, request_data["table_name"])
    result = s2t.convert(result)

    return jsonify(result)

@app.route("/api/runsql", methods=["POST"])
def run_sql():
    """Execute sql command and return table"""
    request_data = request.get_json()
    sql = tw2s.convert(str(request_data["sql"])).replace('\n', '')

    # replace text to fit SQL format
    columns = sql[6:sql.find("FROM")].replace(')', '').replace(
        '(', '').replace('`', '').replace(' ', '').split(',')

    data = []
    rows = dbService.exe_sql(sql)
    for r in rows:
        temp = {}
        for i, col in enumerate(columns):
            temp[col] = s2t.convert(str(r[i]))
        data.append(temp)

    columns = [{"Index":  i, "Name":  s2t.convert(i)} for i in columns]
    result = {"columns": columns, "datas": data}
    return jsonify(result)


if __name__ == '__main__':
    app.run()