import mysql.connector
import json
class DBService():
    def __init__(self,config):
        
        self.config = config


    # get maxdb & cursor
    def get_connect(self):
        maxdb = mysql.connector.connect(
            host = config["host"],
            user = config["user"],
            password = config["password"],
            database = config["database"],
        )
        cursor=maxdb.cursor()
        return maxdb,cursor

    # 拿到單個table內容物 [(col1,col2,col3),(),()...]
    def get_table(self,table_name):
        try:
            maxdb,cursor = self.get_connect()
            sql = f"SELECT * FROM {table_name}"
            cursor.execute(sql)
            result = cursor.fetchall()
            self.close_conn(maxdb,cursor)
            return result
        except:
            return []


     # 拿到table 列表    回傳 ['name1','name2' ...]
    def get_table_list(self,tablemap):
        maxdb,cursor = self.get_connect()
        sql = f"SELECT * FROM {tablemap}"
        cursor.execute(sql)
        result = cursor.fetchall()
        result = [i[0] for i in result]
        return result

    # 拿到表頭資訊 回傳 [(name,type),()]
    def get_headers(self,table_name):
        try:
            maxdb,cursor = self.get_connect()
            sql = f'SELECT COLUMN_NAME,DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME=\'{table_name}\''
            cursor.execute(sql)
            result = cursor.fetchall()
            self.close_conn(maxdb,cursor)
            return result
        except:
            return []

    # close maxdb & cursor
    def close_conn(self,maxdb,cursor):
        maxdb.close()
        cursor.close()

    # get table column list like: [v1,v2,v3...]
    def get_columns(self,table_name,col_name):
        
        try:
            maxdb,cursor = self.get_connect()
            sql = f'SELECT `{col_name}` FROM {table_name}'
            cursor.execute(sql)
            result = cursor.fetchall()
            result = [i[0] for i in result]
            self.close_conn(maxdb,cursor)
            return result
        except:
            return []

config = {
        'host' : '127.0.0.1',
        'user' : "ericzone",
        'password': "ericzone",
        'database' : "nl2sql",
}

service = DBService(config)


print(service.get_columns('Table_43ad6bdc1d7111e988a6f40f24344a08','平均溢价率(%)'))
print(service.get_headers('Table_43ad6bdc1d7111e988a6f40f24344a08'))
#print(service.get_table('Table_43ad93211d7111e998d9f40f24344a08'))
#print(service.get_table_list('namemap'))