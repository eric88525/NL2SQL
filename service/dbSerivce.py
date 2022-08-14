import mysql.connector

class DBService():
    """The service that can interact with the database

    Attributes:
        config: connect information
    """
    def __init__(self, config):
        self.config = config

    def get_connect(self):
        """Get connnection with database"""
        maxdb = mysql.connector.connect(
            host=self.config["host"],
            user=self.config["user"],
            password=self.config["password"],
            database=self.config["database"],
        )
        cursor = maxdb.cursor()
        return maxdb, cursor

    def get_table(self, table_name):
        """Fetch table from database

        Args:
            table_name: the table name in database

        Returns:
            A list format table. Each item is a row of data.
            For example:
            [(10, 'tom'), (30, 'Eric'), (6, 'Jack')...]
        """
        try:
            maxdb, cursor = self.get_connect()
            sql = f"SELECT * FROM {table_name}"
            cursor.execute(sql)
            result = cursor.fetchall()
            self.close_conn(maxdb, cursor)
            return result
        except:
            return []

    def get_table_list(self):
        """Get all the table name in the database"""
        maxdb, cursor = self.get_connect()
        sql = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE' AND TABLE_SCHEMA = '{self.config['database']}';"
        cursor.execute(sql)
        result = cursor.fetchall()
        result = [i[0] for i in result]
        self.close_conn(maxdb, cursor)

        return result

    def get_headers_info(self, table_name):
        """Get target table's column name and data type.

        Returns:
            A list contains two item.
            First item is the column name list, second item is the column data type list.
            For example:
            [
                ['name', 'age', 'gender'],
                ['varchar', 'float', 'varchar']
            ]
        """
        try:
            maxdb, cursor = self.get_connect()
            sql = f'SELECT COLUMN_NAME,DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME=\'{table_name}\''
            cursor.execute(sql)
            result = cursor.fetchall()
            result = [[i[0] for i in result], [i[1] for i in result]]
            self.close_conn(maxdb, cursor)
            return result
        except:
            return []

    def close_conn(self, maxdb, cursor):
        """Disconned with database"""
        maxdb.close()
        cursor.close()

    def get_columns(self, table_name, col_name):
        """Get the enture column from table_name
        
        Returns:
            A list contains column values. For example:
            ['tom', 'jack', 'steve']
        """
        maxdb, cursor = self.get_connect()
        sql = f'SELECT `{col_name}` FROM {table_name}'
        cursor.execute(sql)
        result = cursor.fetchall()
        result = [i[0] for i in result]
        self.close_conn(maxdb, cursor)

        return result

    def exe_sql(self, sql):
        """Execute a SQL command
        Returns:
            SQL commmand execute result. For example:
        """
        maxdb, cursor = self.get_connect()
        sql = str(sql)
        cursor.execute(sql)
        result = cursor.fetchall()
        self.close_conn(maxdb, cursor)

        return result


if __name__ == '__main__':
    config = {
        'host': '127.0.0.1',
        'user': "ericzone",
        'password': "ericzone",
        'database': "nl2sql",
    }
    service = DBService(config)
    ss = "SELECT * FROM student"
    service.get_table_list()
