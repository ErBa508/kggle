import pandas as pd
import psycopg2 as psql

class createSql(object):
    """ 
    createSqlTable will 1. connect to postgres, 2. drop table if exists, 
    3. build new table, 4. close connection. Note, data is added to table
    outside of class and not included as a method here.
    
    Example usage:
        # instantiate
        postgr = createSql('cookingTrain', train, 'postgres', 'postgres')
        # connect
        conn, cur = postgr.connectSql()
        # drop table if exists
        postgr.dropTbl(cur)
        # build new table
        labels = (postgr.table,) + tuple(postgr.colNames)
        create_text = 'CREATE TABLE %s (%s TEXT, %s INTEGER PRIMARY KEY, %s TEXT); ' % labels
        postgr.buildTbl(create_text)
        # add data to table for each row
        rows_text = 'INSERT INTO %s (%s, %s, %s) VALUES (%%s, %%s, %%s);'  % labels
        for i in range(len(postgr.data)):
            cur.execute(rows_text, (postgr.data.iloc[i][0], int(postgr.data.iloc[i][1]), postgr.data.iloc[i][2]))
        # close connection
        postgr.closeSql(conn, cur)
    
    """
    
    def __init__(self, table, data, db, user):
        """Assign raw data. Assumes dataframe """
        self.table = table
        self.data = data
        self.db = db
        self.user = user
        self.colNames = data.columns.values
    
    def connectSql(self):
        connect_text = "dbname=%s user=%s" % (self.db, self.user)
        conn = psql.connect(connect_text)
        cur = conn.cursor()
        return conn, cur
    
    def dropTbl(self, cur):
        drop_text = "DROP TABLE IF EXISTS %s;" % (self.table, )
        cur.execute(drop_text)
        
    def buildTbl(self, create_text):
        cur.execute(create_text)
        
    def closeSql(self, conn, cur):
        conn.commit()
        cur.close()
        conn.close()