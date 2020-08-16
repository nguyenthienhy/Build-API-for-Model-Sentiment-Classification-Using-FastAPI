import MySQLdb

def connect_to(host = None, port = None , user = None , passwd = None , db = None):
    db = MySQLdb.connect(host=host, port=port, user=user, passwd=passwd, db=db)
    c = db.cursor()
    return c , db