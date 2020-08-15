#!/usr/bin/env python3
import MySQLdb

def connect_to(host = None, port = None , user = None , passwd = None , db = None):
    # Create db connection
    db = MySQLdb.connect(host=host, port=port, user=user, passwd=passwd, db=db)

    # To perform a query, you first need a cursor
    c = db.cursor()

    # make select version query
    c.execute("select version()")

    return c , db