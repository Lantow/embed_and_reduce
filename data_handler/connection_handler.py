import psycopg2
from functools import wraps
from os import getenv 
import signal
import sys

class PostgresConnection(object):
    """automatic open and close"""    
    def __init__(self):
        self.conn = None 
        self.curr = None

    def _handle_interrupt(self, signum, frame):
        sys.exit("Aborted by KeyboardInterrupt") 

    def __enter__(self):
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        print("Opening connection")
        self.conn = psycopg2.connect(getenv('DJANGO_CONN_STR'))
        self.curr = self.conn.cursor()
        return self    
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection")
        if exc_tb is None or "Aborted by KeyboardInterrupt" in str(exc_val):
            print("Comitting")
            self.conn.commit()
        else:
            self.conn.rollback()
        self.conn.close()

def with_conn(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        with PostgresConnection() as conn_obj:
            return f(*args, conn=conn_obj.conn, curr=conn_obj.curr, **kwds)
    return wrapper
