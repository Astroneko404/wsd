import pickle

import pickle
import psycopg2
import math
from psycopg2.extras import Json

import json

### change input cluster file here
# upmc_no_mark = "/home/luoz3/wsd_data/upmc_batch1_4/upmc_no_mark_old.json"
# upmc_cluster_result = "/home/luoz3/wsd_data/upmc_batch1_4/upmc_clustering_result_I_L.json"

# upmc_no_mark = "/Users/zhendongwang/Documents/projects/is/upmc-nlp/code/prepare/adam/load_data/upmc_no_mark.json"
# upmc_cluster_result = "/Users/zhendongwang/Documents/projects/is/upmc-nlp/code/prepare/adam/load_data/upmc_clustering_result.json"

###

try:
    # conn = psycopg2.connect(host="127.0.0.1", user="root", password="", database="phdacare", port=5432)
    # conn = psycopg2.connect(host="127.0.0.1", user="phdacareadmin", password="%Gv5xy?yMsR;2D9M", database="phdacare",
    #                         port=9999)
    conn = psycopg2.connect(host="phdacare-postgres-stg.c1vslncdeaao.us-east-1.rds.amazonaws.com", user="phdacareadmin",
                            password="%Gv5xy?yMsR;2D9M", database="phdacare", port=5432)
except:
    print("I am unable to connect to the database")



cursor = conn.cursor()

cursor.execute("""
select abbr,sense,sense_id from sense
""")

sense_realId = {}
senseId_realId = {}

for (abbr,sense,sense_id) in cursor:
    if (abbr,sense) not in sense_realId:
        sense_realId[(abbr,sense)] = sense_id
        senseId_realId[(abbr,sense)] = sense_id
    else:
        print(abbr,sense)
        senseId_realId[(abbr,sense)] = sense_realId[(abbr,sense)]

print(len(senseId_realId) - len(sense_realId))
