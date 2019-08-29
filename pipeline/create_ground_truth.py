import pickle
import psycopg2
from psycopg2.extras import Json

import json



try:
    # conn = psycopg2.connect(host="127.0.0.1", user="root", password="", database="phdacare", port=5432)
    # conn = psycopg2.connect(host="127.0.0.1", user="phdacareadmin", password="%Gv5xy?yMsR;2D9M", database="phdacare",
    #                         port=9999)
    conn = psycopg2.connect(host="phdacare-postgres-stg.c1vslncdeaao.us-east-1.rds.amazonaws.com", user="phdacareadmin",
                            password="%Gv5xy?yMsR;2D9M", database="mimic3", port=5432)
except:
    print("I am unable to connect to the database")


cursor = conn.cursor()

cursor.execute("""
select abbr_id,cluster_id,sense_id
from mimic3.public.cluster_instance
where sense_id is not null
group by abbr_id,cluster_id,sense_id;

""")


abbrCluster_sense_groundTruth_map = {}
for (abbr_id,cluster_id,sense_id) in cursor:
    abbrCluster_sense_groundTruth_map[(abbr_id,cluster_id)] = sense_id

pickle.dump(abbrCluster_sense_groundTruth_map,open("abbrCluster_sense_ground_truth_mimic","wb"))

# pickle.load(open("abbrCluster_sense_ground_truth_mimic","rb"))