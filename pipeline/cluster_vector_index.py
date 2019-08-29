from annoy import AnnoyIndex
import psycopg2
from psycopg2.extras import Json

import json
import pickle

### change input cluster file here
upmc_no_mark = "/home/luoz3/wsd_data/upmc_batch1_4/upmc_no_mark_old.json"
upmc_cluster_result = "/home/luoz3/wsd_data/upmc_batch1_4/upmc_clustering_result_I_L.json"

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


index_cluster_abbr_mapping ={}
## build index to cluster,abbr mapping
with open("/home/wangz12/scripts/cluster_index.map","wb") as mapping_file:
    pickle.dump(index_cluster_abbr_mapping,mapping_file)


DIM = 100

cursor = conn.cursor()


cursor.execute("""
select cluster_abbr_id,cluster_id,word_vec
from cluster
where word_vec is not null;
""")

index = t = AnnoyIndex(DIM)


for i,(cluster_abbr_id,cluster_id,word_vec) in enumerate(cursor):
    index.add_item(i,word_vec)
    index_cluster_abbr_mapping[i] = (cluster_abbr_id,cluster_id)
    if i % 10000 == 0:
        print(i)
## build annoy index
index.build(30)
index.save("/home/wangz12/scripts/cluster_index.ann")
print("index finisehd")

## build index to cluster,abbr mapping
with open("/home/wangz12/scripts/cluster_index.map","wb") as mapping_file:
    pickle.dump(index_cluster_abbr_mapping,mapping_file)


