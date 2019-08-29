from annoy import AnnoyIndex
import psycopg2
from psycopg2.extras import Json

import json
import pickle
import gensim

### change input cluster file here
upmc_no_mark = "/home/luoz3/wsd_data/upmc_batch1_4/upmc_no_mark_old.json"
upmc_cluster_result = "/home/luoz3/wsd_data/upmc_batch1_4/upmc_clustering_result_I_L.json"


# upmc_no_mark = "/Users/zhendongwang/Documents/projects/is/upmc-nlp/code/prepare/adam/load_data/upmc_no_mark.json"
# upmc_cluster_result = "/Users/zhendongwang/Documents/projects/is/upmc-nlp/code/prepare/adam/load_data/upmc_clustering_result.json"

###

def replace_word(word):
    if word == "(":
        return "-LRB-"
    elif word == ")":
        return "-RRB-"
    elif word == "[":
        return "-LSB-"
    elif word == "]":
        return "-RSB-"
    elif word == "LB-LB":
        return "\u21b5"
    return word


try:
    # conn = psycopg2.connect(host="127.0.0.1", user="root", password="", database="phdacare", port=5432)
    # conn = psycopg2.connect(host="127.0.0.1", user="phdacareadmin", password="%Gv5xy?yMsR;2D9M", database="phdacare",
    #                         port=9999)
    conn = psycopg2.connect(host="phdacare-postgres-stg.c1vslncdeaao.us-east-1.rds.amazonaws.com", user="phdacareadmin",
                            password="%Gv5xy?yMsR;2D9M", database="phdacare", port=5432)
except:
    print("I am unable to connect to the database")

model = {}
model = gensim.models.Word2Vec.load('/home/luoz3/wsd_data/upmc_batch1_4/upmc.model')

DIM = 100

cursor = conn.cursor()

index = t = AnnoyIndex(DIM)



index.load('/home/wangz12/scripts/cluster_index.ann')

id_cluster_abbrId_mapping = pickle.load(open("/home/wangz12/scripts/cluster_index.map", "rb"))

text = "Ambulatory . PPI . XX AC with thrombocytopenia".split(" ")
text_ = [x for x in text if x in model]
vector = sum([model[replace_word(x)] for x in text_ if x in model]) / len(text_)

ids = index.get_nns_by_vector(vector, 10,5000000)
for id in ids:
    cursor.execute("""
        select "textLiteral"
        from cluster_instance
        where cluster_id = {1}
          and abbr_id = {0}
        limit 1
        """.format(*id_cluster_abbrId_mapping[id]))
    print(list(cursor)[0][0])
