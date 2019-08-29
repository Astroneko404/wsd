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

abbr = "AAA"

cursor = conn.cursor()

def loadCommonSenses(cui_set):
    cursor.execute("""
        select sense_id
        from sense
        where sense ilike '*%'
        """)
    for (sense_id,) in cursor:
        cui_set[sense_id] = sense_id

def loadSenses(abbr):
    cursor.execute("""
    select sense_id,id
    from sense
    where abbr ilike '{0}'
    """.format(abbr))
    return {row[0]:row[1] for row in cursor}

def get_abbr_id(abbr):
    cursor.execute("""
        select id
        from cluster_abbr
        where abbr = '{0}'
        """.format(abbr))
    return [row[0] for row in cursor][0]

abbr_id = get_abbr_id(abbr)

def batch_update(to_update, batch_size = 1000):
    batch_count = math.ceil(len(to_update) / batch_size)
    for i in range(batch_count):
        sql = ";".join(to_update[i:(i+1)*batch_size])
        cursor.execute(sql)
        conn.commit()
        print("load {0}".format((i+1) * 1000))

cursor.execute("""
with t as (select
from cluster_instance, cluster_abbr
where cluster_instance.abbr_id = cluster_abbr.id
  and cluster_abbr.abbr = '{0}'
  and sense_id is not null
group by cluster_instance.cluster_id)
select count(1)
from t;
""".format(abbr))

annotated_count = list(cursor)[0][0]

CUIs_set = loadSenses(abbr)
loadCommonSenses(CUIs_set)
CUIs_set["-1"] = -1

matching_counter = 0

to_update = []

with open("../result_{0}.pkl".format(abbr),"rb") as result_file:
    data = pickle.load(result_file)
    abbr = abbr.lower()

    target_wsd = None
    for (cluster_id, doc) in data.items():
        abbr_match = 0
        abbr_match_correct = 0

        wsds = doc["wsd"]
        text = doc["tokenized_text"]
        # print(wsds)
        for wsd in wsds:
            position = wsd["position"]
            sense = wsd["sense"]

            if text[int(position)].lower() == abbr:
                abbr_match +=1
                if sense in CUIs_set:
                    abbr_match_correct +=1
                    to_update.append("""
                    update cluster
                      set predicate_sense_id = {2}
                    where cluster_abbr_id = {0}
                      and cluster_id = {1}
                    """.format(abbr_id,cluster_id,CUIs_set[sense]).strip())


        if abbr_match_correct > 0 and abbr_match_correct == abbr_match:
            matching_counter += 1

    not_annotated_count = len(data)
    print("annotated clusters: {0}".format(annotated_count))
    print("not annotated clusters: {0}".format(not_annotated_count))
    print("can populate clusters: {0}".format(matching_counter))
    print("population ratio: {0}".format(matching_counter / (not_annotated_count + annotated_count)))

    batch_update(to_update)




# total clusters: 3382
# matching clusters: 1866


# total clusters: 8899
# matching clusters: 7257

