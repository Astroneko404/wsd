import pickle
import psycopg2
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

abbr = "PT"

cursor = conn.cursor()

cursor.execute("""
select core_clusters::json
from cluster_abbr
where abbr = '{0}'
""".format(abbr))

core_clusters = list(cursor)[0][0]
core_clusters = set([x["id"] for x in core_clusters])


def loadSenses(abbr):
    cursor.execute("""
    select sense_id,abbr
    from sense
    where abbr ilike '{0}' or sense ilike '* %'
    """.format(abbr))
    return {row[0] for row in cursor}


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
CUIs_set.add("-1")




matching_counter = 0
mismatch_but_core_clusters = set()
with open("../result_mimic_{0}.pkl".format(abbr),"rb") as result_file:
    data = pickle.load(result_file)
    abbr = abbr.lower()

    target_wsd = None
    for (t, doc) in data.items():
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




        if abbr_match_correct > 0 and abbr_match_correct == abbr_match:
            matching_counter += 1
        else:
            mismatch_but_core_clusters.add(t)


    not_annotated_count = len(data)
    print("annotated clusters: {0}".format(annotated_count))
    print("not annotated clusters: {0}".format(not_annotated_count))
    print("can populate clusters: {0}".format(matching_counter))
    print("population ratio: {0}".format(matching_counter / (not_annotated_count + annotated_count)))
    print("mismatch but core cluster: {0}".format(len(mismatch_but_core_clusters.intersection(core_clusters))))


# total clusters: 3382
# matching clusters: 1866


# total clusters: 8899
# matching clusters: 7257

