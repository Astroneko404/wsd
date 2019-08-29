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
                            password="%Gv5xy?yMsR;2D9M", database="mimic3", port=5432)
except:
    print("I am unable to connect to the database")



cursor = conn.cursor()



def loadSenses():
    cursor.execute("""
    select id,sense_id,abbr
    from sense
    """)
    return {row[0]:row[1] for row in cursor}


cursor.execute("""
    select id,abbr
    from cluster_abbr
    """)

abbr_id_map = {abbr:id for (id,abbr) in cursor}


cursor.execute("""
 with t as (select abbr_id, cluster_id,sense_id, "textLiteral", ROW_NUMBER() over (partition by abbr_id,cluster_id order by cluster_instance.id) as row
from cluster_instance
where cluster_id != -1
  and sense_id is not null
  )
select abbr_id, cluster_id, sense_id
from t
where row = 1;
""")


abbrCluster_senseId_groundTruth = { (abbr_id, cluster_id):sense_id for (abbr_id, cluster_id, sense_id) in cursor}




CUIs_set = loadSenses()
CUIs_set["-1"] = -1


print("prepared")

matching_counter = 0

with open("../result_mimic_annotated.pkl","rb") as result_file:
    data = pickle.load(result_file)


    target_wsd = None
    for ((abbr_id,cluster_id), wsds) in data.items():
        abbr_match = 0
        is_success = False

        # print(wsds)
        for wsd in wsds:

            senses = wsd["sense"]
            _abbr = wsd["abbr"]

            if _abbr in abbr_id_map and abbr_id_map[_abbr] == abbr_id:
                ground_truth = CUIs_set[abbrCluster_senseId_groundTruth[(abbr_id, cluster_id)]]
                for sense in senses:
                    sense = sense[0]
                    if ground_truth == sense:
                        matching_counter += 1
                        is_success = True
                        break
                if not is_success:
                    print(abbr_id, cluster_id, ground_truth, senses)
            if is_success:
                break



        else:
            pass

print(matching_counter / len(data))



# total clusters: 3382
# matching clusters: 1866


# total clusters: 8899
# matching clusters: 7257

