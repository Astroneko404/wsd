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

abbr_core_depth_file = "/home/wangz12/scripts/mimic-scripts/abbr_core_depth.pkl"

abbr_id_core_depth_map = pickle.load(open(abbr_core_depth_file,"rb"))

import csv

csv_writer = csv.writer(open("active_learning.csv","w"))
csv_writer.writerow(("abbr","k","s","skip","expand","ratio","error"))


try:
    # conn = psycopg2.connect(host="127.0.0.1", user="root", password="", database="phdacare", port=5432)
    # conn = psycopg2.connect(host="127.0.0.1", user="phdacareadmin", password="%Gv5xy?yMsR;2D9M", database="phdacare",
    #                         port=9999)
    conn = psycopg2.connect(host="phdacare-postgres-stg.c1vslncdeaao.us-east-1.rds.amazonaws.com", user="phdacareadmin",
                            password="%Gv5xy?yMsR;2D9M", database="mimic3", port=5432)
except:
    print("I am unable to connect to the database")


abbrs = ['AV', 'HIV', 'RBC', 'CV', 'PCA', 'EMS', 'PTT', 'BAL', 'AM', 'AP', 'JP', 'PT', 'BP', 'OGT', 'LCX', 'PCP', 'MCA', 'SI', 'LOC', 'EKG']

abbr = "RBC"
k = 30
s = 5

sense_search_depth = 5
nearest_depth = 5
is_end_early = True

for k in [20,30,40]:
    for s in [5,10,15]:
        for abbr in abbrs:
            abbr_annotatedCores_map = pickle.load(open(
                "/home/wangz12/scripts/generate_trainning_data/abbr_annotated_core_clusters_map_top{0}_{1}.pkl".format(
                    k, s), "rb"))


            def load_nearest_cluster(abbr_id, core_cluster_ids, abbrCluster_senseId_groundTruth, nearest_depth):

                cursor = conn.cursor()
                cursor.execute("""
                select cluster_id,near_cluster_ids
                from cluster
                where cluster_abbr_id = {0}
                  and cluster_id in ({1})
                """.format(abbr_id, ",".join([str(x) for x in core_cluster_ids])))

                n_clusters = set()

                for cluster_id, near_cluster_ids in cursor:
                    for n_cluster_id in near_cluster_ids[:nearest_depth]:
                        if n_cluster_id[0] in abbrCluster_senseId_groundTruth:
                            n_clusters.add(n_cluster_id[0])
                return n_clusters


            cursor = conn.cursor()

            cursor.execute("""
            select core_clusters::json
            from cluster_abbr
            where abbr = '{0}'
            """.format(abbr))

            core_clusters = list(cursor)[0][0]
            core_clusters = set([x["id"] for x in core_clusters])

            cursor.execute("""
            select id,abbr
            from cluster_abbr
            """)

            abbrId_abbr_map = {id: abbr for (id, abbr) in cursor}
            abbr_id_map = {abbr: id for (id, abbr) in abbrId_abbr_map.items()}

            abbrs_for_train = [abbrId_abbr_map[abbr_id] for abbr_id in abbr_annotatedCores_map.keys()]


            def load_senses_for_core_training_data(abbr_id, core_cluster_ids):
                cursor = conn.cursor()
                cursor.execute("""
                    select sense_id
                    from cluster_instance
                    where abbr_id = {0}
                      and cluster_id in ({1})
                      and sense_id is not null
                    group by sense_id
                    """.format(abbr_id, ",".join([str(x) for x in core_cluster_ids])))
                return [sense_id for (sense_id,) in cursor]


            def load_all_senses_for_abbr(abbr_id):
                cursor = conn.cursor()
                cursor.execute("""
                    select sense_id
                    from cluster_instance
                    where abbr_id = {0}
                      and sense_id is not null
                    group by sense_id
                    """.format(abbr_id))
                return [sense_id for (sense_id,) in cursor]


            def loadSenses(abbr):
                cursor.execute("""
                select sense_id,abbr
                from sense
                where abbr ilike '{0}' or sense ilike '* %'
                """.format(abbr))
                return {row[0] for row in cursor}


            def loadSensesMap():
                cursor.execute("""
                select id,sense_id,abbr
                from sense
                """)
                return {row[0]: row[1] for row in cursor}


            id_senseId_map = loadSensesMap()

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

            cursor.execute("""
             with t as (select abbr_id, cluster_id,sense_id, "textLiteral", ROW_NUMBER() over (partition by abbr_id,cluster_id order by cluster_instance.id) as row
            from cluster_instance
            where cluster_id != -1
              and sense_id is not null
              and abbr_id = {0}
              )
            select abbr_id, cluster_id, sense_id
            from t
            where row = 1;
            """.format(abbr_id_map[abbr]))

            #

            abbrCluster_senseId_groundTruth = {cluster_id: sense_id for (abbr_id, cluster_id, sense_id) in cursor}
            #
            #
            abbrCluster_senseId_groundTruth_keys = set(abbrCluster_senseId_groundTruth.keys())

            abbr_id = abbr_id_map[abbr]
            # nearest_core_clusters = load_nearest_cluster(abbr_id, abbr_annotatedCores_map[abbr_id],
            #                                              abbrCluster_senseId_groundTruth, nearest_depth)

            print(abbrs_for_train)
            # print(load_senses_for_core_training_data(abbr_id,core_clusters))
            # print(load_all_senses_for_abbr(abbr_id))


            matching_counter = 0
            ground_truth_match_count = 0
            ground_truth_score_accumulator = 0
            near_ground_truth_match_count = 0

            mismatch_but_core_clusters = set()

            depth = 0
            target_depth = 0

            addition = 0
            annotated_count = 0

            depth_bound = abbr_id_core_depth_map[abbr_id]

            missed_core_count = 0
            skiped_core_count = 0

            with open("../result_mimic_k{1}_s{2}_{0}.pkl".format(abbr, k, s), "rb") as result_file:
                data = pickle.load(result_file)
                abbr = abbr.lower()

                cursor.execute("""
                   select core_clusters
                   from cluster_abbr
                   where id = {0}
                   """.format(abbr_id))

                core_clusters = [x["id"] for x in list(cursor)[0][0]]

                cursor.execute("""
                select DISTINCT cluster_id
                from cluster_instance
                where sense_id is not null
                  and abbr_id = {0}
                """.format(abbr_id))

                annotated_clusters = {x[0] for x in cursor}

                annotated_core_clusters = [x for x in core_clusters if x in annotated_clusters]

                begin_expand = core_clusters.index(annotated_core_clusters[s - 1])

                next_5_core_clusters = annotated_core_clusters[s:(s + 5)]
                the_lastone_of_next_5 = next_5_core_clusters[-1]
                next_5_core_clusters = set(next_5_core_clusters)

                for i, core_cluster_id in enumerate(core_clusters):
                    if core_cluster_id in annotated_clusters:
                        depth += (addition + 1)

                        addition = 0
                        annotated_count += 1
                        if annotated_count == s:
                            target_depth = depth

                    else:
                        addition += 1

                # result.append((annotated_count,depth,annotated_abbrId_senseCount[abbr_id],total_depth,abbrId_abbr_map[abbr_id],abbr_id))

                target_wsd = None
                core_cluster_cursor = 0
                core_cluster_annotated_cursor = 0
                not_reach_next_5_end = True

                # calculate match

                for (cluster_id, wsds) in data.items():
                    abbr_match = 0
                    abbr_match_correct = 0

                    if cluster_id in core_clusters:
                        core_cluster_cursor += 1
                        if cluster_id in annotated_clusters:
                            core_cluster_annotated_cursor += 1

                    # print(wsds)
                    for wsd in wsds:

                        sense = wsd["sense"][0][0]
                        _abbr = wsd["abbr"]

                        if _abbr.lower() == abbr:
                            abbr_match += 1
                            if sense in CUIs_set:
                                abbr_match_correct += 1

                    is_success = False
                    if abbr_match_correct > 0 and abbr_match_correct == abbr_match:
                        matching_counter += 1
                        is_success = True

                    else:
                        mismatch_but_core_clusters.add(cluster_id)
                        is_success = False

                target_wsd = None
                core_cluster_cursor = 0
                core_cluster_annotated_cursor = 0
                not_reach_next_5_end = True

                # calculate active learning expansion match
                cluster_ground_truth = []
                for cluster_id in core_clusters:
                    for (_cluster_id, wsds) in data.items():
                        if cluster_id == _cluster_id:
                            cluster_ground_truth.append((cluster_id, wsds))

                for (cluster_id, wsds) in cluster_ground_truth:
                    abbr_match = 0
                    abbr_match_correct = 0

                    if cluster_id in core_clusters:
                        core_cluster_cursor += 1
                        if cluster_id in annotated_clusters:
                            core_cluster_annotated_cursor += 1

                    is_correct = False

                    # check whether prediction is correct
                    for wsd in wsds:

                        senses = wsd["sense"]
                        _abbr = wsd["abbr"]

                        if _abbr.lower() == abbr:

                            if cluster_id not in abbrCluster_senseId_groundTruth_keys:
                                continue

                            ground_truth = id_senseId_map[abbrCluster_senseId_groundTruth[cluster_id]]
                            for sense in senses[:sense_search_depth]:
                                score = sense[1]
                                sense = sense[0]

                                if ground_truth == sense:
                                    # ground_truth_match_count += 1
                                    # ground_truth_score_accumulator += score

                                    is_correct = True
                                    break
                            if not is_correct:
                                # print( cluster_id, ground_truth, senses)
                                pass
                        if is_correct:
                            break

                    # print(wsds)
                    for wsd in wsds:

                        sense = wsd["sense"][0][0]
                        _abbr = wsd["abbr"]

                        if _abbr.lower() == abbr:
                            abbr_match += 1
                            if sense in CUIs_set:
                                abbr_match_correct += 1
                            if cluster_id not in abbrCluster_senseId_groundTruth:
                                continue
                                # ground_truth = id_senseId_map[abbrCluster_senseId_groundTruth[cluster_id]]
                                # if sense == ground_truth:
                                #     is_correct = True

                    is_success = False
                    if abbr_match_correct > 0 and abbr_match_correct == abbr_match:
                        # matching_counter += 1
                        is_success = True

                    else:
                        # mismatch_but_core_clusters.add(cluster_id)
                        is_success = False

                    if core_cluster_annotated_cursor > s and core_cluster_cursor < depth_bound:

                        if is_success and not_reach_next_5_end:
                            if core_cluster_cursor in annotated_clusters:
                                if is_correct:
                                    pass
                                else:
                                    missed_core_count += 1

                            else:
                                skiped_core_count += 1
                    if is_end_early:
                        if cluster_id == the_lastone_of_next_5:
                            not_reach_next_5_end = False

                # caculate how many ground truth is correct
                for (cluster_id, wsds) in data.items():
                    abbr_match = 0
                    is_success = False

                    # print(wsds)
                    for wsd in wsds:

                        senses = wsd["sense"]
                        _abbr = wsd["abbr"]

                        if _abbr.lower() == abbr:

                            if cluster_id not in abbrCluster_senseId_groundTruth_keys:
                                continue

                            ground_truth = id_senseId_map[abbrCluster_senseId_groundTruth[cluster_id]]
                            for sense in senses[:sense_search_depth]:
                                score = sense[1]
                                sense = sense[0]

                                if ground_truth == sense:
                                    ground_truth_match_count += 1
                                    ground_truth_score_accumulator += score

                                    is_success = True
                                    break
                            if not is_success:
                                # print( cluster_id, ground_truth, senses)
                                pass
                        if is_success:
                            break

                ## calculate how many near instances is correct
                # for (cluster_id, wsds) in data.items():
                #     abbr_match = 0
                #     is_success = False
                #     if cluster_id not in nearest_core_clusters:
                #         continue
                #     # print(wsds)
                #     for wsd in wsds:
                #
                #         senses = wsd["sense"]
                #         _abbr = wsd["abbr"]
                #
                #         if _abbr.lower() == abbr:
                #             if cluster_id not in abbrCluster_senseId_groundTruth:
                #                 continue
                #             ground_truth = id_senseId_map[abbrCluster_senseId_groundTruth[cluster_id]]
                #             for sense in senses[:sense_search_depth]:
                #                 sense = sense[0]
                #                 if ground_truth == sense:
                #                     near_ground_truth_match_count += 1
                #                     is_success = True
                #                     break
                #             if not is_success:
                #                 # print( cluster_id, ground_truth, senses)
                #                 pass
                #         if is_success:
                #             break

                not_annotated_count = len(data) - s
                # print("annotated clusters: {0}".format(s))
                # print("not annotated clusters: {0}".format(not_annotated_count))
                # print("can populate clusters: {0}".format(matching_counter))
                # print("population ratio: {0}".format(matching_counter / (not_annotated_count + annotated_count)))
                #
                # print("---------------")
                # print("review count for top {0}: {1}".format(s, target_depth))
                # print("skip count for top {0}: {1}".format(s, skiped_core_count))
                #
                end_expand = core_clusters.index(the_lastone_of_next_5)
                # print("expand area {0}: [{1}, {2}]".format(s, begin_expand, end_expand))
                # print("expand length: {0}".format(end_expand - begin_expand))
                # print("expand ratio: {0}".format(skiped_core_count / (end_expand - begin_expand)))
                #
                # print("missed count for top {0}: {1}".format(s, missed_core_count))
                #
                # print("---------------")
                #
                # print("total core: {0}".format(len(core_clusters)))
                # print("mismatch but core cluster: {0}".format(
                #     len(mismatch_but_core_clusters.intersection(core_clusters))))
                #
                # print("ground truth count {0}".format(len(abbrCluster_senseId_groundTruth)))
                # print("ground truth match count {0}".format(ground_truth_match_count))
                #
                # print("ground truth match score {0}".format(
                #     ground_truth_match_count / len(abbrCluster_senseId_groundTruth)))
                #
                # print("nearest count {0}".format(len(nearest_core_clusters)))
                # print("nearest ground truth match count {0}".format(near_ground_truth_match_count))


                print(abbr,k,s)
                csv_writer.writerow((abbr, k, s, skiped_core_count, end_expand - begin_expand, skiped_core_count / (end_expand - begin_expand), missed_core_count))











# total clusters: 3382
# matching clusters: 1866


# total clusters: 8899
# matching clusters: 7257

