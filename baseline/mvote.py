"""Simple majority vote impl"""
from preprocess.file_helper import pickle_writer, pickle_reader
from baseline.dataset_helper import DataSetPaths, AbbrInstanceCollector, process_abbr_token, evaluation, \
    InstancePred, save_instance_collection_to_json
from datetime import datetime


def predict_majority_vote(train_counter, test_path):
    """
    Make prediction using majority vote, return list of InstancePred.

    :param train_counter:
    :param test_path:
    :return: list of InstancePred
    """
    assign_map = {}
    for abbr in train_counter:
        assign_map[abbr] = train_counter[abbr].most_common(1)[0][0]

    instance_collection = []
    idx = 0
    for line in open(test_path):
        for token in line.rstrip('\n').split(" "):
            items = process_abbr_token(token)
            if items is not None:
                abbr, sense, _ = items
                if abbr in assign_map and sense in train_counter[abbr]:
                    sense_pred = assign_map[abbr]
                else:
                    sense_pred = None
                instance_collection.append(InstancePred(
                    index=idx,
                    abbr=abbr,
                    sense_pred=sense_pred))
                idx += 1
    return instance_collection


def evaluate_score_majority_vote(train_counter, test_path):
    assign_map = {}
    correct_cnt, total_cnt = 0.0, 0.0
    for abbr in train_counter:
        assign_map[abbr] = train_counter[abbr].most_common(1)[0][0]

    for line in open(test_path):
        ents = [w for w in line.split() if w.startswith('abbr|')]
        for ent in ents:
            items = ent.split('|')
            abbr = items[1]
            cui = items[2]
            if abbr in assign_map:
                pred_cui = assign_map[abbr]
                if cui == pred_cui:
                    correct_cnt += 1.0
            total_cnt += 1.0

    acc = correct_cnt / total_cnt
    print('Accuray = %s' % acc)
    print()
    return acc


if __name__ == '__main__':
    # dataset_paths = DataSetPaths('luoz3_x1')
    dataset_paths = DataSetPaths('xil222')
    # train_counter_path = dataset_paths.mimic_train_folder+'train_abbr_counter.pkl'
    start_time = datetime.now()

    ##############################
    # process train abbr info
    ##############################
    # mimic_train_collector = AbbrInstanceCollector(dataset_paths.mimic_train_txt)
    # mimic_train_counter = mimic_train_collector.generate_counter(train_counter_path)

    # mimic_train_counter = pickle_reader(train_counter_path)

    # upmc_ab_train_collector = AbbrInstanceCollector(dataset_paths.upmc_ab_train_txt)
    # upmc_ab_train_counter = upmc_ab_train_collector.generate_counter(
    #     dataset_paths.upmc_ab_train_folder + "/train_abbr_counter.pkl")

    # upmc_ab_train_counter = pickle_reader(dataset_paths.upmc_ab_train_folder+"/train_abbr_counter.pkl")

    # upmc_ad_train_collector = AbbrInstanceCollector(dataset_paths.upmc_ad_train_txt)
    # upmc_ad_train_counter = upmc_ad_train_collector.generate_counter(
    #     dataset_paths.upmc_ad_train_folder + "/train_abbr_counter.pkl")

    # upmc_ag_train_collector = AbbrInstanceCollector(dataset_paths.upmc_ag_train_txt)
    # upmc_ag_train_counter = upmc_ag_train_collector.generate_counter(
    #     dataset_paths.upmc_ag_train_folder + "/train_abbr_counter.pkl")

    # upmc_al_train_collector = AbbrInstanceCollector(dataset_paths.upmc_al_train_txt)
    # upmc_al_train_counter = upmc_al_train_collector.generate_counter(
    #     dataset_paths.upmc_al_train_folder + "/train_abbr_counter.pkl")

    # upmc_ao_train_collector = AbbrInstanceCollector(dataset_paths.upmc_ao_train_txt)
    # upmc_ao_train_counter = upmc_ao_train_collector.generate_counter(
    #     dataset_paths.upmc_ao_train_folder + "/train_abbr_counter.pkl")

    # pe_self_train_collector = AbbrInstanceCollector(dataset_paths.pe_self_train_txt)
    # pe_self_train_counter = pe_self_train_collector.generate_counter(
    #     dataset_paths.pe_self_train_folder + "/train_abbr_counter.pkl")

    mimic_clus_all_train_collector = AbbrInstanceCollector(dataset_paths.mimic_clus_all_train_txt)
    mimic_clus_all_train_counter = mimic_clus_all_train_collector.generate_counter(
        dataset_paths.mimic_clus_all_train_folder + "/train_abbr_counter.pkl")

    #####################################
    # testing (directly compute score, not using standard pipeline)
    #####################################
    # print("Mvote on MIMIC test: ")
    # evaluate_score_majority_vote(mimic_train_counter, dataset_paths.mimic_eval_txt)
    # print("Mvote on ShARe/CLEF: ")
    # evaluate_score_majority_vote(mimic_train_counter, dataset_paths.share_txt)
    # print("Mvote on MSH: ")
    # evaluate_score_majority_vote(mimic_train_counter, dataset_paths.msh_txt)

    #####################################
    # testing (using standard evaluation pipeline)
    #####################################

    # # load test sets
    # mimic_test_collector = AbbrInstanceCollector(dataset_paths.mimic_eval_txt)
    # share_collector = AbbrInstanceCollector(dataset_paths.share_txt)
    # msh_collector = AbbrInstanceCollector(dataset_paths.msh_txt)
    # umn_collector = AbbrInstanceCollector(dataset_paths.umn_txt)
    # upmc_example_collector = AbbrInstanceCollector(dataset_paths.upmc_example_txt)
    # upmc_ab_test_collector = AbbrInstanceCollector(dataset_paths.upmc_ab_test_txt)
    # upmc_ad_test_collector = AbbrInstanceCollector(dataset_paths.upmc_ad_test_txt)
    # upmc_ag_test_collector = AbbrInstanceCollector(dataset_paths.upmc_ag_test_txt)
    # upmc_al_test_collector = AbbrInstanceCollector(dataset_paths.upmc_al_test_txt)
    # upmc_ao_test_collector = AbbrInstanceCollector(dataset_paths.upmc_ao_test_txt)

    mimic_clus_all_test_collector = AbbrInstanceCollector(dataset_paths.mimic_clus_all_test_txt)

    # pe_50000_test_collector = AbbrInstanceCollector(dataset_paths.pe_50000_nm_test_txt)
    # pe_self_test_collector = AbbrInstanceCollector(dataset_paths.pe_self_test_txt)
    # ipdc_test_collector = AbbrInstanceCollector(dataset_paths.ipdc_50000_test_txt)

    # print("Mvote on MIMIC test: ")
    # mimic_test_collection_true = mimic_test_collector.generate_instance_collection()
    # mimic_test_collection_pred = predict_majority_vote(mimic_train_counter, dataset_paths.mimic_eval_txt)
    # print(evaluation(mimic_test_collection_true, mimic_test_collection_pred))
    #
    # print("Mvote on ShARe/CLEF: ")
    # share_collection_true = share_collector.generate_instance_collection()
    # share_collection_pred = predict_majority_vote(mimic_train_counter, dataset_paths.share_txt)
    # print(evaluation(share_collection_true, share_collection_pred))
    #
    # print("Mvote on MSH: ")
    # msh_collection_true = msh_collector.generate_instance_collection()
    # msh_collection_pred = predict_majority_vote(mimic_train_counter, dataset_paths.msh_txt)
    # print(evaluation(msh_collection_true, msh_collection_pred))
    #
    # print("Mvote on UMN: ")
    # umn_collection_true = umn_collector.generate_instance_collection()
    # umn_collection_pred = predict_majority_vote(mimic_train_counter, dataset_paths.umn_txt)
    # print(evaluation(umn_collection_true, umn_collection_pred))
    #
    # print("Mvote on UPMC example: ")
    # upmc_example_collection_true = upmc_example_collector.generate_instance_collection()
    # upmc_example_collection_pred = predict_majority_vote(mimic_train_counter, dataset_paths.upmc_example_txt)
    # print(evaluation(upmc_example_collection_true, upmc_example_collection_pred))
    # save_instance_collection_to_json(upmc_example_collection_pred,
    #                                  dataset_paths.upmc_example_folder + "/upmc_mvote_pred.json")

    # print("Mvote on UPMC AB test: ")
    # upmc_ab_test_collection_true = upmc_ab_test_collector.generate_instance_collection()
    # upmc_ab_test_collection_pred = predict_majority_vote(upmc_ab_train_counter, dataset_paths.upmc_ab_test_txt)
    # print(evaluation(upmc_ab_test_collection_true, upmc_ab_test_collection_pred))

    # print("Mvote on UPMC AD test: ")
    # upmc_ad_test_collection_true = upmc_ad_test_collector.generate_instance_collection()
    # upmc_ad_test_collection_pred = predict_majority_vote(upmc_ad_train_counter, dataset_paths.upmc_ad_test_txt)
    # print(evaluation(upmc_ad_test_collection_true, upmc_ad_test_collection_pred))

    # print("Mvote on UPMC AG test: ")
    # upmc_ag_test_collection_true = upmc_ag_test_collector.generate_instance_collection()
    # upmc_ag_test_collection_pred = predict_majority_vote(upmc_ag_train_counter, dataset_paths.upmc_ag_test_txt)
    # runtime = datetime.now() - start_time
    # print(evaluation(upmc_ag_test_collection_true, upmc_ag_test_collection_pred))
    # print('Finished in', runtime)

    # print("Mvote on UPMC AL test: ")
    # upmc_al_test_collection_true = upmc_al_test_collector.generate_instance_collection()
    # upmc_al_test_collection_pred = predict_majority_vote(upmc_al_train_counter, dataset_paths.upmc_al_test_txt)
    # runtime = datetime.now() - start_time
    # print(evaluation(upmc_al_test_collection_true, upmc_al_test_collection_pred))
    # print('Finished in', runtime)

    # print("Mvote on UPMC AO test: ")
    # upmc_ao_test_collection_true = upmc_ao_test_collector.generate_instance_collection()
    # upmc_ao_test_collection_pred = predict_majority_vote(upmc_ao_train_counter, dataset_paths.upmc_ao_test_txt)
    # runtime = datetime.now() - start_time
    # print(evaluation(upmc_ao_test_collection_true, upmc_ao_test_collection_pred))
    # print('Finished in', runtime)

    print("Mvote on MIMIC cluster only all test: ")
    mimic_clus_all_test_collection_true = mimic_clus_all_test_collector.generate_instance_collection()
    mimic_clus_all_test_collection_pred = predict_majority_vote(
        mimic_clus_all_train_counter,
        dataset_paths.mimic_clus_all_test_txt)
    runtime = datetime.now() - start_time
    print(evaluation(mimic_clus_all_test_collection_true, mimic_clus_all_test_collection_pred))
    print('Finished in', runtime)

    # print("Mvote on PE 50000 test: ")
    # pe_test_collection_true = pe_50000_test_collector.generate_instance_collection()
    # pe_test_collection_pred = predict_majority_vote(upmc_al_train_counter, dataset_paths.pe_50000_nm_test_txt)
    # runtime = datetime.now() - start_time
    # print(evaluation(pe_test_collection_true, pe_test_collection_pred))
    # print('Finished in', runtime)

    # print("Mvote on PE test: ")
    # pe_self_test_collection_true = pe_self_test_collector.generate_instance_collection()
    # pe_self_test_collection_pred = predict_majority_vote(pe_self_train_counter, dataset_paths.pe_self_test_txt)
    # runtime = datetime.now() - start_time
    # print(evaluation(pe_self_test_collection_true, pe_self_test_collection_pred))
    # print('Finished in', runtime)

    # print("Mvote on IPDC test: ")
    # ipdc_test_collection_true = ipdc_test_collector.generate_instance_collection()
    # ipdc_test_collection_pred = predict_majority_vote(upmc_al_train_counter, dataset_paths.ipdc_50000_test_txt)
    # runtime = datetime.now() - start_time
    # print(evaluation(ipdc_test_collection_true, ipdc_test_collection_pred))
    # print('Finished in', runtime)
