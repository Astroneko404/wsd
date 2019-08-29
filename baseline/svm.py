"""
Train and test SVM baseline.
"""

import os
import numpy as np
import tqdm
import random
from collections import defaultdict
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from preprocess.file_helper import pickle_reader, pickle_writer
from baseline.dataset_helper import DataSetPaths, InstancePred, AbbrInstanceCollector, evaluation, save_instance_collection_to_json


def train_sample(x, y, sample_size, random_seed=None):
    """
    Sample training data. Maximum sample_size per class.

    :param x:
    :param y:
    :param sample_size:
    :param random_seed:
    :return:
    """
    random.seed(random_seed)
    # collect label pos
    pos_map = defaultdict(list)
    for i, label in enumerate(y):
        pos_map[label].append(i)
    # sample for each class
    sample_pos = []
    for _, positions in pos_map.items():
        if len(positions) <= sample_size:
            sample_pos.extend(positions)
        else:
            sample_pos.extend(random.sample(positions, sample_size))
    # generate samples
    x_train, x_test, y_train, y_test = [], [], [], []
    for i, (data, target) in enumerate(zip(x, y)):
        if i in sample_pos:
            x_train.append(data)
            y_train.append(target)

    return np.vstack(x_train), y_train


def svm_cross_validation(train_processed_path, abbr_idx=0):
    """
    Tuning the parameters on largest abbr in train set.

    :param train_processed_path:
    :param abbr_idx:
    :return:
    """
    content_vector = pickle_reader(train_processed_path + '/content_vectors/%d_vector.pkl' % abbr_idx)
    label2idx = {}
    label_idx = 0
    x = []
    y = []
    for instance_id, doc_id, pos, content_pos, content_vec, content, label in content_vector:
        if label not in label2idx:
            label2idx[label] = label_idx
            label_idx += 1
        x.append(content_vec)
        y.append(label2idx[label])

    x_train, y_train = train_sample(x, y, 500)
    parameters = {'gamma': [1e-4, 1e-3, 1e-2],
                  'C': [1e-1, 1, 10, 100, 1000]}
    model = SVC(kernel='rbf')
    model_cv = GridSearchCV(model, parameters, cv=5).fit(x_train, y_train)
    print(model_cv.best_params_)
    print(model_cv.best_score_)
    return model_cv


def train_svm(train_processed_path):
    # Load abbr index
    abbr_idx_mapper = pickle_reader(train_processed_path+'/abbr_idx_mapper.pkl')
    abbr_cui2idx_inventory = {}
    os.makedirs(train_processed_path+'/svm_models/', exist_ok=True)
    # generate training data & train model
    for abbr, abbr_idx in tqdm.tqdm(abbr_idx_mapper['abbr2idx'].items()):
        content_vector = pickle_reader(train_processed_path + '/content_vectors/%d_vector.pkl' % abbr_idx)
        label2idx = {}
        label_idx = 0
        x = []
        y = []
        for global_instance_idx, doc_id, pos, content_pos, content_vec, content, label in content_vector:
            if label not in label2idx:
                label2idx[label] = label_idx
                label_idx += 1
            x.append(content_vec)
            y.append(label2idx[label])

        abbr_cui2idx_inventory[abbr] = label2idx
        # no need to train if only have 1 CUI
        if len(label2idx) > 1:
            x_train, y_train = train_sample(x, y, 2000)
            # train svm model
            model = SVC(kernel='rbf', gamma=0.01, C=100).fit(x_train, y_train)
            pickle_writer(model, train_processed_path + '/svm_models/%d_svm.pkl' % abbr_idx)
    pickle_writer(abbr_cui2idx_inventory, train_processed_path+'/abbr_cui_idx_inventory.pkl')


def evaluate_score_svm(test_processed_path, train_processed_path):
    # Load abbr index
    abbr_idx_mapper = pickle_reader(test_processed_path + '/abbr_idx_mapper.pkl')
    abbr_idx_mapper_train = pickle_reader(train_processed_path + '/abbr_idx_mapper.pkl')
    abbr2train_idx = abbr_idx_mapper_train['abbr2idx']
    abbr_cui2idx_inventory = pickle_reader(train_processed_path+'/abbr_cui_idx_inventory.pkl')

    count_correct = 0
    count_all = 0
    count_model_correct = 0
    count_model_all = 0
    count_no_label = 0
    count_correct_without_predict = 0
    # generate testing data
    for abbr, abbr_idx in tqdm.tqdm(abbr_idx_mapper['abbr2idx'].items()):
        content_vector = pickle_reader(test_processed_path + '/content_vectors/%d_vector.pkl' % abbr_idx)
        if abbr not in abbr_cui2idx_inventory:
            count_all += len(content_vector)
            count_no_label += len(content_vector)
        else:
            label2idx = abbr_cui2idx_inventory[abbr]
            count_all += len(content_vector)
            x = []
            y = []
            for _, _, _, _, content_vec, _, label in content_vector:
                # if true label not in train collection
                if label not in label2idx:
                    count_no_label += 1
                # if only have 1 CUI
                elif len(label2idx) == 1:
                    count_correct += 1
                    count_correct_without_predict += 1
                # need predict
                else:
                    x.append(content_vec)
                    y.append(label2idx[label])
            # predict
            if len(y) > 0:
                count_model_all += len(y)
                model = pickle_reader(train_processed_path + '/svm_models/%d_svm.pkl' % abbr2train_idx[abbr])
                y_pred = model.predict(np.vstack(x))
                temp_correct = sum(y == y_pred)
                count_correct += temp_correct
                count_model_correct += temp_correct

    print("DataSet Accuracy (all instances): ", count_correct/count_all)
    print("Model Accuracy (only ambiguous instances): ", count_model_correct/count_model_all)
    print("Num.instances: ", count_all)
    print("Num.gt abbr-CUI mapping not found: ", count_no_label)
    print("Num.correct without predict", count_correct_without_predict)
    print()


def predict_svm(test_processed_path, train_processed_path):
    # Load abbr index
    abbr_idx_mapper = pickle_reader(test_processed_path + '/abbr_idx_mapper.pkl')
    abbr_idx_mapper_train = pickle_reader(train_processed_path + '/abbr_idx_mapper.pkl')
    abbr2train_idx = abbr_idx_mapper_train['abbr2idx']
    abbr_cui2idx_inventory = pickle_reader(train_processed_path+'/abbr_cui_idx_inventory.pkl')

    instance_collection = []
    # generate testing data
    for abbr, abbr_idx in tqdm.tqdm(abbr_idx_mapper['abbr2idx'].items()):
        content_vector = pickle_reader(test_processed_path + '/content_vectors/%d_vector.pkl' % abbr_idx)
        if abbr not in abbr_cui2idx_inventory:
            for global_instance_idx, _, _, _, _, _, _ in content_vector:
                instance_collection.append(InstancePred(
                    index=global_instance_idx,
                    abbr=abbr,
                    sense_pred=None
                ))
        else:
            label2idx = abbr_cui2idx_inventory[abbr]
            x = []
            y = []
            global_idx_list = []

            for global_instance_idx, _, _, _, content_vec, _, label in content_vector:
                # if true label not in train collection
                if label not in label2idx:
                    instance_collection.append(InstancePred(
                        index=global_instance_idx,
                        abbr=abbr,
                        sense_pred=None
                    ))
                # if only have 1 CUI
                elif len(label2idx) == 1:
                    instance_collection.append(InstancePred(
                        index=global_instance_idx,
                        abbr=abbr,
                        sense_pred=label
                    ))
                # need predict
                else:
                    x.append(content_vec)
                    y.append(label2idx[label])
                    global_idx_list.append(global_instance_idx)
            # predict
            if len(y) > 0:
                model = pickle_reader(train_processed_path + '/svm_models/%d_svm.pkl' % abbr2train_idx[abbr])

                y_pred = model.predict(np.vstack(x))

                # get idx2label
                idx2label = {}
                for label, idx in label2idx.items():
                    idx2label[idx] = label

                for idx_pred, global_instance_idx in zip(y_pred, global_idx_list):
                    instance_collection.append(InstancePred(
                        index=global_instance_idx,
                        abbr=abbr,
                        sense_pred=idx2label[idx_pred]
                    ))

    # sort collection list based on global instance idx
    instance_collection = sorted(instance_collection, key=lambda x: x.index)
    return instance_collection
    # return instance_collection, skip


if __name__ == '__main__':
    # dataset_paths = DataSetPaths('luoz3_x1')
    dataset_paths = DataSetPaths('xil222')
    start_time = datetime.now()

    #####################################
    # train
    #####################################

    # svm_cross_validation(dataset_paths.mimic_train_folder, abbr_idx=40)

    # train_svm(dataset_paths.mimic_train_folder)
    # train_svm(dataset_paths.upmc_ab_train_folder)
    # train_svm(dataset_paths.upmc_ad_train_folder)
    # train_svm(dataset_paths.upmc_al_train_folder)
    # train_svm(dataset_paths.upmc_ao_train_folder)
    # train_svm(dataset_paths.pe_self_train_folder)

    train_svm(dataset_paths.mimic_clus_all_train_folder)

    # #####################################
    # # testing (directly compute score, not using standard pipeline)
    # #####################################
    # print("SVM on MIMIC test: ")
    # evaluate_score_svm(dataset_paths.mimic_test_folder, dataset_paths.mimic_train_folder)
    # print("SVM on ShARe/CLEF: ")
    # evaluate_score_svm(dataset_paths.share_test_folder, dataset_paths.mimic_train_folder)
    # print("SVM on MSH: ")
    # evaluate_score_svm(dataset_paths.msh_test_folder, dataset_paths.mimic_train_folder)

    #####################################
    # testing (using standard evaluation pipeline)
    #####################################

    # load test sets
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
    # pe_test_collector = AbbrInstanceCollector(dataset_paths.pe_50000_nm_test_txt)
    # pe_self_test_collector = AbbrInstanceCollector(dataset_paths.pe_self_test_txt)
    # ipdc_test_collector = AbbrInstanceCollector(dataset_paths.ipdc_50000_test_txt)

    mimic_test_collector = AbbrInstanceCollector(dataset_paths.mimic_clus_all_test_txt)

    # print("SVM on MIMIC test: ")
    # mimic_test_collection_true = mimic_test_collector.generate_instance_collection()
    # mimic_test_collection_pred = predict_svm(dataset_paths.mimic_test_folder, dataset_paths.mimic_train_folder)
    # print(evaluation(mimic_test_collection_true, mimic_test_collection_pred))
    #
    # print("SVM on ShARe/CLEF: ")
    # share_collection_true = share_collector.generate_instance_collection()
    # share_collection_pred = predict_svm(dataset_paths.share_test_folder, dataset_paths.mimic_train_folder)
    # print(evaluation(share_collection_true, share_collection_pred))
    #
    # print("SVM on MSH: ")
    # msh_collection_true = msh_collector.generate_instance_collection()
    # msh_collection_pred = predict_svm(dataset_paths.msh_test_folder, dataset_paths.mimic_train_folder)
    # print(evaluation(msh_collection_true, msh_collection_pred))
    #
    # print("SVM on UMN: ")
    # umn_collection_true = umn_collector.generate_instance_collection()
    # umn_collection_pred = predict_svm(dataset_paths.umn_test_folder, dataset_paths.mimic_train_folder)
    # print(evaluation(umn_collection_true, umn_collection_pred))
    #
    # print("SVM on UPMC example: ")
    # upmc_example_collection_true = upmc_example_collector.generate_instance_collection()
    # upmc_example_collection_pred = predict_svm(dataset_paths.upmc_example_folder, dataset_paths.mimic_train_folder)
    # print(evaluation(upmc_example_collection_true, upmc_example_collection_pred))
    # save_instance_collection_to_json(upmc_example_collection_pred,
    #                                  dataset_paths.upmc_example_folder + "/upmc_svm_pred.json")

    # print("SVM on UPMC AB test: ")
    # upmc_ab_test_collection_true = upmc_ab_test_collector.generate_instance_collection()
    # upmc_ab_test_collection_pred = predict_svm(dataset_paths.upmc_ab_test_folder, dataset_paths.upmc_ab_train_folder)
    # print(evaluation(upmc_ab_test_collection_true, upmc_ab_test_collection_pred))

    # print("SVM on UPMC AD test: ")
    # upmc_ad_test_collection_true = upmc_ad_test_collector.generate_instance_collection()
    # upmc_ad_test_collection_pred = predict_svm(dataset_paths.upmc_ad_test_folder, dataset_paths.upmc_ad_train_folder)
    # print(evaluation(upmc_ad_test_collection_true, upmc_ad_test_collection_pred))

    # print("SVM on UPMC AG test: ")
    # upmc_ag_test_collection_true = upmc_ag_test_collector.generate_instance_collection()
    # upmc_ag_test_collection_pred = predict_svm(dataset_paths.upmc_ag_test_folder, dataset_paths.upmc_ag_train_folder)
    # print(evaluation(upmc_ag_test_collection_true, upmc_ag_test_collection_pred))
    # runtime = datetime.now() - start_time

    # print("SVM on UPMC AL test: ")
    # upmc_al_test_collection_true = upmc_al_test_collector.generate_instance_collection()
    # upmc_al_test_collection_pred = predict_svm(dataset_paths.upmc_al_test_folder,
    #                                            dataset_paths.upmc_al_train_folder)
    # print(evaluation(upmc_al_test_collection_true, upmc_al_test_collection_pred))
    # runtime = datetime.now() - start_time
    # print('Finished in', runtime)

    # print("SVM on UPMC AO test: ")
    # upmc_al_test_collection_true = upmc_ao_test_collector.generate_instance_collection()
    # upmc_al_test_collection_pred = predict_svm(dataset_paths.upmc_ao_test_folder,
    #                                            dataset_paths.upmc_ao_train_folder)
    # print(evaluation(upmc_al_test_collection_true, upmc_al_test_collection_pred))
    # runtime = datetime.now() - start_time
    # print('Finished in', runtime)
    #
    # print()

    print("SVM on MIMIC test: ")
    mimic_test_collection_true = mimic_test_collector.generate_instance_collection()
    mimic_test_collection_pred = predict_svm(
        dataset_paths.mimic_clus_all_test_folder,
        dataset_paths.mimic_clus_all_train_folder)
    print(evaluation(mimic_test_collection_true, mimic_test_collection_pred))
    runtime = datetime.now() - start_time
    print('Finished in', runtime)

    print()

    # print("SVM on Patient Education test: ")
    # pe_test_collection_true = pe_test_collector.generate_instance_collection()
    # pe_test_collection_pred = predict_svm(dataset_paths.pe_50000_nm_test_folder,
    #                                       dataset_paths.upmc_al_train_folder)
    # print(evaluation(pe_test_collection_true, pe_test_collection_pred))
    # runtime = datetime.now() - start_time
    # print('Finished in', runtime)

    # print("SVM on Patient Education test: ")
    # pe_self_test_collection_true = pe_self_test_collector.generate_instance_collection()
    # pe_self_test_collection_pred = predict_svm(dataset_paths.pe_self_test_folder,
    #                                            dataset_paths.pe_self_train_folder)
    # print(evaluation(pe_self_test_collection_true, pe_self_test_collection_pred))
    # runtime = datetime.now() - start_time
    # print('Finished in', runtime)

    # print("SVM on IPDC test: ")
    # ipdc_test_collection_pred = predict_svm(dataset_paths.ipdc_50000_test_folder,
    #                                         dataset_paths.upmc_al_train_folder)
    # for i in range(len(ipdc_test_collector.generate_instance_collection())):
    # ipdc_test_collection_true = ipdc_test_collector.generate_instance_collection()
    # print(evaluation(ipdc_test_collection_true, ipdc_test_collection_pred))
