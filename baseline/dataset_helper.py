"""
Helper functions for processed DataSet files.

"""

import tqdm
import json
import pickle

from datetime import datetime
from joblib import Parallel, delayed

from collections import Counter, defaultdict, namedtuple, OrderedDict
from model.model_config import get_path
from preprocess.file_helper import pickle_writer, txt_reader, pickle_reader, json_writer, json_reader


class DataSetPaths:
    """
    Paths of DataSets
    """

    def __init__(self, environment):

        # DataSet Corpus files
        if environment == 'luoz3_x1':
            mimic_base_folder = "/home/mengr/Project/wsd/wsd_data/mimic/data"
            self.mimic_train_txt = mimic_base_folder + "/train"
            self.mimic_eval_txt = mimic_base_folder + "/eval"
        elif environment == 'xil222':
            mimic_base_folder = "/home/mengr/Project/wsd/wsd_data/mimic/data"
            self.mimic_train_txt = mimic_base_folder + "/train"
            self.mimic_eval_txt = mimic_base_folder + "/eval"
        else:
            self.mimic_train_txt = "/exp_data/wsd_data/mimic/train"
            self.mimic_eval_txt = "/exp_data/wsd_data/mimic/eval"
            # self.mimic_train_txt = get_path('../wsd_data/mimic/train', env=environment)
            # self.mimic_eval_txt = get_path('../wsd_data/mimic/eval', env=environment)
            # # mimic v1 (deprecated)
            # mimic_train_txt = '/home/zhaos5/projs/wsd/wsd_data/mimic/train'
            # mimic_eval_txt = '/home/zhaos5/projs/wsd/wsd_data/mimic/eval'

        self.share_txt = get_path('../wsd_data/share/processed/share_all_processed.txt', env=environment)
        self.msh_txt = get_path('../wsd_data/msh/msh_processed/msh_processed.txt', env=environment)
        self.umn_txt = get_path('../wsd_data/umn/umn_processed/umn_processed.txt', env=environment)

        self.pe_test_txt = get_path('../wsd_data_test/PE/processed/upmc_test.txt', env=environment)
        self.pe_self_train_txt = get_path('../wsd_data_test/PE_SELF/processed/upmc_train.txt', env=environment)
        self.pe_self_test_txt = get_path('../wsd_data_test/PE_SELF/processed/upmc_test.txt', env=environment)
        self.pe_50000_nm_test_txt = get_path('../wsd_data_test/PE_50000_NO_M/processed/upmc_test.txt',
                                             env=environment)

        self.ipdc_50000_test_txt = get_path('../wsd_data_test/IPDC_50000/processed/upmc_test.txt',
                                            env=environment)

        self.upmc_example_txt = get_path('../wsd_data/upmc/example/processed/upmc_example_processed.txt',
                                         env=environment)
        self.upmc_ab_train_txt = get_path('../wsd_data/upmc/AB/processed/upmc_ab_train.txt', env=environment)
        self.upmc_ab_test_txt = get_path('../wsd_data/upmc/AB/processed/upmc_ab_test.txt', env=environment)
        self.upmc_all_no_mark_txt = get_path('../wsd_data/upmc/batch1_4/processed/train_no_mark.txt', env=environment)
        # self.upmc_ad_train_txt = get_path('../wsd_data/upmc/AD/processed/upmc_train.txt', env=environment)
        # self.upmc_ad_test_txt = get_path('../wsd_data/upmc/AD/processed/upmc_test.txt', env=environment)
        self.upmc_ad_train_txt = get_path('../wsd_data_test/upmc/AD/processed/upmc_train.txt', env=environment)
        self.upmc_ad_test_txt = get_path('../wsd_data_test/upmc/AD/processed/upmc_test.txt', env=environment)
        self.upmc_ag_train_txt = get_path('../wsd_data_test/upmc/AG/processed/upmc_train.txt', env=environment)
        self.upmc_ag_test_txt = get_path('../wsd_data_test/upmc/AG/processed/upmc_test.txt', env=environment)
        self.upmc_al_train_txt = get_path('../wsd_data_test/upmc/AL/processed/upmc_train.txt', env=environment)
        self.upmc_al_test_txt = get_path('../wsd_data_test/upmc/AL/processed/upmc_test.txt', env=environment)
        self.upmc_ao_train_txt = get_path('../wsd_data_test/upmc/AO/processed/upmc_train.txt', env=environment)
        self.upmc_ao_test_txt = get_path('../wsd_data_test/upmc/AO/processed/upmc_test.txt', env=environment)

        self.mimic_new_train_txt = get_path(
            '../wsd_data_test/mimic_test/processed/upmc_train.txt',
            env=environment)
        self.mimic_new_test_txt = get_path(
            '../wsd_data_test/mimic_test/processed/upmc_test.txt',
            env=environment)

        self.mimic_clus_all_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_all/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_all_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_all/processed/upmc_test.txt',
            env=environment)

        self.mimic_clus_k20_s5_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k20_s5/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k20_s5_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k20_s5/processed/upmc_test.txt',
            env=environment)
        self.mimic_clus_k20_s10_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k20_s10/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k20_s10_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k20_s10/processed/upmc_test.txt',
            env=environment)
        self.mimic_clus_k20_s15_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k20_s15/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k20_s15_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k20_s15/processed/upmc_test.txt',
            env=environment)

        self.mimic_clus_k30_s5_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k30_s5/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k30_s5_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k30_s5/processed/upmc_test.txt',
            env=environment)
        self.mimic_clus_k30_s10_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k30_s10/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k30_s10_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k30_s10/processed/upmc_test.txt',
            env=environment)
        self.mimic_clus_k30_s15_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k30_s15/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k30_s15_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k30_s15/processed/upmc_test.txt',
            env=environment)

        self.mimic_clus_k40_s5_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k40_s5/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k40_s5_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k40_s5/processed/upmc_test.txt',
            env=environment)
        self.mimic_clus_k40_s10_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k40_s10/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k40_s10_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k40_s10/processed/upmc_test.txt',
            env=environment)
        self.mimic_clus_k40_s15_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k40_s15/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k40_s15_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k40_s15/processed/upmc_test.txt',
            env=environment)

        self.mimic_clus_k2_s10_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k2_s10/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k2_s10_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k2_s10/processed/upmc_test.txt',
            env=environment)
        self.mimic_clus_k5_s10_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k5_s10/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k5_s10_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k5_s10/processed/upmc_test.txt',
            env=environment)
        self.mimic_clus_k10_s10_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k10_s10/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k10_s10_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k10_s10/processed/upmc_test.txt',
            env=environment)
        self.mimic_clus_k15_s10_train_txt = get_path(
            '../wsd_data_test/mimic_cluster_k15_s10/processed/upmc_train.txt',
            env=environment)
        self.mimic_clus_k15_s10_test_txt = get_path(
            '../wsd_data_test/mimic_cluster_k15_s10/processed/upmc_test.txt',
            env=environment)

        # paths for processed files
        # self.mimic_train_folder = get_path('../wsd_data/mimic/processed/train/', env=environment)
        # self.mimic_test_folder = get_path('../wsd_data/mimic/processed/test/', env=environment)
        # self.share_test_folder = get_path('../wsd_data/share/processed/test/', env=environment)
        # self.msh_test_folder = get_path('../wsd_data/msh/msh_processed/test/', env=environment)
        # self.umn_test_folder = get_path('../wsd_data/umn/umn_processed/test/', env=environment)
        # self.upmc_example_folder = get_path('../wsd_data/upmc/example/processed/test/', env=environment)
        # self.upmc_ab_train_folder = get_path('../wsd_data/upmc/AB/processed/train/', env=environment)
        # self.upmc_ab_test_folder = get_path('../wsd_data/upmc/AB/processed/test/', env=environment)
        # self.upmc_ad_train_folder = get_path('../wsd_data/upmc/AD/processed/train/', env=environment)
        # self.upmc_ad_test_folder = get_path('../wsd_data/upmc/AD/processed/test/', env=environment)
        # self.upmc_all_no_mark_folder = get_path('../wsd_data/upmc/batch1_4/processed/', env=environment)
        self.mimic_train_folder = get_path('../wsd_data_test/mimic/processed/train/', env=environment)
        self.mimic_test_folder = get_path('../wsd_data_test/mimic/processed/test/', env=environment)
        self.share_test_folder = get_path('../wsd_data_test/share/processed/test/', env=environment)
        self.msh_test_folder = get_path('../wsd_data_test/msh/msh_processed/test/', env=environment)

        self.mimic_new_train_folder = get_path('../wsd_data_test/mimic_test/processed/train/', env=environment)
        self.mimic_new_test_folder = get_path('../wsd_data_test/mimic_test/processed/test/', env=environment)

        self.mimic_clus_all_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_all/processed/train/',
            env=environment)
        self.mimic_clus_all_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_all/processed/test/',
            env=environment)

        self.mimic_clus_k20_s5_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k20_s5/processed/train/',
            env=environment)
        self.mimic_clus_k20_s5_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k20_s5/processed/test/',
            env=environment)
        self.mimic_clus_k20_s10_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k20_s10/processed/train/',
            env=environment)
        self.mimic_clus_k20_s10_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k20_s10/processed/test/',
            env=environment)
        self.mimic_clus_k20_s15_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k20_s15/processed/train/',
            env=environment)
        self.mimic_clus_k20_s15_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k20_s15/processed/test/',
            env=environment)

        self.mimic_clus_k30_s5_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k30_s5/processed/train/',
            env=environment)
        self.mimic_clus_k30_s5_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k30_s5/processed/test/',
            env=environment)
        self.mimic_clus_k30_s10_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k30_s10/processed/train/',
            env=environment)
        self.mimic_clus_k30_s10_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k30_s10/processed/test/',
            env=environment)
        self.mimic_clus_k30_s15_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k30_s15/processed/train/',
            env=environment)
        self.mimic_clus_k30_s15_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k30_s15/processed/test/',
            env=environment)

        self.mimic_clus_k40_s5_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k40_s5/processed/train/',
            env=environment)
        self.mimic_clus_k40_s5_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k40_s5/processed/test/',
            env=environment)
        self.mimic_clus_k40_s10_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k40_s10/processed/train/',
            env=environment)
        self.mimic_clus_k40_s10_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k40_s10/processed/test/',
            env=environment)
        self.mimic_clus_k40_s15_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k40_s15/processed/train/',
            env=environment)
        self.mimic_clus_k40_s15_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k40_s15/processed/test/',
            env=environment)

        self.mimic_clus_k2_s10_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k2_s10/processed/train/',
            env=environment)
        self.mimic_clus_k2_s10_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k2_s10/processed/test/',
            env=environment)
        self.mimic_clus_k5_s10_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k5_s10/processed/train/',
            env=environment)
        self.mimic_clus_k5_s10_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k5_s10/processed/test/',
            env=environment)
        self.mimic_clus_k10_s10_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k10_s10/processed/train/',
            env=environment)
        self.mimic_clus_k10_s10_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k10_s10/processed/test/',
            env=environment)
        self.mimic_clus_k15_s10_train_folder = get_path(
            '../wsd_data_test/mimic_cluster_k15_s10/processed/train/',
            env=environment)
        self.mimic_clus_k15_s10_test_folder = get_path(
            '../wsd_data_test/mimic_cluster_k15_s10/processed/test/',
            env=environment)

        self.pe_train_folder = get_path('../wsd_data_test/PE/processed/train/', env=environment)
        self.pe_test_folder = get_path('../wsd_data_test/PE/processed/test/', env=environment)
        self.pe_50000_nm_test_folder = get_path('../wsd_data_test/PE_50000_NO_M/processed/test/', env=environment)
        self.pe_50000_self_train_folder = get_path('../wsd_data_test/PE_50000_SELF/processed/train/',
                                                   env=environment)
        self.pe_50000_self_test_folder = get_path('../wsd_data_test/PE_50000_SELF/processed/test/',
                                                  env=environment)
        self.pe_self_train_folder = get_path('../wsd_data_test/PE_SELF/processed/train/', env=environment)
        self.pe_self_test_folder = get_path('../wsd_data_test/PE_SELF/processed/test/', env=environment)

        self.ipdc_50000_test_folder = get_path('../wsd_data_test/IPDC_50000/processed/test/', env=environment)

        self.pipeline_test_txt = get_path('../wsd_data_test/pipeline_test/processed/upmc_test.txt', env=environment)
        self.pipeline_test_folder = get_path('../wsd_data_test/pipeline_test/processed/test/', env=environment)

        self.umn_test_folder = get_path('../wsd_data_test/umn/umn_processed/test/', env=environment)
        self.upmc_example_folder = get_path('../wsd_data_test/upmc/example/processed/test/', env=environment)
        self.upmc_ab_train_folder = get_path('../wsd_data_test/upmc/AB/processed/train/', env=environment)
        self.upmc_ab_test_folder = get_path('../wsd_data_test/upmc/AB/processed/test/', env=environment)
        self.upmc_ad_train_folder = get_path('../wsd_data_test/upmc/AD/processed/train/', env=environment)
        self.upmc_ad_test_folder = get_path('../wsd_data_test/upmc/AD/processed/test/', env=environment)
        self.upmc_ag_train_folder = get_path('../wsd_data_test/upmc/AG/processed/train/', env=environment)
        self.upmc_ag_test_folder = get_path('../wsd_data_test/upmc/AG/processed/test/', env=environment)
        self.upmc_al_train_folder = get_path('../wsd_data_test/upmc/AL/processed/train/', env=environment)
        self.upmc_al_test_folder = get_path('../wsd_data_test/upmc/AL/processed/test/', env=environment)
        self.upmc_ao_train_folder = get_path('../wsd_data_test/upmc/AO/processed/train/', env='docker')
        self.upmc_ao_test_folder = get_path('../wsd_data_test/upmc/AO/processed/test/', env='docker')
        self.upmc_all_no_mark_folder = get_path('../wsd_data_test/upmc/batch1_4/processed/', env=environment)

        # path to sense inventory
        self.sense_inventory_json = get_path(
            '../wsd_data/sense_inventory/final_cleaned_sense_inventory_with_testsets.json', env=environment)
        self.sense_inventory_pkl = get_path(
            '../wsd_data/sense_inventory/final_cleaned_sense_inventory_with_testsets.pkl', env=environment)


def process_abbr_token(token):
    """
    Unpack token (e.g., abbr|ab|C1234567|long_form).
    If instance format of DataSet change, please only change here.

    :param token:
    :return:
    """
    items = token.split("|")
    if len(items) == 4:
        _, abbr, sense, long_form = items
        return abbr, sense, long_form
    elif len(items) == 3:
        _, abbr, sense = items
        return abbr, sense, None


def chunk(txt_list, n):
    result = []
    for i in range(0, len(txt_list), n):
        print(txt_list[i:i + n])
        result.append(txt_list[i:i + n])
    return result


class AbbrInstanceCollector:
    """
    Collect abbr instance information from processed DataSets.
    """

    def __init__(self, dataset_file_path):
        self.corpus = txt_reader(dataset_file_path)

    def generate_instance_collection(self, save_collection_path=None):
        """
        Collect list of instances (index, abbr, sense, long_form).

        :param save_collection_path:
        :return:
        """
        instance_collection = []
        global_instance_idx = 0
        for line in self.corpus:
            for token in line.split(" "):
                items = process_abbr_token(token)
                if items is not None:
                    abbr, sense, long_form = items
                    instance_collection.append(Instance(
                        index=global_instance_idx,
                        abbr=abbr,
                        sense=sense,
                        long_form=long_form))
                    global_instance_idx += 1

        # save instance collection
        if save_collection_path is not None:
            pickle_writer(instance_collection, save_collection_path)
        return instance_collection

    def generate_counter(self, save_collection_path=None):
        """
        Generate Counters for every abbr-CUI mappings.

        :param save_collection_path:
        :return:
        """
        dataset_counter = defaultdict(Counter)
        for line in self.corpus:
            for token in line.split(" "):
                items = process_abbr_token(token)
                if items is not None:
                    abbr, sense, _ = items
                    dataset_counter[abbr].update([sense])

        # save DataSet Counter
        if save_collection_path is not None:
            pickle_writer(dataset_counter, save_collection_path)
        return dataset_counter

    def generate_inverted_index(self, offset_mapping=None):
        """
        Generate abbr inverted index and remove abbr marks.

        :return:
        """
        from baseline.word_embedding import AbbrIndex
        abbr_index = AbbrIndex()
        txt_post_processed = []
        global_instance_idx = 0
        offset_index = defaultdict(list)

        for doc_idx, doc in enumerate(self.corpus):
            doc_abbr = defaultdict(list)
            doc_processed = []
            tokens = doc.rstrip('\n').split(" ")
            for idx, token in enumerate(tokens):
                items = process_abbr_token(token)
                # if '|' in token:
                #     print(token, items)
                if items is not None:
                    abbr, sense, _ = items
                    # print(abbr, sense)
                    # add abbr info to inverted index
                    if offset_mapping:
                        offset = offset_mapping[idx][1]
                        offset_index[abbr].append(":".join([str(global_instance_idx), str(offset), sense]))
                        # print(abbr, ":".join([str(global_instance_idx), str(offset), sense]))
                        # print(offset_mapping[idx])
                    doc_abbr[abbr].append(":".join([str(global_instance_idx), str(idx), sense]))
                    doc_processed.append(abbr)
                    global_instance_idx += 1
                else:
                    doc_processed.append(token)
            txt_post_processed.append(" ".join(doc_processed))

            # convert doc_abbr dict to string
            for abbr, pos in doc_abbr.items():
                abbr_index.add_posting(abbr, doc_idx, pos)
                # print(abbr, doc_idx, pos)

        return abbr_index, txt_post_processed, offset_index

    def inverted_index_job(self, txt_list, offset_index, offset_mapping=None):
        """
        The txt_list should be chunked from self.corpus
        """
        txt_list_processed = []
        for doc_idx, doc in enumerate(txt_list):
            doc_abbr = defaultdict(list)
            doc_processed = []
            tokens = doc.rstrip('\n').split(" ")
            for idx, token in enumerate(tokens):
                items = process_abbr_token(token)
                if items is not None:
                    abbr, sense, _ = items
                    # add abbr info to inverted index
                    if offset_mapping:
                        offset = offset_mapping[idx][1]
                        offset_index[abbr].append(":".join([str(offset), sense]))
                        # print(offset_mapping[idx])
                    doc_abbr[abbr].append(":".join([str(idx), sense]))
                    doc_processed.append(abbr)
                else:
                    doc_processed.append(token)
            txt_list_processed.append(" ".join(doc_processed))
        return txt_list_processed, doc_abbr

    def generate_inverted_index_mp(self, pkl_path=None, offset_mapping=None, n_jobs=8):
        """
        Generate abbr inverted index and remove abbr marks.

        :return:
        """
        start_time = datetime.now()

        from baseline.word_embedding import AbbrIndex
        abbr_index = AbbrIndex()
        txt_post_processed = []
        global_instance_idx = 0
        offset_index = defaultdict(list)
        corpus_chunked = chunk(self.corpus, n_jobs)
        print(corpus_chunked)

        print('Generating inverted index...')
        corpus_processed = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(self.inverted_index_job)(c, offset_index, offset_mapping) for c in corpus_chunked)
        # for i in corpus_processed:
        #     print(i)

        # for doc_idx, doc in enumerate(self.corpus):
        #     doc_abbr = defaultdict(list)
        #     doc_processed = []
        #     tokens = doc.rstrip('\n').split(" ")
        #     for idx, token in enumerate(tokens):
        #         items = process_abbr_token(token)
        #         if items is not None:
        #             abbr, sense, _ = items
        #             # add abbr info to inverted index
        #             if offset_mapping:
        #                 offset = offset_mapping[idx][1]
        #                 offset_index[abbr].append(":".join([str(global_instance_idx), str(offset), sense]))
        #                 # print(offset_mapping[idx])
        #             doc_abbr[abbr].append(":".join([str(global_instance_idx), str(idx), sense]))
        #             doc_processed.append(abbr)
        #             global_instance_idx += 1
        #         else:
        #             doc_processed.append(token)
        #     txt_post_processed.append(" ".join(doc_processed))
        #
        #     # convert doc_abbr dict to string
        #     for abbr, pos in doc_abbr.items():
        #         abbr_index.add_posting(abbr, doc_idx, pos)
        #         print(abbr, doc_idx, pos)

        end_time = datetime.now()
        # if pkl_path:
        #     pkl = [abbr_index, txt_post_processed, offset_index]
        #     pickle_writer(pkl, pkl_path)
        print('Runtime:', str(end_time - start_time))
        # return abbr_index, txt_post_processed, offset_index
        return

###################################
# Comparision between Corpus
###################################

def get_cui_set(counter: dict):
    cui_set = []
    for _, cuis in counter.items():
        cui_set.extend(list(cuis))
    return set(cui_set)


def dataset_summary(counter: dict):
    abbrs = counter.keys()
    print("No.abbrs: ", len(abbrs))

    cui_set = get_cui_set(counter)
    print("No.CUIs: ", len(cui_set))

    # count number of instances in DataSet
    count_all_instances = 0
    for abbr, items in counter.items():
        for cui, count in items.items():
            count_all_instances += count
    print("No.instances", count_all_instances)
    print()
    return abbrs, cui_set, count_all_instances


def generate_sense_inventory_json_by_counter(counter: dict, json_path: str):
    counter_ordered = {}
    for key in counter:
        counter_ordered[key] = OrderedDict(counter[key].most_common())
    json_writer(counter_ordered, json_path)


def compare_dataset_summary(counter_train: dict, counter_test: dict):
    # collect CUIs
    cui_set_train = get_cui_set(counter_train)
    cui_set_test = get_cui_set(counter_test)
    print("No.CUIs on test: ", len(cui_set_test))
    print("CUI overlap ratio: ", len(cui_set_train & cui_set_test) / len(cui_set_test))

    # collect abbr info
    abbr_train = set(counter_train.keys())
    abbr_test = set(counter_test.keys())
    print("No. abbrs on test: ", len(abbr_test))
    print("Abbr overlap ratio: ", len(abbr_train & abbr_test) / len(abbr_test))
    print()


def compare_dataset_instances(counter_train: dict, counter_test: dict):
    count_all_instances = 0
    count_no_abbr_instances = 0
    count_overlap_instances = 0
    # out_file = open('/home/luoz3/wsd_data_test/PE_50000_NO_M/cui_not_overlap.txt', 'w')
    # out_file.close()
    for abbr, items in counter_test.items():
        for cui, count in items.items():
            count_all_instances += count
            if abbr not in counter_train:
                count_no_abbr_instances += count
            elif cui in counter_train[abbr]:
                count_overlap_instances += count
            # else:
            #     with open('/home/luoz3/wsd_data_test/PE_50000_NO_M/cui_not_overlap.txt', 'a') as file:
            #         file.write("%s | %s\n" % (abbr, cui))

    result_dict = {
        "overlap_ratio": count_overlap_instances / count_all_instances,
        "num_overlap": count_overlap_instances,
        "num_has_abbr_no_cui": count_all_instances - count_no_abbr_instances,
        "num_total": count_all_instances
    }
    return result_dict


def overlap_analysis(counter_train: dict, counter_test: dict):
    overlap_dict = defaultdict(dict)
    # out_file = open('/home/luoz3/wsd_data_test/PE_50000_NO_M/not_overlap.txt', 'w')
    # out_file.close()
    for abbr, items in counter_test.items():
        # rank for each cui in testset
        count_test_abbr_instances = 0
        test_cui_rank_dict = {}
        for idx, (cui, count) in enumerate(items.most_common()):
            count_test_abbr_instances += count
            test_cui_rank_dict[cui] = idx + 1

        if abbr in counter_train:
            # rank for each cui in trainset
            count_train_abbr_instances = 0
            train_cui_rank_dict = {}
            for idx, (cui, count) in enumerate(counter_train[abbr].most_common()):
                count_train_abbr_instances += count
                train_cui_rank_dict[cui] = idx + 1

            temp_abbr_dict = {}
            for cui, count in items.most_common():
                if cui in counter_train[abbr]:
                    temp_abbr_dict[cui] = {
                        "trainset sense rank": train_cui_rank_dict[cui],
                        "trainset sense count": counter_train[abbr][cui],
                        "trainset sense ratio": counter_train[abbr][cui] / count_train_abbr_instances,
                        "testset sense rank": test_cui_rank_dict[cui],
                        "testset sense count": count,
                        "testset sense ratio": count / count_test_abbr_instances
                    }
            overlap_dict[abbr] = temp_abbr_dict
        # else:
        #     print(items)
        #     with open('/home/luoz3/wsd_data_test/PE_50000_NO_M/not_overlap.txt', 'a') as out_file:
        #         out_file.write("%s | %s\n" % (abbr, items))
    return overlap_dict


Instance = namedtuple('Instance', ['index', 'abbr', 'sense', 'long_form'])
InstancePred = namedtuple('InstancePred', ['index', 'abbr', 'sense_pred'])


def save_instance_collection_to_json(instance_collection: list, json_path: str):
    with open(json_path, 'w') as file:
        for instance in instance_collection:
            file.write(json.dumps(instance._asdict()) + '\n')


def evaluation(instance_collection_true: list, instance_collection_pred: list, num_limited_samples=100):
    """
    Evaluate accuracy based on instance collections.

    :param instance_collection_true:
    :param instance_collection_pred:
    :param num_limited_samples: abbr with instances lower than it will count as abbr_with_limited_samples
    :return:
    """
    # print(len(instance_collection_true), len(instance_collection_pred))
    assert len(instance_collection_true) == len(instance_collection_pred)
    # save_instance_collection_to_json(instance_collection_true,
    #                                  '/home/luoz3/wsd_data_test/PE_50000_NO_M/instance_collection_true')
    # save_instance_collection_to_json(instance_collection_pred,
    #                                  '/home/luoz3/wsd_data_test/PE_50000_NO_M/instance_collection_pred')

    # Load train abbr index
    from baseline.word_embedding import AbbrIndex
    dataset_paths = DataSetPaths('luoz3_x1')
    # dataset_paths = DataSetPaths('docker') # For docker build

    # train_abbr_index = AbbrIndex(dataset_paths.mimic_train_folder + '/abbr_index_data.pkl')
    # train_abbr_index = AbbrIndex(dataset_paths.upmc_ag_train_folder + '/abbr_index_data.pkl')
    # train_abbr_index = AbbrIndex(dataset_paths.upmc_al_train_folder + '/abbr_index_data.pkl')
    train_abbr_index = AbbrIndex(dataset_paths.upmc_ao_train_folder + '/abbr_index_data.pkl')
    # train_abbr_index = AbbrIndex(dataset_paths.mimic_new_train_folder + '/abbr_index_data.pkl')

    count_correct, count_total, count_capable_total = 0.0, 0.0, 0.0
    count_by_abbr = {}
    count_by_abbr_only_limited_samples = {}
    # abbr_acc = open("/home/luoz3/wsd_data_test/pipeline_test/ao_result.txt", 'w+')

    for instance_true, instance_pred in zip(instance_collection_true, instance_collection_pred):
        # print(instance_true.sense, instance_pred.sense_pred)
        assert instance_true.index == instance_pred.index
        abbr = instance_true.abbr
        if abbr not in count_by_abbr:
            count_by_abbr[abbr] = [0, 0]
        if instance_true.sense == instance_pred.sense_pred:
            count_correct += 1.0
            count_by_abbr[abbr][0] += 1.0
        count_total += 1.0
        count_by_abbr[abbr][1] += 1.0
        # count the number of datapoints that model is able to predict
        if instance_pred.sense_pred:
            count_capable_total += 1.0

        if abbr not in train_abbr_index or train_abbr_index.num_instances(abbr) < num_limited_samples:
            if abbr not in count_by_abbr_only_limited_samples:
                count_by_abbr_only_limited_samples[abbr] = [0, 0]
            if instance_true.sense == instance_pred.sense_pred:
                count_by_abbr_only_limited_samples[abbr][0] += 1.0
            count_by_abbr_only_limited_samples[abbr][1] += 1.0

    if count_total > 0:
        acc = count_correct / count_total

        acc_by_abbr = []
        for abbr, (count_correct_by_abbr, count_total_by_abbr) in count_by_abbr.items():
            acc = count_correct_by_abbr / count_total_by_abbr
            acc_by_abbr.append(acc)

            # # Write the whole acc to a file
            # l = abbr + ': ' + str(acc) + ' [' + str(count_correct_by_abbr) + '/' + str(count_total_by_abbr) + ']'
            # print(l)
            # abbr_acc.write(l + '\n')
        acc_by_abbr = sum(acc_by_abbr) / len(acc_by_abbr)

        acc_by_abbr_only_limited_samples = []
        for abbr, (count_correct_by_abbr, count_total_by_abbr) in count_by_abbr_only_limited_samples.items():
            acc_by_abbr_only_limited_samples.append(count_correct_by_abbr / count_total_by_abbr)
        num_abbr_only_limited_samples = len(acc_by_abbr_only_limited_samples)
        acc_by_abbr_only_limited_samples = sum(acc_by_abbr_only_limited_samples) / num_abbr_only_limited_samples
    else:
        acc = 0.0
        acc_by_abbr = 0.0
        acc_by_abbr_only_limited_samples = 0.0
        num_abbr_only_limited_samples = 0

    if count_capable_total > 0:
        acc_capable = count_correct / count_capable_total
    else:
        acc_capable = 0.0

    score_dict = {
        'accuracy': acc,
        'accuracy_capable': acc_capable,
        'accuracy_by_abbr': acc_by_abbr,
        'accuracy_by_abbr_only_limited_samples': acc_by_abbr_only_limited_samples,
        'num_correct': count_correct,
        'num_total': count_total,
        'num_capable_total': count_capable_total,
        'num_abbr_only_limited_samples': num_abbr_only_limited_samples
    }
    return score_dict


if __name__ == '__main__':
    # dataset_paths = DataSetPaths('luoz3_x1')
    dataset_paths = DataSetPaths('xil222')
    # train_counter_path = dataset_paths.mimic_train_folder+'train_abbr_counter.pkl'

    #############################
    # build train collectors
    #############################
    # mimic_train_collector = AbbrInstanceCollector(dataset_paths.mimic_train_txt)
    # mimic_train_counter = mimic_train_collector.generate_counter(train_counter_path)

    # upmc_ad_train_collector = AbbrInstanceCollector(dataset_paths.upmc_ad_train_txt)
    # upmc_ad_train_counter = upmc_ad_train_collector.generate_counter()
    # upmc_ag_train_collector = AbbrInstanceCollector(dataset_paths.upmc_ag_train_txt)
    # upmc_ag_train_counter = upmc_ag_train_collector.generate_counter()
    # upmc_al_train_collector = AbbrInstanceCollector(dataset_paths.upmc_al_train_txt)
    # upmc_al_train_counter = upmc_al_train_collector.generate_counter()
    upmc_ao_train_collector = AbbrInstanceCollector(dataset_paths.upmc_ao_train_txt)
    upmc_ao_train_counter = upmc_ao_train_collector.generate_counter()

    # pe_self_train_collector = AbbrInstanceCollector(dataset_paths.pe_self_train_txt)
    # pe_self_train_counter = pe_self_train_collector.generate_counter()

    # # read train counter from file mimic_train_counter = pickle_reader(train_counter_path)
    # generate_sense_inventory_json_by_counter(mimic_train_counter,
    # dataset_paths.mimic_train_folder+'mimic_train_inventory.json') upmc_ab_train_counter = pickle_reader(
    # dataset_paths.upmc_ab_train_folder + "/train_abbr_counter.pkl")

    #############################
    # summary of training set
    #############################
    # print("Summary of MIMIC train:")
    # dataset_summary(mimic_train_counter)
    # print("Summary of UPMC AB train:")
    # dataset_summary(upmc_ab_train_counter)
    # print("Summary of UPMC AD train:")
    # dataset_summary(upmc_ad_train_counter)
    # print("Summary of UPMC AG train:")
    # dataset_summary(upmc_ag_train_counter)
    # print("Summary of UPMC AL train:")
    # dataset_summary(upmc_al_train_counter)
    print("Summary of UPMC AO train:")
    dataset_summary(upmc_ao_train_counter)
    # print("Summary of PE train:")
    # dataset_summary(pe_self_train_counter)

    #############################
    # build test collectors
    #############################
    # mimic_test_collector = AbbrInstanceCollector(dataset_paths.mimic_eval_txt)
    # share_collector = AbbrInstanceCollector(dataset_paths.share_txt)
    # msh_collector = AbbrInstanceCollector(dataset_paths.msh_txt)
    # umn_collector = AbbrInstanceCollector(dataset_paths.umn_txt)
    # upmc_example_collector = AbbrInstanceCollector(dataset_paths.upmc_example_txt)
    # upmc_ab_test_collector = AbbrInstanceCollector(dataset_paths.upmc_ab_test_txt)
    # upmc_ad_test_collector = AbbrInstanceCollector(dataset_paths.upmc_ad_test_txt)
    # upmc_ag_test_collector = AbbrInstanceCollector(dataset_paths.upmc_ag_test_txt)
    # upmc_al_test_collector = AbbrInstanceCollector(dataset_paths.upmc_al_test_txt)
    upmc_ao_test_collector = AbbrInstanceCollector(dataset_paths.upmc_ao_test_txt)
    # pe_test_collector = AbbrInstanceCollector(dataset_paths.pe_test_txt)
    # pe_self_test_collector = AbbrInstanceCollector(dataset_paths.pe_self_test_txt)
    # pe_50000_nm_test_collector = AbbrInstanceCollector(dataset_paths.pe_50000_nm_test_txt)
    # ipdc_50000_test_collector = AbbrInstanceCollector(dataset_paths.ipdc_50000_test_txt)

    #############################
    # generate test counters
    #############################
    # mimic_test_counter = mimic_test_collector.generate_counter()
    # share_counter = share_collector.generate_counter()
    # msh_counter = msh_collector.generate_counter()
    # umn_counter = umn_collector.generate_counter()
    # upmc_example_counter = upmc_example_collector.generate_counter()
    # upmc_ab_test_counter = upmc_ab_test_collector.generate_counter()
    # upmc_ad_test_counter = upmc_ad_test_collector.generate_counter()
    # upmc_ag_test_counter = upmc_ag_test_collector.generate_counter()
    upmc_ao_test_counter = upmc_ao_test_collector.generate_counter()
    # pe_50000_nm_test_counter = pe_50000_nm_test_collector.generate_counter()
    # pe_self_test_counter = pe_self_test_collector.generate_counter()
    # ipdc_50000_test_counter = ipdc_50000_test_collector.generate_counter()

    #############################
    # summary of testing set
    #############################
    print('Summary of UPMC AO test:')
    dataset_summary(upmc_ao_test_counter)
    # print("Summary of PE test:")
    # dataset_summary(pe_50000_nm_test_counter)
    # print("Summary of PE test:")
    # dataset_summary(pe_self_test_counter)
    # print('Summary of IPDC test')
    # dataset_summary(ipdc_50000_test_counter)

    # # generate sense inventories generate_sense_inventory_json_by_counter(mimic_test_counter,
    # dataset_paths.mimic_test_folder + 'mimic_test_inventory.json') generate_sense_inventory_json_by_counter(
    # share_counter, dataset_paths.share_test_folder + 'share_inventory.json')
    # generate_sense_inventory_json_by_counter(msh_counter, dataset_paths.msh_test_folder + 'msh_inventory.json')
    # generate_sense_inventory_json_by_counter(umn_counter, dataset_paths.umn_test_folder + 'umn_inventory.json')
    # generate_sense_inventory_json_by_counter(upmc_example_counter, dataset_paths.upmc_example_folder +
    # 'upmc_example_inventory.json')
    # generate_sense_inventory_json_by_counter(upmc_ad_train_counter,
    #                                          dataset_paths.upmc_ad_train_folder + "upmc_ad_train_inventory.json")
    # generate_sense_inventory_json_by_counter(upmc_ad_test_counter,
    #                                          dataset_paths.upmc_ad_test_folder + "upmc_ad_test_inventory.json")
    # generate_sense_inventory_json_by_counter(upmc_ag_train_counter,
    #                                          dataset_paths.upmc_ag_train_folder + "upmc_ag_train_inventory.json")
    # generate_sense_inventory_json_by_counter(upmc_ag_test_counter,
    #                                          dataset_paths.upmc_ag_test_folder + "upmc_ag_test_inventory.json")
    # generate_sense_inventory_json_by_counter(upmc_al_train_counter,
    #                                          dataset_paths.upmc_al_train_folder + "upmc_al_train_inventory.json")
    # generate_sense_inventory_json_by_counter(pe_50000_nm_test_counter,
    #                                          dataset_paths.pe_50000_nm_test_folder + "pe_test_inventory.json")
    # generate_sense_inventory_json_by_counter(ipdc_50000_test_counter,
    #                                          dataset_paths.ipdc_50000_test_folder + "ipdc_test_inventory.json")

    ##################################
    # compare dataset intersections
    ##################################
    # print("Intersection on MIMIC test: ")
    # compare_dataset_summary(mimic_train_counter, mimic_test_counter)
    # print("Intersection on share: ")
    # compare_dataset_summary(mimic_train_counter, share_counter)
    # print("Intersection on msh: ")
    # compare_dataset_summary(mimic_train_counter, msh_counter)
    # print("Intersection on umn: ")
    # compare_dataset_summary(mimic_train_counter, umn_counter)
    # print("Intersection on upmc example: ")
    # compare_dataset_summary(mimic_train_counter, upmc_example_counter)
    # print("Intersection on UPMC AB train: ")
    # compare_dataset_summary(mimic_train_counter, upmc_ab_train_counter)
    # print("Intersection on UPMC AB test: ")
    # compare_dataset_summary(mimic_train_counter, upmc_ab_test_counter)
    # print("Intersection on UPMC AB test (UPMC AB train): ")
    # compare_dataset_summary(upmc_ab_train_counter, upmc_ab_test_counter)
    # print("Intersection on UPMC AD test (UPMC AD train): ")
    # compare_dataset_summary(upmc_ad_train_counter, upmc_ad_test_counter)
    # print("Intersection on UPMC AG test (UPMC AG train): ")
    # compare_dataset_summary(upmc_ag_train_counter, upmc_ag_test_counter)
    print("Intersection on UPMC AO test (UPMC AO train): ")
    compare_dataset_summary(upmc_ao_train_counter, upmc_ao_test_counter)

    # print("Intersection on PE test (UPMC AL train): ")
    # compare_dataset_summary(upmc_al_train_counter, pe_50000_nm_test_counter)
    # print("Intersection on PE test (PE train): ")
    # compare_dataset_summary(pe_self_train_counter, pe_self_test_counter)
    # print("Intersection on IPDC test (UPMC AL train): ")
    # compare_dataset_summary(upmc_al_train_counter, ipdc_50000_test_counter)

    # # compare mapping instances
    # print("Compare instances on MIMIC test:")
    # print(compare_dataset_instances(mimic_train_counter, mimic_test_counter))
    # print("Compare instances on ShARe/CLEF:")
    # print(compare_dataset_instances(mimic_train_counter, share_counter))
    # print("Compare instances on MSH:")
    # print(compare_dataset_instances(mimic_train_counter, msh_counter))
    # print("Compare instances on UMN:")
    # print(compare_dataset_instances(mimic_train_counter, umn_counter))
    # print("Compare instances on UPMC example:")
    # print(compare_dataset_instances(mimic_train_counter, upmc_example_counter))
    # print("Compare instances on UPMC AB train:")
    # print(compare_dataset_instances(mimic_train_counter, upmc_ab_train_counter))
    # print("Compare instances on UPMC AB test:")
    # print(compare_dataset_instances(mimic_train_counter, upmc_ab_test_counter))
    # print("Compare instances on UPMC AB test (UPMC AB train):")
    # print(compare_dataset_instances(upmc_ab_train_counter, upmc_ab_test_counter))
    # print("Compare instances on UPMC AD test (UPMC AD train):")
    # print(compare_dataset_instances(upmc_ad_train_counter, upmc_ad_test_counter))
    # print("Compare instances on UPMC AG test (UPMC AG train):")
    # print(compare_dataset_instances(upmc_ag_train_counter, upmc_ag_test_counter))
    # print("Compare instances on PE test (UPMC AL train):")
    # print(compare_dataset_instances(upmc_al_train_counter, pe_50000_nm_test_counter))
    # print("Compare instances on PE test (PE train):")
    # print(compare_dataset_instances(pe_self_train_counter, pe_self_test_counter))
    # print("Compare instances on IPDC test (UPMC AL train):")
    # print(compare_dataset_instances(upmc_al_train_counter, ipdc_50000_test_counter))

    # upmc_overlap = overlap_analysis(mimic_train_counter, upmc_example_counter)
    # overlap = overlap_analysis(upmc_al_train_counter, pe_50000_nm_test_counter)
    # json_writer(upmc_overlap, dataset_paths.upmc_example_folder+"/upmc_overlap.json")
    # print()
