"""
Processing UPMC data.
"""
import datetime
import os
import re
import tqdm
import random
import operator
import multiprocessing as mp
from preprocess.text_helper import sub_patterns, white_space_remover, repeat_non_word_remover, recover_upper_cui
from preprocess.text_helper import TextProcessor, CoreNLPTokenizer, TextTokenFilter
from preprocess.file_helper import txt_reader, txt_writer, json_writer
from preprocess.dataset.Tokenizer import Tokenizer


def replace(token: str):
    """
    Fix annotation error
    :param token:
    :return:
    """
    if token.startswith("abbr|"):
        segments = token.split("|")
        middle = int((len(segments) - 1) / 2)
        return segments[0] + "|" + segments[middle] + "|" + segments[len(segments) - 1]
    else:
        return token


def process_annotated_data(txt_preprocessed_path, upmc_processed_path, train_ratio=0.8, n_jobs=30):
    os.makedirs(upmc_processed_path, exist_ok=True)
    upmc_txt_annotated = txt_reader(txt_preprocessed_path)

    #################
    # pre-processing
    #################
    # Process white spaces
    upmc_txt_no_space = []
    for txt in upmc_txt_annotated:
        txt_ns = white_space_remover(txt)
        upmc_txt_no_space.append(txt_ns)
        # print(txt_ns)

    # Tokenize
    txt_tokenized = []
    for txt in upmc_txt_annotated:
        txt_t = tokenizer.get_tokens(txt)
        txt_tokenized.append(txt_t)
    # upmc_txt = all_processor.process_texts(upmc_txt_annotated, n_jobs=n_jobs)
    # print(txt_tokenized)

    # Convert back
    upmc_txt = all_processor.process_texts(txt_tokenized, n_jobs=n_jobs)
    upmc_txt_pure = []
    for txt in upmc_txt:
        txt_pure = ' '.join([token[0] for token in txt])
        upmc_txt_pure.append(txt_pure)

    upmc_txt = upmc_txt_pure

    ###############################
    # train/test split (80% train)
    ###############################
    random.shuffle(upmc_txt)
    num_instances = len(upmc_txt)
    train_idx = set(random.sample(range(num_instances), int(train_ratio*num_instances)))
    upmc_train_txt = []
    upmc_test_txt = []
    for idx, txt in enumerate(tqdm.tqdm(upmc_txt)):
        if idx in train_idx:
            upmc_train_txt.append(txt)
        else:
            upmc_test_txt.append(txt)

    # Variable type checking
    print()
    print('upmc_train_txt:', type(upmc_train_txt))
    # print('upmc_train_txt[0]:', type(upmc_train_txt[0]))
    print('upmc_test_txt', type(upmc_test_txt))
    print()

    # Write to file
    txt_writer(upmc_train_txt, upmc_processed_path+"/upmc_train.txt")
    txt_writer(upmc_test_txt, upmc_processed_path+"/upmc_test.txt")


if __name__ == '__main__':

    ######################################
    # Read texts from dataset
    ######################################

    # File paths
    # data_path = "/home/luoz3/wsd_data"
    data_path = "/home/luoz3/wsd_data_test"
    # dataset_path = data_path + "/upmc/example"
    # dataset_processed_path = data_path + "/upmc/example/processed"
    # os.makedirs(dataset_processed_path, exist_ok=True)

    # # fix annotation error
    # with open(dataset_path + "/training_data.txt") as input, open(dataset_path + "/training_data_fixed.txt",
    #                                                               "w") as output:
    #     for line in input:
    #         new_line = " ".join([replace(token) for token in line.rstrip("\n").split(" ")])
    #         output.write(new_line + "\n")

    #############################
    # Process DataSet documents (only one word abbrs)
    #############################

    # dataset_txt_annotated = txt_reader(dataset_path + "/training_data_fixed.txt")

    # Initialize processor and tokenizer
    processor = TextProcessor([
        white_space_remover])

    # toknizer = CoreNLPTokenizer()
    tokenizer = Tokenizer()

    token_filter = TextTokenFilter()

    filter_processor = TextProcessor([
        token_filter,
        repeat_non_word_remover,
        recover_upper_cui])

    all_processor = TextProcessor([
        # white_space_remover,
        token_filter,
        repeat_non_word_remover,
        recover_upper_cui])

    # # pre-processing
    # dataset_txt = processor.process_texts(dataset_txt_annotated, n_jobs=30)
    # # tokenizing
    # dataset_txt_tokenized = toknizer.process_texts(dataset_txt, n_jobs=30)
    # # Filter trivial tokens and Remove repeat non-words
    # dataset_txt_filtered = filter_processor.process_texts(dataset_txt_tokenized, n_jobs=30)
    # # Write to file
    # txt_writer(dataset_txt_filtered, dataset_processed_path+"/upmc_example_processed.txt")

    # ######################################
    # # Processing UPMC AB
    # ######################################

    # upmc_ab_path = data_path + "/upmc/AB"
    # upmc_ab_processed_path = upmc_ab_path + "/processed"
    # os.makedirs(upmc_ab_processed_path, exist_ok=True)
    #
    # upmc_ab_txt_annotated = txt_reader(upmc_ab_path + "/training_data_AB.txt")
    # # pre-processing
    # upmc_ab_txt = processor.process_texts(upmc_ab_txt_annotated, n_jobs=30)
    # # tokenizing
    # upmc_ab_txt_tokenized = toknizer.process_texts(upmc_ab_txt, n_jobs=30)
    # # Filter trivial tokens and Remove repeat non-words
    # upmc_ab_txt_filtered = filter_processor.process_texts(upmc_ab_txt_tokenized, n_jobs=30)
    #
    # # train/test split (80% train)
    # random.shuffle(upmc_ab_txt_filtered)
    # num_instances = len(upmc_ab_txt_filtered)
    # train_idx = random.sample(range(num_instances), int(0.8*num_instances))
    # upmc_ab_train_txt = []
    # upmc_ab_test_txt = []
    # for idx, txt in enumerate(upmc_ab_txt_filtered):
    #     if idx in train_idx:
    #         upmc_ab_train_txt.append(txt)
    #     else:
    #         upmc_ab_test_txt.append(txt)
    # # Write to file
    # txt_writer(upmc_ab_train_txt, upmc_ab_processed_path+"/upmc_ab_train.txt")
    # txt_writer(upmc_ab_test_txt, upmc_ab_processed_path + "/upmc_ab_test.txt")

    ######################################
    # Processing UPMC AD
    ######################################

    # process_annotated_data("/home/wangz12/scripts/generate_trainning_data/training_data_AD.txt",
    #                        data_path + "/upmc/AD/processed")

    ######################################
    # Processing UPMC AG
    ######################################

    # process_annotated_data("/home/wangz12/scripts/generate_trainning_data/training_data_AG.txt",
    #                        data_path + "/upmc/AG/processed")

    ######################################
    # Processing UPMC AL
    ######################################

    # process_annotated_data("/home/wangz12/scripts/generate_trainning_data/training_data_AL.txt",
    #                        data_path + "/upmc/AL/processed")

    ######################################
    # Processing UPMC AO
    ######################################

    # process_annotated_data("/home/wangz12/scripts/generate_trainning_data/training_data_AO.txt",
    #                        data_path + "/upmc/AO/processed")

    # process_annotated_data("/home/luoz3/wsd_clone/preprocess/HPI_test.txt",
    #                        data_path + "/pipeline_test/processed",
    #                        train_ratio=0)

    process_annotated_data(
        "/home/wangz12/scripts/generate_trainning_data/training_data_all.txt",
        data_path + "/training_all/processed"
    )

    # process_annotated_data(
    #     '/home/luoz3/wsd_data_test/Patient_Education_Processed/PE_processed.txt',
    #     data_path+'/PE_SELF/processed')
    # process_annotated_data(
    #     '/home/luoz3/wsd_data_test/IPDCSummary_Processed/ipdc_processed_50000.txt',
    #     data_path+'/IPDC_50000/processed',
    #     train_ratio=0)
    print()
