"""
Helper functions for MIMIC-III dataset.

"""
import json
import multiprocessing as mp
import operator
import os
import pandas as pd
import pickle
import re
import string
import tqdm
from datetime import datetime
from joblib import Parallel, delayed
from preprocess.text_helper import sub_patterns, white_space_remover, repeat_non_word_remover, recover_upper_cui
from preprocess.text_helper import TextProcessor, CoreNLPTokenizer, TextTokenFilter
from preprocess.file_helper import txt_reader, txt_writer, json_writer, pickle_reader


# DeID replacement for MIMIC (ShARe/CLEF)
def sub_deid_patterns_mimic(txt):
    # DATE
    txt = sub_patterns(txt, [
        # normal date
        r"\[\*\*(\d{4}-)?\d{1,2}-\d{1,2}\*\*\]",
        # date range
        r"\[\*\*Date [rR]ange.+?\*\*\]",
        # month/year
        r"\[\*\*-?\d{1,2}-?/\d{4}\*\*\]",
        # year
        r"\[\*\*(Year \([24] digits\).+?)\*\*\]",
        # holiday
        r"\[\*\*Holiday.+?\*\*\]",
        # XXX-XX-XX
        r"\[\*\*\d{3}-\d{1,2}-\d{1,2}\*\*\]",
        # date with format
        r"\[\*\*(Month(/Day)?(/Year)?|Year(/Month)?(/Day)?|Day Month).+?\*\*\]",
        # uppercase month year
        r"\[\*\*(January|February|March|April|May|June|July|August|September|October|November|December).+?\*\*\]",
    ], "DATE-DEID")

    # NAME
    txt = sub_patterns(txt, [
        # name
        r"\[\*\*(First |Last )?Name.+?\*\*\]",
        # name initials
        r"\[\*\*Initial.+?\*\*\]",
        # name with sex
        r"\[\*\*(Female|Male).+?\*\*\]",
        # doctor name
        r"\[\*\*Doctor.+?\*\*\]",
        # known name
        r"\[\*\*Known.+?\*\*\]",
        # wardname
        r"\[\*\*Wardname.+?\*\*\]",
    ], "NAME-DEID")

    # INSTITUTION
    txt = sub_patterns(txt, [
        # hospital
        r"\[\*\*Hospital.+?\*\*\]",
        # university
        r"\[\*\*University.+?\*\*\]",
        # company
        r"\[\*\*Company.+?\*\*\]",
    ], "INSTITUTION-DEID")

    # clip number
    txt = sub_patterns(txt, [
        r"\[\*\*Clip Number.+?\*\*\]",
    ], "CLIP-NUMBER-DEID")

    # digits
    txt = sub_patterns(txt, [
        r"\[\*\* ?\d{1,5}\*\*\]",
    ], "DIGITS-DEID")

    # tel/fax
    txt = sub_patterns(txt, [
        r"\[\*\*Telephone/Fax.+?\*\*\]",
        r"\[\*\*\*\*\]",
    ], "PHONE-DEID")

    # EMPTY
    txt = sub_patterns(txt, [
        r"\[\*\* ?\*\*\]",
    ], "EMPTY-DEID")

    # numeric identifier
    txt = sub_patterns(txt, [
        r"\[\*\*Numeric Identifier.+?\*\*\]",
    ], "NUMERIC-DEID")

    # AGE
    txt = sub_patterns(txt, [
        r"\[\*\*Age.+?\*\*\]",
    ], "AGE-DEID")

    # PLACE
    txt = sub_patterns(txt, [
        # country
        r"\[\*\*Country.+?\*\*\]",
        # state
        r"\[\*\*State.+?\*\*\]",
    ], "PLACE-DEID")

    # STREET-ADDRESS
    txt = sub_patterns(txt, [
        r"\[\*\*Location.+?\*\*\]",
        r"\[\*\*.+? Address.+?\*\*\]",
    ], "STREET-ADDRESS-DEID")

    # MD number
    txt = sub_patterns(txt, [
        r"\[\*\*MD Number.+?\*\*\]",
    ], "MD-NUMBER-DEID")

    # other numbers
    txt = sub_patterns(txt, [
        # job
        r"\[\*\*Job Number.+?\*\*\]",
        # medical record number
        r"\[\*\*Medical Record Number.+?\*\*\]",
        # SSN
        r"\[\*\*Social Security Number.+?\*\*\]",
        # unit number
        r"\[\*\*Unit Number.+?\*\*\]",
        # pager number
        r"\[\*\*Pager number.+?\*\*\]",
        # serial number
        r"\[\*\*Serial Number.+?\*\*\]",
        # provider number
        r"\[\*\*Provider Number.+?\*\*\]",
    ], "OTHER-NUMBER-DEID")

    # info
    txt = sub_patterns(txt, [
        r"\[\*\*.+?Info.+?\*\*\]",
    ], "INFO-DEID")

    # E-mail
    txt = sub_patterns(txt, [
        r"\[\*\*E-mail address.+?\*\*\]",
        r"\[\*\*URL.+?\*\*\]"
    ], "EMAIL-DEID")

    # other
    txt = sub_patterns(txt, [
        r"\[\*\*(.*)?\*\*\]",
    ], "OTHER-DEID")
    return txt


def split_non_valid_cui(txt):
    txt = re.sub(r"(abbr\|\w+\|C?\d+)([\\/])(?=[^ ]+?)", r"\1 \2 ", txt)
    return txt


# def longform_replacer_job(file_list_sub, mapper):
#     """
#     Find a longform in the text, replace it to the target format abbr|
#     :param file_list_sub: Sub-list of doc files
#     :param mapper: mapper[longform][0] is abbr, mapper[longform][1] is CUI
#     :return:
#     """
#     # PATH_FOLDER = '/data/Patient_Education/'
#
#     for file_name in file_list_sub:
#         print("-" * 50)
#         print("Start File for %s" % file_name)
#
#         file_content = ''
#         with open(PATH_FOLDER+file_name, 'r') as file:
#             for l in file:
#                 file_content += l
#         for longform in mapper:
#             if longform in file_content:
#                 abbr = 'abbr|' + mapper[longform][0] + '|' + mapper[longform][1]
#                 file_content.replace(longform, abbr)


def longform_replacer_job(file_list_sub, text_list_sub, mapper):
    for i in range(len(text_list_sub)):
        # print("-" * 50)
        # print("Start File for %s" % file_list_sub[i])

        for longform in mapper:
            if longform in text_list_sub[i]:
                abbr = 'abbr|' + mapper[longform][0] + '|' + mapper[longform][1] + ' '
                # print(longform, abbr)
                text_list_sub[i] = text_list_sub[i].replace(longform, abbr)

    return text_list_sub


def longform_replacer(file_list, text_list, mapper, n_jobs=16):
    print("Replacing long forms...")
    file_list_chunked = chunk(file_list, n_jobs)
    text_list_chunked = chunk(text_list, n_jobs)

    text_list_processed = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(longform_replacer_job)(file_list_sub, text_list_sub, mapper)
        for file_list_sub, text_list_sub in zip(file_list_chunked, text_list_chunked)
    )
    # print(text_list_processed)

    result = [item for sublist in text_list_processed for item in sublist]
    return result


def chunk(txt_list, n):
    result = []
    for i in range(0, len(txt_list), n):
        result.append(txt_list[i:i+n])
    return result


if __name__ == '__main__':

    # PATH_FOLDER = '/data/Patient_Education/'
    # PATH_FOLDER_PROCESSED = '/home/luoz3/wsd_data_test/Patient_Education_Processed/'
    PATH_FOLDER = '/data/IPDCSummary/'
    PATH_FOLDER_PROCESSED = '/home/luoz3/wsd_data_test/IPDCSummary_Processed/'

    if not os.path.exists(PATH_FOLDER_PROCESSED):
        os.makedirs(PATH_FOLDER_PROCESSED)

    ######################################
    # Generate Pickle Dictionary
    ######################################

    # sense_file = pd.read_csv('/home/luoz3/abbr_sense_new_no_m.csv', na_filter=False)
    # sense_mapper = {}
    # for i, row in sense_file.iterrows():
    #     print(row.abbr + ', ' + row.sense_id)
    #     sense_mapper[row.sense] = (row.abbr, row.sense_id)
    # pickle.dump(sense_mapper, open('/home/luoz3/abbr_sense_mapper.pkl', 'wb'))

    ######################################
    # Processing
    ######################################

    file_list = os.listdir(PATH_FOLDER)
    file_list.sort()
    file_list = file_list[:50000]

    # 460993 files in total in Patient Education
    # 311232 files in total in IPDC Summary
    mapper = pickle.load(open('/home/luoz3/abbr_sense_mapper.pkl', 'rb'))
    # menu_file_name = 'PE_menu_50000_no_m.txt'
    # processed_file_name = 'PE_processed_50000_no_m.txt'
    menu_file_name = 'ipdc_menu_50000.txt'
    processed_file_name = 'ipdc_processed_50000.txt'

    # Write the doc list
    with open(PATH_FOLDER_PROCESSED+menu_file_name, 'w') as mf:
        for item in file_list:
            mf.write("%s\n" % item)

    # Initialize processor and tokenizer
    processor = TextProcessor([
        white_space_remover])

    tokenizer = CoreNLPTokenizer()

    token_filter = TextTokenFilter()

    filter_processor = TextProcessor([
        token_filter,
        repeat_non_word_remover,
        recover_upper_cui])

    all_processor = TextProcessor([
        white_space_remover,
        token_filter,
        repeat_non_word_remover,
        recover_upper_cui])

    start_time = datetime.now()
    upmc_txt_list = []
    for file_name in file_list:
        with open(PATH_FOLDER+file_name, 'r') as file:
            content = file.read()
            upmc_txt_list.append(content)
    print('Loading text list finished in ' + str(datetime.now() - start_time))

    # pre-processing
    start_time = datetime.now()
    upmc_txt = processor.process_texts(upmc_txt_list, n_jobs=30)
    print('Pre-processing finished in ' + str(datetime.now()-start_time))
    # print(len(upmc_txt))

    # Replace Long forms to abbreviations
    start_time = datetime.now()
    upmc_txt_processed = longform_replacer(file_list, upmc_txt, mapper, n_jobs=16)
    print('Long-form replacement finished in ' + str(datetime.now() - start_time))
    # print(len(upmc_txt_processed))

    # tokenizing
    # start_time = datetime.now()
    # upmc_txt_tokenized = tokenizer.process_texts(upmc_txt_processed, n_jobs=40)
    # print('Tokenization finished in ' + str(datetime.now() - start_time))

    # Filter trivial tokens
    # start_time = datetime.now()
    # upmc_txt_filtered = filter_processor.process_texts(upmc_txt_tokenized, n_jobs=40)
    # print('Filtering trivial tokens finished in ' + str(datetime.now() - start_time))

    # Save to file
    txt_writer(upmc_txt_processed, PATH_FOLDER_PROCESSED + processed_file_name)
