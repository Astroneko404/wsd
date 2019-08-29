"""
Acronyms/Abbreviation (only consider one word) disambiguation pipeline for UPMC.
"""
import os
import re
import tqdm
import json
import pickle
import multiprocessing as mp
from collections import defaultdict, OrderedDict
from fastText import load_model
from preprocess.file_helper import txt_reader
from preprocess.text_helper import repeat_non_word_remover, recover_upper_cui
from preprocess.text_helper import TextProcessor, CoreNLPTokenizer, TextTokenFilter
from baseline.dataset_helper import AbbrInstanceCollector, DataSetPaths, process_abbr_token
from baseline.generate_content import AbbrCorpus, Doc
from preprocess.dataset.Tokenizer import Tokenizer
from joblib import Parallel, delayed
from datetime import datetime


ALL_DEID_PATTERN = "\*\*DATE\b(\[.*?\])?|" \
                   "\*\*NAME\b(\[.*?\])?|" \
                   "\*\*PLACE\b(\[.*?\])?|" \
                   "\*\*INSTITUTION\b(\[.*?\])?|" \
                   "\*\*ID-NUM[A-Z]*\b(\[.*?\])?|" \
                   "\*\*ZIP-CODE\b(\[.*?\])?|" \
                   "\*\*PHONE-?\b(\[.*?\])?|" \
                   "\*\*STREET-ADDRESS\b(\[.*?\])?|" \
                   "\*\*AGE\b(\[.*?\])?|" \
                   "\*\*INITIALS[A-Z]*\b(\[.*?\])?|" \
                   "\*\*ROOM\b(\[.*?\])?|" \
                   "\*\*EMAIL\b(\[.*?\])?|" \
                   "\*\*PATH-NUMBER\b(\[.*?\])?|" \
                   "\*\*WEB-LOC\b(\[.*?\])?|" \
                   "\*\*ACCESSION-NUMBER\b(\[.*?\])?|" \
                   "\*\*ID-\b(\[.*?\])?|"\
                   "\*\*DEVICE-ID\b(\[.*?\])?"


def some(array, fn):
    for item in array:
        if fn(item):
            return True
    return False


def white_space_remover_upmc(txt):
    """
    Remove '\n' and redundant spaces.

    :param txt: Single Document String
    :return: Processed Document String
    """
    # remove all "\n"
    txt = re.sub(r"\n", "\u21B5", txt)
    # remove all redundant spaces
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt


# # DeID replacement for UPMC
def sub_deid_patterns_upmc(txt):
    txt = re.sub(r"\*\*DATE\b(\[.*?\])?", "DATE-DEID", txt)
    txt = re.sub(r"\*\*NAME\b(\[.*?\])?", "NAME-DEID", txt)
    txt = re.sub(r"\*\*PLACE\b(\[.*?\])?", "PLACE-DEID", txt)
    txt = re.sub(r"\*\*INSTITUTION\b(\[.*?\])?", "INSTITUTION-DEID", txt)
    txt = re.sub(r"\*\*ID-NUM[A-Z]*\b(\[.*?\])?", "ID-NUM-DEID", txt)
    txt = re.sub(r"\*\*ZIP-CODE\b(\[.*?\])?", "ZIP-CODE-DEID", txt)
    txt = re.sub(r"\*\*PHONE-?\b(\[.*?\])?", "PHONE-DEID", txt)
    txt = re.sub(r"\*\*STREET-ADDRESS\b(\[.*?\])?", "STREET-ADDRESS-DEID", txt)
    txt = re.sub(r"\*\*AGE\b(\[.*?\])?", "AGE-DEID", txt)
    txt = re.sub(r"\*\*INITIALS[A-Z]*\b(\[.*?\])?", "INITIALS-DEID", txt)
    txt = re.sub(r"\*\*ROOM\b(\[.*?\])?", "ROOM-DEID", txt)
    txt = re.sub(r"\*\*EMAIL\b(\[.*?\])?", "EMAIL-DEID", txt)
    txt = re.sub(r"\*\*PATH-NUMBER\b(\[.*?\])?", "PATH-NUMBER-DEID", txt)
    txt = re.sub(r"\*\*WEB-LOC\b(\[.*?\])?", "WEB-LOC-DEID", txt)
    txt = re.sub(r"\*\*ACCESSION-NUMBER\b(\[.*?\])?", "ACCESSION-NUMBER-DEID", txt)
    txt = re.sub(r"\*\*ID-\b(\[.*?\])?", "ID-DEID", txt)
    txt = re.sub(r"\*\*DEVICE-ID\b(\[.*?\])?", "DEVICE-ID-DEID", txt)
    return txt


class AbbrDetector:
    """Detect Abbrs
    :return: content and abbrs after processed
    """

    def __init__(self, abbr_inventory_path):
        self.abbr_inventory = pickle.load(open(abbr_inventory_path, "rb"))
        # Patterns for Abbr detection
        self.abbr_patterns = [
            re.compile(r"[A-Z\-_0-9#]+")
        ]
        self.black_list = {
            "-",
            "Dr.",
            "Mr.",
            "Ms.",
            "vs.",
            "a.m.",
            "p.m.",
        }
        self.black_pattern_list = [
            re.compile(r"[0-9a-z]+"),
            re.compile(r"[A-Z][a-z]*"),
            # DeID strings
            re.compile(r"[A-Z-]+-DEID"),
            # non-words
            re.compile(r"[^a-zA-Z]+"),
            # start with "-"
            re.compile(r"-.+"),
            # times
            re.compile(r"\d{2}:\d{2}(AM|PM)"),
            # ages
            re.compile(r"\d+-year-old"),
            # 's
            re.compile(r"'[sS]"),
            # mmHg
            re.compile(r"(\d+\))?mmHg"),
            # mEq
            re.compile(r"(w/)?\d*(mEq|MEQ|meq|MeQs)"),
        ]

    # def __call__(self, txt):
    #     content = []
    #     for word in txt.split():
    #         if word not in self.black_list \
    #                 and not some(self.black_pattern_list, lambda p: p.fullmatch(word)) \
    #                 and (word or some(self.abbr_patterns, lambda p: p.fullmatch(word)))\
    #                 and word in self.abbr_inventory:
    #             content.append("abbr|{0}|".format(word))
    #         else:
    #             content.append(word)
    #     return " ".join(content)

    def __call__(self, txt_tuple):
        for i in range(len(txt_tuple)):
            word = txt_tuple[i][0]
            if word not in self.black_list \
                    and not some(self.black_pattern_list, lambda p: p.fullmatch(word)) \
                    and (word or some(self.abbr_patterns, lambda p: p.fullmatch(word)))\
                    and word in self.abbr_inventory:
                txt_tuple[i][0] = 'abbr|{0}|'.format(word)
        return txt_tuple

    # def chunks(self, l, n):
    #     """Yield successive n-sized chunks from l."""
    #     for i in range(0, len(l), n):
    #         yield l[i:i + n]
    #
    # def __call__(self, txt):
    #     txt_list = txt.split()
    #     txt_list = list(self.chunks(txt_list, 8))
    #     txt_list_processed = Parallel(n_jobs=8, verbose=1)(
    #         delayed(self.detect_job)(sub_list) for sub_list in txt_list
    #     )
    #     return ' '.join(txt_list_processed)
    #
    # def detect_job(self, txt_list):
    #     content = []
    #     for word in txt_list:
    #         if word not in self.black_list \
    #                 and not some(self.black_pattern_list, lambda p: p.fullmatch(word)) \
    #                 and (word or some(self.abbr_patterns, lambda p: p.fullmatch(word))) \
    #                 and word in self.abbr_inventory:
    #             content.append("abbr|{0}|".format(word))
    #         else:
    #             content.append(word)
    #     return " ".join(content)


# def global_instance_idx_mapper(abbr_index):
#     global_instance_mapper = {}
#     for abbr in abbr_index:
#         for doc_id, pos_list in abbr_index[abbr].items():
#             for global_instance_idx, pos, label in pos_list:
#                 global_instance_mapper[global_instance_idx] = pos
#     return global_instance_mapper

def global_instance_idx_mapper(abbr_index, char_offset_mapper):
    global_instance_mapper = {}
    global_instance_offset_mapper = {}
    for abbr in abbr_index:
        for doc_id, pos_list in abbr_index[abbr].items():
            instance_info = char_offset_mapper[abbr]
            i = 0
            # print(abbr, len(instance_info))
            for global_instance_idx, pos, label in pos_list:
                pos_info = instance_info[i].split(':')
                global_instance_mapper[global_instance_idx] = pos
                global_instance_offset_mapper[global_instance_idx] = pos_info[1]
                i += 1
    return global_instance_mapper, global_instance_offset_mapper


class AbbrInstanceCollectorUPMC(AbbrInstanceCollector):
    def __init__(self, notes):
        self.corpus = notes


def instance_generator(abbr_index, docs, window_size=5):
    dataset = {}
    for abbr in abbr_index:
        corpus = AbbrCorpus(abbr, abbr_index, docs, window_size=window_size)
        corpus_content = corpus.content_generator()
        abbr_instances = []
        for _, _, _, content_pos, content, _ in corpus_content:
            content.insert(content_pos, abbr)
            abbr_instances.append(" ".join(content))
        dataset[abbr] = abbr_instances
    return dataset


def fasttext_classifier(model, pred_abbr_index, pred_abbr_instances, global_char_offset_mapper):
    wsd_results = defaultdict(list)
    for abbr in pred_abbr_index:
        eval_abbr_instance_list = pred_abbr_instances[abbr]
        abbr_instance_idx = 0
        for doc_id, pos_list in pred_abbr_index[abbr].items():
            for global_instance_idx, _, _ in pos_list:
                # get instance
                context = eval_abbr_instance_list[abbr_instance_idx]
                # print(model.predict("year old female . abbr|HPI| Pt is here"))
                # sense = model.predict(context, k=5)[0][0].lstrip("__label__")
                k = 5
                sense = model.predict(context, k=k)
                sense_list = []
                for i in range(k):
                    tup = [sense[0][i].lstrip("__label__"), sense[1][i]]
                    sense_list.append(tup)
                wsd_results[doc_id].append({"abbr": abbr,
                                            "position": global_char_offset_mapper[global_instance_idx],
                                            "sense": sense_list})
                # print(wsd_results[doc_id])
                # print('[' + str(global_instance_idx) + ']' + abbr + ' '
                #       + str(global_char_offset_mapper[global_instance_idx]) + '\n'
                #       + context)
                # print(sense_list)
                # print()
                abbr_instance_idx += 1
    return wsd_results

# def fasttext_classifier(model, pred_abbr_index, pred_abbr_instances, result_global_idx_mapper):
#     wsd_results = defaultdict(list)
#     for abbr in pred_abbr_index:
#         eval_abbr_instance_list = pred_abbr_instances[abbr]
#         abbr_instance_idx = 0
#         for doc_id, pos_list in pred_abbr_index[abbr].items():
#             for global_instance_idx, _, _ in pos_list:
#                 # get instance
#                 context = eval_abbr_instance_list[abbr_instance_idx]
#                 wsd_results[doc_id].append({"position": result_global_idx_mapper[global_instance_idx],
#                                             "sense": model.predict(context)[0][0].lstrip("__label__")})
#                 abbr_instance_idx += 1
#     return wsd_results


# def save_result_to_json(wsd_result, documents_tokenized, file_name=None, indent=False):
#     result_dict = {}
#     if file_name is not None:
#         with open(file_name, 'w') as file:
#             for idx, doc in enumerate(documents_tokenized):
#                 doc_json = OrderedDict()
#                 doc_json["tokenized_text"] = doc.split()
#                 doc_json["wsd"] = wsd_result[idx]
#                 if indent:
#                     json.dump(doc_json, file, indent=4)
#                 else:
#                     json.dump(doc_json, file)
#                 file.write("\n")
#                 result_dict[idx] = doc_json
#     else:
#         for idx, doc in enumerate(documents_tokenized):
#             doc_json = OrderedDict()
#             doc_json["tokenized_text"] = doc.split()
#             doc_json["wsd"] = wsd_result[idx]
#             result_dict[idx] = doc_json
#     return result_dict

def save_result_to_json(wsd_result, documents_tokenized, file_name=None, indent=False):
    """
    Exporting json file using new adaptation
    No tokenized text
    :param wsd_result:
    :param documents_tokenized:
    :param file_name:
    :param indent:
    :return:
    """
    result_dict = {}
    if file_name is not None:
        with open(file_name, 'w') as file:
            for idx, doc in enumerate(documents_tokenized):
                doc_json = OrderedDict()
                doc_json["wsd"] = wsd_result[idx]
                if indent:
                    json.dump(doc_json, file, indent=4)
                else:
                    json.dump(doc_json, file)
                file.write("\n")
                result_dict[idx] = doc_json
    else:
        for idx, doc in enumerate(documents_tokenized):
            doc_json = OrderedDict()
            doc_json["wsd"] = wsd_result[idx]
            result_dict[idx] = doc_json
    return result_dict


class AbbrDisambiguation:

    def __init__(self, train_processed_path, abbr_inventory_path, use_pretrain=False, use_softmax=False):
        """
        Initialize environment & model.
        """

        # Initialize processor and tokenizer
        self.pre_processor = TextProcessor([
            white_space_remover_upmc,
            sub_deid_patterns_upmc])
        # self.tokenizer = CoreNLPTokenizer()
        self.tokenizer = Tokenizer()
        self.post_processor = TextProcessor([AbbrDetector(abbr_inventory_path)])
        self.filter_processor = TextProcessor([
            TextTokenFilter(lowercase=False),
            repeat_non_word_remover])

        # Load model
        train_path = train_processed_path + '/fasttext'
        if use_pretrain:
            model_path = train_path + '/model/pre_train'
        else:
            model_path = train_path + '/model'
        if use_softmax:
            model_file = model_path + '/all_softmax.bin'
        else:
            model_file = model_path + '/all.bin'
        self.model = load_model(model_file)

    def process_single_text(self, text, save_json_path=None):
        """
        Process one text.
        """
        #############################
        # Process document
        #############################

        # text.encode('ascii', 'ignore').decode('utf-8')

        # pre-processing
        # text_processed = self.pre_processor.process_single_text(text)
        text_processed = text
        # print(text_processed)

        # tokenizing
        # text_tokenized = self.tokenizer.process_single_text(text_processed)
        text_tokenized = self.tokenizer.get_tokens(text)
        text_tokenized_pure = ' '.join([item[0] for item in text_tokenized])
        # print(text_tokenized_pure)
        # print(text_tokenized[:3])
        # print()

        # detect abbrs
        text_detected = self.post_processor.process_single_text(text_tokenized)
        text_detected_pure = ' '.join([item[0] for item in text_detected])
        # print(text_detected)
        # print(text_detected_pure)

        # Filter trivial tokens and Remove repeat non-words
        # text_filtered = self.filter_processor.process_single_text(text_tokenized)
        text_filtered = self.filter_processor.process_single_text(text_detected)
        text_filtered_pure = ' '.join([item[0] for item in text_filtered])
        # print(text_filtered)
        # print(text_filtered_pure)

        # print()
        # print('<=' + str(time_0) + '=> Start')
        # print('<=' + str(time_1) + '=> Pre-processing')
        # print('<=' + str(time_2) + '=> Tokenizing')
        # print('<=' + str(time_3) + '=> Abbr detect')
        # print('<=' + str(time_4) + '=> Filter')

        #############################
        # Build index
        #############################
        result_collector = AbbrInstanceCollectorUPMC([text_detected_pure])
        abbr_index_result, document_no_mark_result, abbr_offset = result_collector.generate_inverted_index(
            # '/home/luoz3/wsd_data_test/abbr_index_result.pkl',
            offset_mapping=text_detected,
            # offset_mapping=text_tokenized
        )
        result_global_idx_mapper, global_offset_mapper = global_instance_idx_mapper(abbr_index_result, abbr_offset)
        # print('result collector finished')

        pred_collector = AbbrInstanceCollectorUPMC([text_filtered_pure])
        abbr_index_pred, document_no_mark_pred, abbr_offset_pred = pred_collector.generate_inverted_index(
            # '/home/luoz3/wsd_data_test/abbr_index_pred.pkl',
            offset_mapping=text_filtered)
        # print(abbr_index_pred,document_no_mark_pred,abbr_offset_pred)
        abbr_instances_pred = instance_generator(abbr_index_pred, Doc(document_no_mark_pred))
        # print('predict collector finished')

        # print(abbr_index_result['HPI'])
        # print(abbr_index_result['AE'])

        #############################
        # Do classification
        #############################

        wsd_results = fasttext_classifier(self.model,
                                          abbr_index_pred,
                                          abbr_instances_pred,
                                          # result_global_idx_mapper,
                                          # abbr_offset_pred,
                                          global_offset_mapper)
        return text_processed, save_result_to_json(wsd_results, document_no_mark_result, save_json_path)

    def process_texts(self, text_list, save_json_path=None, n_jobs=8):
        """
        Process list of texts.
        """
        #############################
        # Process document
        #############################

        # pre-processing
        # text = self.pre_processor.process_texts(text_list, n_jobs=n_jobs)
        text = text_list

        # tokenizing
        text_tokenized = self.tokenizer.process_texts(text, n_jobs=n_jobs)

        # detect abbrs
        text_detected = self.post_processor.process_texts(text_tokenized, n_jobs=n_jobs)
        # Filter trivial tokens and Remove repeat non-words
        text_filtered = self.filter_processor.process_texts(text_detected, n_jobs=n_jobs)

        #############################
        # Build index
        #############################
        print("Building index...")
        result_collector = AbbrInstanceCollectorUPMC(text_detected)
        abbr_index_result, document_no_mark_result = result_collector.generate_inverted_index()
        result_global_idx_mapper = global_instance_idx_mapper(abbr_index_result)

        pred_collector = AbbrInstanceCollectorUPMC(text_filtered)
        abbr_index_pred, document_no_mark_pred = pred_collector.generate_inverted_index()
        abbr_instances_pred = instance_generator(abbr_index_pred, Doc(document_no_mark_pred))

        #############################
        # Do classification
        #############################
        print("Predicting...")
        wsd_results = fasttext_classifier(self.model, abbr_index_pred, abbr_instances_pred, result_global_idx_mapper)
        return save_result_to_json(wsd_results, document_no_mark_result, save_json_path)


if __name__ == '__main__':

    ###############
    # File paths
    ###############
    # dataset_paths = DataSetPaths('luoz3_x1')
    # data_path = "/home/luoz3/wsd_data"
    # dataset_processed_path = data_path + "/upmc/example/processed"
    # abbr_inventory_path = data_path + "/abbr_inventory.pkl"
    # example_note_path = "/data/batch4/500K_By_Sources_2013_output/PEGREMO_21/doc.9071.txt"
    dataset_paths = DataSetPaths('xil222')
    data_path = "/home/luoz3/wsd_data_test"
    dataset_processed_path = data_path + "/upmc/AL/processed"
    abbr_inventory_path = data_path + "/abbr_inventory.pkl"
    dataset_text = '/data/batch4/500K_By_Sources_2013_output/PEGREMO_21/'
    example_note_path = "/data/batch4/500K_By_Sources_2013_output/PEGREMO_21/doc.9071.txt"
    example_note_path2 = '/home/wangz12/scripts/generate_trainning_data/training_data_AL.txt'
    sample_train_file_path = data_path + '/upmc/AL/processed/upmc_train.txt'

    ####################
    # load raw txt note
    ####################
    # with open(example_note_path, 'r') as file:
    #     example_note = "".join(file.readlines())
    example_note = ''
    with open(example_note_path2, 'r') as file:
        for line in file:
            example_note += line
            # print(line)
            if 'HPI' in line:
                break
    # print(example_note)

    # example_note = "Patient MRN : ID-NUM-DEID \nPatient XXXX of XXXXX : DATE-DEID \nPatient Sex : M \nDocumented On : " \
    #                "DATE-DEID \n\nPATIENT MRN : ID-NUM-DEID \nPATIENT DOB : DATE-DEID \nAGE : AGE-DEID years \nGENDER " \
    #                ": XXXX \nPROVIDED CLINICAL HISTORY : `` Increasing O2 requirements '' \n\nCOMPARISON : \nXXXXX " \
    #                "x-ray from DATE-DEID . \n\nXXXXXXXXX : \nXXXXXX portable XXXXXXX AP  chest radiograph . " \
    #                "\n\nFINDINGS : \nWhen compared to chest x-ray performed at 6:08 AM on DATE-DEID , \nthere is " \
    #                "markedly increased size of the right pleural effusion , \ncurrently moderate to large in size . " \
    #                "The left small pleural effusion \nis unchanged . No pneumothorax is identified . \n\nThe NG tube " \
    #                "is in the proximal stomach . \n\nIMPRESSION : \n* Increased size of right pleural effusion , " \
    #                "currently moderate to \nlarge . \n\nDictated by : NAME-DEID \nSigned by : NAME-DEID \nSigned on : " \
    #                "DATE-DEID 05:27 AM "

    example_note_2 = 'Addendum by NAME-DEID MD , NAME-DEID on DATE-DEID 8:34 PM : ↵ I was called to see the patient ' \
                     'due to persistent tachycardia . The admission note and the events of the day XXXX reviewed . I ' \
                     'also spoke personally with the Urology housestaff . The patient likely has a XXXX with urinoma ' \
                     'in the pelvis . She has GNR bacteremia and sepsis . The urology team would like to take the ' \
                     'patient to the OR tomorrow for stenting . She remains critically ill at this time with : ↵ ↵ 1 ' \
                     '. Sepsis - tachycardic with GNR bacteremia - urine source as noted above . She has antibiotics ' \
                     'on board that are broad and should cover thi . We continue to resuscitate her with IVF . She is ' \
                     'developng a metabolic acidosis , and I will add some sodium acetate to her fluids . q4 hours ' \
                     'labs have XXXX ordered . ABG and lactate now . ↵ ↵ 2 . NPO after midnight for urology procedure ' \
                     'in the AM All labs done and appropriate . ↵ ↵ I spent an additional 55 minutes of CCT ' \
                     'evaluating this patient and discussing her plan with the patient , family , and surgical team . ' \
                     'Sepsis . ↵ ↵ NAME-DEID , MD , MPH , FACP ↵ Pulmonary and Critical XXXX Medicine ↵ XXXXX : ' \
                     'PHONE-DEID ↵ ↵ ↵ Perform - Completed by NAME-DEID MD , NAME-DEID -LRB- on DATE-DEID 06:39 -RRB- ' \
                     '↵ Modify - Completed by NAME-DEID MD , NAME-DEID -LRB- on DATE-DEID 06:44 -RRB- ↵ Modify - ' \
                     'Completed by NAME-DEID MD , NAME-DEID -LRB- on DATE-DEID 06:47 -RRB- ↵ Modify - Completed by ' \
                     'NAME-DEID MD , NAME-DEID -LRB- on DATE-DEID 06:54 -RRB- ↵ Modify - Completed by NAME-DEID MD , ' \
                     'NAME-DEID -LRB- on DATE-DEID 06:54 -RRB- ↵ Sign - Completed by NAME-DEID MD , NAME-DEID -LRB- ' \
                     'on DATE-DEID 06:54 -RRB- ↵ VERIFY - Completed by NAME-DEID MD , NAME-DEID -LRB- on DATE-DEID ' \
                     '06:54 -RRB- ↵ Sign - Completed by NAME-DEID MD , NAME-DEID -LRB- on DATE-DEID 20:34 -RRB- ↵ ' \
                     'Modify - Completed by NAME-DEID MD , NAME-DEID -LRB- on DATE-DEID 20:34 -RRB- '

    example_note_3 = 'DOCUMENT INFO:Document Date:  **DATE[Oct 11 2013] 13:54Document Type:  PEGREMO_21Document ' \
                     'Source:        PROVATIONFacility:Document Status:        finalDocument Baseline:      ' \
                     'RadiologyType Description:       Non-endoscopic G-tube exchangeDOCUMENT CONTENT:**INSTITUTION   ' \
                     '             Division of Gastroenterology, Hepatology & Nutrition                MRN: **ID-NUM  ' \
                     '              Date of Birth: **DATE[Apr 09 1979]                Admit Type: Outpatient          ' \
                     '      Age: **AGE[in 30s]                Gender: Female                Patient Location: MPUP    ' \
                     '            Account Number: **ID-NUM                Procedure Date: **DATE[Oct 11 2013] 1:54 PM ' \
                     '               **NAME[ZZZ] MD: **NAME[YYY XXX], MD                Note Status: Finalized        ' \
                     '        Procedure:         Non-endoscopic G-tube exchange                Indications:       ' \
                     'Exchange G-Tube                Providers:         **NAME[YYY XXX], MD                Referring ' \
                     'MD:      **NAME[WWW VVV], MD                Medicines:         General anesthesia               ' \
                     ' Complications:     No immediate complications.                Procedure:         ' \
                     'Pre-Anesthesia Assessment:                                   - Pre-procedure physical ' \
                     'examination revealed no                                   contraindications to sedation.        ' \
                     '                           After obtaining informed consent, the site was                       ' \
                     '            prepped and the procedure was performed. The                                   ' \
                     'procedure was accomplished without difficulty. The                                   patient ' \
                     'tolerated the procedure well.                Findings:          Position of the surgically ' \
                     'placed PEG tube into the                                   excluded stomach was confirmed with ' \
                     'injection of                                   contrast under fluoroscopy.                      ' \
                     '             A guidewire was advanced through the tube and coiled                               ' \
                     '    in the stomach under fluoroscopic guidance.                                   The balloon ' \
                     'was then deflated and the PEG tube was                                   removed over the ' \
                     'guidewire.                                   The new 24Fr 4 cm PEG button was advanced over the ' \
                     '                                  guidewire through the previously performed                    ' \
                     '               gastrostomy into the stomach.                                   The balloon was ' \
                     'inflated with 8 cc of water and the                                   position of the button ' \
                     'was secured.                Impression:        - Successful PEG tube exchange.                ' \
                     'Recommendation:    - Keflex 500 mg po QID for 7 days.                                   - ' \
                     'Topical Nystatin BID for 7 days.                                   - Return to referring ' \
                     'physician as previously                                   scheduled.                **NAME[YYY ' \
                     'XXX], MD                **DATE[Oct 11 2013] 3:25 PM                Number of Addenda: 0         ' \
                     '       Note Initiated On: **DATE[Oct 11 2013] 1:54 PM                **STREET-ADDRESS, **PLACE, ' \
                     'PA **ZIP-CODE '

    # abbr|HPI|797335
    example_note_4 = 'Cats ↵ ↵ ↵ ↵ Review of Systems ↵ 12 pt ROS negative except as per HPI ↵ ↵ Objective ' \
                     '↵ Vital Signs '

    wsd = AbbrDisambiguation(
        # train_processed_path=dataset_paths.upmc_ab_train_folder,
        train_processed_path=dataset_paths.upmc_ao_train_folder,
        abbr_inventory_path=abbr_inventory_path,
        use_pretrain=False,
        use_softmax=False)

    # wsd = AbbrDisambiguation(
    #     train_processed_path=dataset_paths.upmc_al_train_folder,
    #     abbr_inventory_path=abbr_inventory_path,
    #     use_pretrain=False,
    #     use_softmax=True)

    result = wsd.process_single_text(example_note_4, save_json_path=dataset_processed_path+"/wsd_result.json")
    # result2 = wsd.process_texts(dataset_text, save_json_path=dataset_processed_path+"/wsd_result.json")

    print()
