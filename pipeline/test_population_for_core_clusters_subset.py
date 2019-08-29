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
from baseline.dataset_helper import AbbrInstanceCollector, DataSetPaths
from baseline.generate_content import AbbrCorpus, Doc
from pipeline.fasttext import AbbrDisambiguation as AbbrDisambiguation2

import psycopg2
from psycopg2.extras import Json

import json

### change input cluster file here
upmc_no_mark = "/home/luoz3/wsd_data/upmc_batch1_4/upmc_no_mark_old.json"
upmc_cluster_result = "/home/luoz3/wsd_data/upmc_batch1_4/upmc_clustering_result_I_L.json"

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


def some(array, fn):
    for item in array:
        if fn(item):
            return True
    return False


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

    def __call__(self, txt):
        content = []
        for word in txt.split():
            if word not in self.black_list \
                    and not some(self.black_pattern_list, lambda p: p.fullmatch(word)) \
                    and (word or some(self.abbr_patterns, lambda p: p.fullmatch(word))) \
                    and word in self.abbr_inventory:
                content.append("abbr|{0}|".format(word))
            else:
                content.append(word)
        return " ".join(content)


def global_instance_idx_mapper(abbr_index):
    global_instance_mapper = {}
    for abbr in abbr_index:
        for doc_id, pos_list in abbr_index[abbr].items():
            for global_instance_idx, pos, label in pos_list:
                global_instance_mapper[global_instance_idx] = pos
    return global_instance_mapper


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


def fasttext_classifier(model, pred_abbr_index, pred_abbr_instances, result_global_idx_mapper):
    wsd_results = defaultdict(list)
    for abbr in pred_abbr_index:
        eval_abbr_instance_list = pred_abbr_instances[abbr]
        abbr_instance_idx = 0
        for doc_id, pos_list in pred_abbr_index[abbr].items():
            for global_instance_idx, _, _ in pos_list:
                # get instance
                context = eval_abbr_instance_list[abbr_instance_idx]
                wsd_results[doc_id].append({"position": result_global_idx_mapper[global_instance_idx],
                                            "sense": model.predict(context)[0][0].lstrip("__label__")})
                abbr_instance_idx += 1
    return wsd_results


def save_result_to_json(wsd_result, documents_tokenized, file_name=None, indent=False):
    result_dict = {}
    if file_name is not None:
        with open(file_name, 'w') as file:
            for idx, doc in enumerate(documents_tokenized):
                doc_json = OrderedDict()
                doc_json["tokenized_text"] = doc.split()
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
            doc_json["tokenized_text"] = doc.split()
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
        self.tokenizer = CoreNLPTokenizer()
        self.post_processor = TextProcessor([AbbrDetector(abbr_inventory_path)])
        self.filter_processor = TextProcessor([
            TextTokenFilter(),
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

        # pre-processing
        print("pre processing...")
        text = self.pre_processor.process_single_text(text)
        # tokenizing
        print("tokenizing...")
        text_tokenized = self.tokenizer.process_single_text(text)
        # detect abbrs
        print("abbr detect...")
        text_detected = self.post_processor.process_single_text(text_tokenized)
        # Filter trivial tokens and Remove repeat non-words
        print("filter...")
        text_filtered = self.filter_processor.process_single_text(text_detected)

        #############################
        # Build index
        #############################

        result_collector = AbbrInstanceCollectorUPMC([text_detected])
        abbr_index_result, document_no_mark_result = result_collector.generate_inverted_index()
        result_global_idx_mapper = global_instance_idx_mapper(abbr_index_result)

        pred_collector = AbbrInstanceCollectorUPMC([text_filtered])
        abbr_index_pred, document_no_mark_pred = pred_collector.generate_inverted_index()
        abbr_instances_pred = instance_generator(abbr_index_pred, Doc(document_no_mark_pred))

        #############################
        # Do classification
        #############################

        wsd_results = fasttext_classifier(self.model, abbr_index_pred, abbr_instances_pred, result_global_idx_mapper)
        return save_result_to_json(wsd_results, document_no_mark_result, save_json_path)

    def process_texts(self, text_list, save_json_path=None, n_jobs=8):
        """
        Process list of texts.
        """
        #############################
        # Process document
        #############################

        # pre-processing
        print("pre processing...")
        text = self.pre_processor.process_texts(text_list, n_jobs=n_jobs)
        print("tokenizing...")
        # tokenizing
        text_tokenized = self.tokenizer.process_texts(text, n_jobs=n_jobs)
        print("abbr detect...")
        # detect abbrs
        text_detected = self.post_processor.process_texts(text_tokenized, n_jobs=n_jobs)
        print("filter...")
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


    abbrs = ['AV', 'HIV', 'RBC', 'CV', 'PCA', 'EMS', 'PTT', 'BAL', 'AM', 'AP', 'JP', 'PT', 'BP', 'OGT', 'LCX', 'PCP', 'MCA',
     'SI', 'LOC', 'EKG']

    abbr = "AP"
    k = 20
    s = 15

    for s in [5,10,15]:
        for k in [20,30,40]:
            for abbr in abbrs:

                dataset_paths = DataSetPaths('xil222')
                data_path = "/home/luoz3/wsd_data_test"
                dataset_processed_path = data_path + "/mimic_cluster_k{0}_s{1}/processed".format(k, s)
                abbr_inventory_path = data_path + "/abbr_inventory.pkl"
                example_note_path = "/data/batch4/500K_By_Sources_2013_output/PEGREMO_21/doc.9071.txt"

                # # load raw txt note
                # with open(example_note_path, 'r') as file:
                #     example_note = "".join(file.readlines())

                wsd = AbbrDisambiguation2(
                    train_processed_path=getattr(dataset_paths, "mimic_clus_k{0}_s{1}_train_folder".format(k, s)),
                    abbr_inventory_path=abbr_inventory_path,
                    use_pretrain=False,
                    use_softmax=False)


                cursor = conn.cursor()
                cursor.execute("""
                       with t as (select cluster_id, "textLiteral", ROW_NUMBER() over (partition by cluster_id order by cluster_instance.id) as row
                   from cluster_instance, cluster_abbr
                   where cluster_instance.abbr_id = cluster_abbr.id
                     and cluster_abbr.abbr = '{0}'
                     and cluster_id != -1
                     )
                   select *
                   from t
                   where row = 1
                       """.format(abbr))
                cluster_data = [(row[0], row[1]) for row in cursor]


                def loadSenses(abbr):
                    cursor.execute("""
                           select sense_id,abbr
                           from sense
                           where abbr ilike '{0}'
                           """.format(abbr))
                    return {row[0] for row in cursor}


                CUIs_set = loadSenses(abbr)

                result = {}

                for (cluster_id, text) in cluster_data:
                    try:
                        raw, wsd_result = wsd.process_single_text(text)
                        result[cluster_id] = wsd_result[0]["wsd"]
                    except:
                        print("error", cluster_id, text)

                import pickle

                with open("./result_mimic_k{1}_s{2}_{0}.pkl".format(abbr, k, s), "wb") as result_file:
                    pickle.dump(result, result_file)




