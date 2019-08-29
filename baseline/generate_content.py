"""
Generate content vectors for datasets.

"""
import pickle
import os
import gensim
import numpy as np
import tqdm
from joblib import Parallel, delayed
from baseline.word_embedding import AbbrIndex
from baseline.dataset_helper import DataSetPaths
from datetime import datetime
from preprocess.file_helper import txt_reader, pickle_writer


class Doc:
    def __init__(self, txt):
        self.txt = txt

    def __len__(self):
        return len(self.txt)

    def __getitem__(self, item):
        doc = self.txt[item].split(" ")
        return doc


class AbbrCorpus:
    """
    Get content for each instances of an abbr.
    """

    def __init__(self, word, abbr_index, docs, window_size=5):
        assert isinstance(word, str)
        assert isinstance(abbr_index, AbbrIndex)
        assert isinstance(docs, Doc)
        self.word = word
        self.window_size = window_size
        self.index = abbr_index[word]
        self.docs = docs
        self.instance_idx = []
        self._build_instance_index()

    def __getitem__(self, idx):
        return self.instance_idx[idx]

    def __len__(self):
        return len(self.instance_idx)

    def _build_instance_index(self):
        """Build instance index for every instances of this abbr.
        Only called when initializing.
        """
        for doc_id, pos_list in self.index.items():
            for global_instance_idx, pos, label in pos_list:
                self.instance_idx.append((global_instance_idx, doc_id, pos, label))

    def get_content(self, idx, window_size=None):
        if window_size is None:
            window_size = self.window_size

        global_instance_idx, doc_id, pos, label = self.instance_idx[idx]
        doc = self.docs[doc_id]
        # get content words
        if pos - window_size < 0:
            content = doc[:pos + window_size + 1]
        elif pos + window_size + 1 > len(doc):
            content = doc[pos - window_size:]
        else:
            content = doc[pos - window_size:pos + window_size + 1]
        content_pos = content.index(self.word)
        content.pop(content_pos)
        return global_instance_idx, doc_id, pos, content_pos, content, label

    def content_generator(self, window_size=None):
        if window_size is None:
            window_size = self.window_size

        for doc_id, pos_list in self.index.items():
            doc = self.docs[doc_id]
            for global_instance_idx, pos, label in pos_list:
                # print(pos, label)
                # get content words
                if pos - window_size < 0:
                    content = doc[:pos + window_size + 1]
                elif pos + window_size + 1 > len(doc):
                    content = doc[pos - window_size:]
                else:
                    content = doc[pos - window_size:pos + window_size + 1]
                content_pos = content.index(self.word)
                content.pop(content_pos)
                yield (global_instance_idx, doc_id, pos, content_pos, content, label)

def compute_content_word2vec(content, model):
    assert content != []
    vector = np.zeros(model.vector_size)
    count = 0
    for word in content:
        if word in model.wv:
            count += 1
            vector += model.wv[word]

    # For IPDC only
    # vector = np.nan_to_num(vector)
    # for i in range(len(vector)):
    #     if np.isnan(vector[i]):
    #         vector[i] = 0
    #         print(vector[i])
    # if any(np.isnan(item) for item in vector):
    #     # print('!!!!!')
    #     print(vector)
    return vector / count


def compute_jaccard(content1, content2):
    assert content2 != []
    set1 = set(content1)
    set2 = set(content2)
    return len(set1 & set2) / len(set1 | set2)


def abbr_job(abbr, abbr_index, abbr_idx_mapper, docs, model, content_dir):
    corpus = AbbrCorpus(abbr, abbr_index, docs)
    corpus_content = corpus.content_generator()

    abbr_content_vec = []
    for global_instance_idx, doc_id, pos, content_pos, content, label in corpus_content:
        content_vec = compute_content_word2vec(content, model)
        old_len = len(content_vec)
        content_vec = np.nan_to_num(content_vec)
        assert old_len == len(content_vec)
        for item in content_vec:
            if np.isnan(item):
                print(abbr)
                # print(global_instance_idx, doc_id, pos)
                print(content)
                # print(item)
                print('==============================')
        content.insert(content_pos, abbr)
        content = " ".join(content)
        abbr_content_vec.append((global_instance_idx, doc_id, pos, content_pos, content_vec, content, label))

    # save vector to pickle file
    index = abbr_idx_mapper['abbr2idx'][abbr]
    pickle_writer(abbr_content_vec, content_dir + '%d_vector.pkl' % index)


def build_index_of_abbrs(abbr_index):
    """
    Build index for abbrs (for saving pickle files)

    :return:
    """
    abbr2idx = {}
    idx2abbr = {}
    idx = 0
    for abbr in abbr_index:
        abbr2idx[abbr] = idx
        idx2abbr[idx] = abbr
        idx += 1
    abbr_idx_mapper = {
        'abbr2idx': abbr2idx,
        'idx2abbr': idx2abbr
    }
    return abbr_idx_mapper


def generate_train_content(train_processed_path):
    # Load word2vec vectors
    model = gensim.models.Word2Vec.load(train_processed_path + '/train.model')

    # Load abbr index
    abbr_index = AbbrIndex(train_processed_path + '/abbr_index_data.pkl')
    train_docs = Doc(txt_reader(train_processed_path + "/train_no_mark.txt"))

    # Build index for abbrs (for saving pickle files)
    abbr_idx_mapper = build_index_of_abbrs(abbr_index)
    pickle_writer(abbr_idx_mapper, train_processed_path + '/abbr_idx_mapper.pkl')

    # Save all content vectors to pickle files
    content_dir = train_processed_path + '/content_vectors/'
    os.makedirs(content_dir, exist_ok=True)

    print("Saving content vectors...")
    print(len(abbr_index))

    for abbr in tqdm.tqdm(abbr_index):
        abbr_job(abbr, abbr_index, abbr_idx_mapper, train_docs, model, content_dir)
        # print(abbr, abbr_index, abbr_idx_mapper)

    # Parallel(n_jobs=30, verbose=3)(
    #     delayed(abbr_job)(abbr, abbr_index, train_docs, model, content_dir) for abbr in abbr_index if "/" not in abbr)


def generate_test_content(test_processed_path, train_processed_path):
    # Load word2vec vectors
    model = gensim.models.Word2Vec.load(train_processed_path + '/train.model')

    # Load abbr index
    abbr_index = AbbrIndex(test_processed_path + '/abbr_index_data.pkl')
    train_docs = Doc(txt_reader(test_processed_path + "/test_no_mark.txt"))

    # Build index for abbrs (for saving pickle files)
    abbr_idx_mapper = build_index_of_abbrs(abbr_index)
    pickle_writer(abbr_idx_mapper, test_processed_path + '/abbr_idx_mapper.pkl')

    # Save all content vectors to pickle files
    content_dir = test_processed_path + '/content_vectors/'
    os.makedirs(content_dir, exist_ok=True)

    print("Saving content vectors...")
    print(len(abbr_index))

    for abbr in tqdm.tqdm(abbr_index):
        abbr_job(abbr, abbr_index, abbr_idx_mapper, train_docs, model, content_dir)

    # Parallel(n_jobs=30, verbose=3)(
    #     delayed(abbr_job)(abbr, abbr_index, train_docs, model, content_dir) for abbr in abbr_index if "/" not in abbr)


if __name__ == '__main__':
    # dataset_paths = DataSetPaths('luoz3_x1')
    dataset_paths = DataSetPaths('xil222')
    start_time = datetime.now()

    # generate_train_content(dataset_paths.mimic_train_folder)
    # generate_train_content(dataset_paths.upmc_ab_train_folder)
    # generate_train_content(dataset_paths.upmc_ad_train_folder)
    # generate_train_content(dataset_paths.upmc_ag_train_folder)
    # generate_train_content(dataset_paths.upmc_al_train_folder)
    # generate_train_content(dataset_paths.upmc_ao_train_folder)

    # generate_test_content(dataset_paths.mimic_test_folder, dataset_paths.mimic_train_folder)
    # generate_test_content(dataset_paths.msh_test_folder, dataset_paths.mimic_train_folder)
    # generate_test_content(dataset_paths.share_test_folder, dataset_paths.mimic_train_folder)
    # generate_test_content(dataset_paths.umn_test_folder, dataset_paths.mimic_train_folder)
    # generate_test_content(dataset_paths.upmc_example_folder, dataset_paths.mimic_train_folder)
    # generate_test_content(dataset_paths.upmc_ab_test_folder, dataset_paths.upmc_ab_train_folder)
    # generate_test_content(dataset_paths.upmc_ad_test_folder, dataset_paths.upmc_ad_train_folder)
    # generate_test_content(dataset_paths.upmc_ag_test_folder, dataset_paths.upmc_ag_train_folder)
    # generate_test_content(dataset_paths.upmc_al_test_folder, dataset_paths.upmc_al_train_folder)
    # generate_test_content(dataset_paths.upmc_ao_test_folder, dataset_paths.upmc_ao_train_folder)
    # generate_test_content(dataset_paths.pipeline_test_folder, dataset_paths.upmc_ao_train_folder)

    ##############################
    # Discharge notes
    ##############################
    # generate_train_content(dataset_paths.pe_self_train_folder)

    # generate_test_content(dataset_paths.pe_50000_nm_test_folder, dataset_paths.upmc_al_train_folder)
    # generate_test_content(dataset_paths.pe_self_test_folder, dataset_paths.pe_self_train_folder)
    # generate_test_content(dataset_paths.ipdc_50000_test_folder, dataset_paths.upmc_al_train_folder)

    ##############################
    # MIMIC
    ##############################
    # generate_train_content(dataset_paths.mimic_new_train_folder)
    # generate_train_content(dataset_paths.mimic_clus_k20_s5_train_folder)
    # generate_train_content(dataset_paths.mimic_clus_k20_s10_train_folder)
    # generate_train_content(dataset_paths.mimic_clus_k20_s15_train_folder)
    # generate_train_content(dataset_paths.mimic_clus_k30_s5_train_folder)
    # generate_train_content(dataset_paths.mimic_clus_k30_s10_train_folder)
    # generate_train_content(dataset_paths.mimic_clus_k30_s15_train_folder)
    # generate_train_content(dataset_paths.mimic_clus_k40_s5_train_folder)
    # generate_train_content(dataset_paths.mimic_clus_k40_s10_train_folder)
    # generate_train_content(dataset_paths.mimic_clus_k40_s15_train_folder)
    # generate_train_content(dataset_paths.mimic_clus_all_train_folder)
    # generate_train_content(dataset_paths.mimic_clus_k2_s10_train_folder)
    generate_train_content(dataset_paths.mimic_clus_k15_s10_train_folder)

    # generate_test_content(dataset_paths.mimic_new_test_folder, dataset_paths.mimic_new_train_folder)
    # generate_test_content(dataset_paths.mimic_clus_k20_s5_test_folder, dataset_paths.mimic_clus_k20_s5_train_folder)
    # generate_test_content(dataset_paths.mimic_clus_k20_s10_test_folder, dataset_paths.mimic_clus_k20_s10_train_folder)
    # generate_test_content(dataset_paths.mimic_clus_k20_s15_test_folder, dataset_paths.mimic_clus_k20_s15_train_folder)
    # generate_test_content(dataset_paths.mimic_clus_k30_s5_test_folder, dataset_paths.mimic_clus_k30_s5_train_folder)
    # generate_test_content(dataset_paths.mimic_clus_k30_s10_test_folder, dataset_paths.mimic_clus_k30_s10_train_folder)
    # generate_test_content(dataset_paths.mimic_clus_k30_s15_test_folder, dataset_paths.mimic_clus_k30_s15_train_folder)
    # generate_test_content(dataset_paths.mimic_clus_k40_s5_test_folder, dataset_paths.mimic_clus_k40_s5_train_folder)
    # generate_test_content(dataset_paths.mimic_clus_k40_s10_test_folder, dataset_paths.mimic_clus_k40_s10_train_folder)
    # generate_test_content(dataset_paths.mimic_clus_k40_s15_test_folder, dataset_paths.mimic_clus_k40_s15_train_folder)
    # generate_test_content(dataset_paths.mimic_clus_all_test_folder, dataset_paths.mimic_clus_all_train_folder)
    # generate_test_content(dataset_paths.mimic_clus_k2_s10_test_folder, dataset_paths.mimic_clus_k2_s10_train_folder)
    generate_test_content(dataset_paths.mimic_clus_k15_s10_test_folder, dataset_paths.mimic_clus_k15_s10_train_folder)

    runtime = datetime.now() - start_time
    print('Finished in [' + str(runtime) + ']')
