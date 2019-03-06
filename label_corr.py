import numpy as np
import itertools
import json
import sys
from collections import defaultdict
from tqdm import tqdm
import gluonnlp
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

sys.path.insert(0, './resources')
import constant

def build_concurr_matrix(emb_name='fasttext', emb_source='wiki-news-300d-1M-subword', goal='open'):
# def build_concurr_matrix(emb_name='glove', emb_source='glove.840B.300d', goal='open'):
    data_path = 'data/release/'
    # build the yid concurr matrix
    if goal == 'onto':
        label2id = constant.ANS2ID_DICT["onto"]
        id2label = constant.g_id2ans
    else:
        label2id = constant.ANS2ID_DICT["open"]
        id2label = constant.open_id2ans

    if goal != 'onto':
        print('Building label word embedding for open')
        words = []
        for label in label2id.keys():
            words += label.split('_')  
        word_counter = gluonnlp.data.count_tokens(words)
        word_vocab = gluonnlp.Vocab(word_counter)
        embed = gluonnlp.embedding.create(emb_name, source=emb_source)
        word_vocab.set_embedding(embed)
        label_vectors = []
        for id_ in range(len(label2id.keys())):
            label = id2label[id_]
            label_words = label.split('_')
            label_vectors.append(word_vocab.embedding[label_words].asnumpy().sum(0))
        affinity = cosine_similarity(label_vectors)
    else:
        print("BOW features for ontonotes")
        words = []
        for label in label2id.keys():
            label = label.replace('/', ' ')
            labels = label.strip().split()
            words += labels
        word_counter = gluonnlp.data.count_tokens(words)
        word_vocab = gluonnlp.Vocab(word_counter)
        embed = gluonnlp.embedding.create(emb_name, source=emb_source)
        word_vocab.set_embedding(embed)

        label_list = []
        label_vectors = []
        for id_ in range(len(label2id.keys())):
            label = id2label[id_]
            label = label.replace('/', ' ')
            labels = label.strip().split()
            label_list.append(labels)
            label_vectors.append(word_vocab.embedding[labels].asnumpy().sum(0))
        label_vectors = np.array(label_vectors)
        affinity = cosine_similarity(label_vectors)

    matrix = np.zeros((len(label2id.keys()), len(label2id.keys())))
    if goal == 'onto':
        train_file_list = ['ontonotes/augmented_train.json']
    else:
        train_file_list = ['distant_supervision/headword_train.json', 'distant_supervision/el_train.json', 'crowd/train_m.json']

    type_count = defaultdict(int)
    for f_id, file in enumerate(train_file_list):
        file = data_path + file
        with open(file) as f:
            for sent in tqdm(f.readlines()):
                line_elem = json.loads(sent.strip())
                y_strs = line_elem['y_str']
                # y_ids = list(set([label2id[x] for x in y_strs if x in label2id]))
                for x in y_strs:
                    type_count[x] += 1


                y_ids = [label2id[x] for x in y_strs if x in label2id]
                if len(y_ids) > 1:
                    for (x, y) in itertools.combinations(y_ids, 2):
                        # if x == y:
                        #     print(y_strs)
                        #     assert False
                        matrix[x,y] = matrix[x,y] + 1


    # print(type_count['child'])
    # print(type_count['daughter'])
    # print(np.mean(list(type_count.values())))

    # add self-connection
    matrix += np.identity(matrix.shape[0])

    # print(len(concurr_labels))
    # print(np.count_nonzero(matrix)/np.prod(matrix.shape))
    target = np.tanh(np.log(matrix + 1e-8))
    mask = (matrix == 0).astype(float)
    mask_inverse = (matrix > 0).astype(float)

    return matrix, affinity, target, mask, mask_inverse

if __name__ == '__main__':
    co_occurence, _, _, _, _ = build_concurr_matrix()
    co_occurence = co_occurence - np.identity(co_occurence.shape[0])
    print(np.max(co_occurence))

    # id2label = constant.open_id2ans
    # label2id = constant.ANS2ID_DICT["open"]
    # person_id = label2id['person']

    # # print(co_occurence[:10,:10])

    # person_id_row = co_occurence[person_id, :]
    # label_freq = {}
    # inconsistent_pairs = []
    # for index, value in enumerate(person_id_row):
    #     if value != 0:
    #         label_freq[id2label[index]] = value
    #     else:
    #         inconsistent_pairs.append(['person', id2label[index]])
    # # print(label_freq)
    # print(inconsistent_pairs)

    # with open(z)