import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer

from sklearn.feature_extraction.text import TfidfTransformer

from gensim.models import Word2Vec, KeyedVectors

# from gensim.utils import simple_preprocess
# # from gensim.models.doc2vec import TaggedDocument, Doc2Vec

# # import smart_open
# # import random

"""
Some codes are taken from https://github.com/dennybritz/cnn-text-classification-tf
"""

data_location = "/extra/zhuzhang001/intention_mining_revision_2/data/"

def load_text_data(dataset):
    text_list = list(open(data_location + dataset + "/tweets_no_urls.txt").readlines())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_list)

    seq = tokenizer.texts_to_sequences(text_list)
    vocabulary_length = len(tokenizer.word_index) + 1

    return seq, vocabulary_length


def load_text_data_for_sentiment(dataset):
    text_list = list(open(data_location + dataset + "/tweets_no_urls.txt").readlines())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_list)

    seq = tokenizer.texts_to_sequences(text_list)
    vocabulary_length = len(tokenizer.word_index) + 1

    return text_list, seq, vocabulary_length


def load_text_data2(dataset):
    text_list = list(open(data_location + dataset + "/tweets_no_urls.txt").readlines())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_list)

    seq = tokenizer.texts_to_sequences(text_list)
    text_matrix = tokenizer.texts_to_matrix(text_list, mode="tfidf")

    vocabulary_length = text_matrix.shape[1]

    return seq, text_matrix, vocabulary_length


def load_text_data3(dataset):
    # for open() in python 3
    text_list = list(open(data_location + dataset + "/tweets_no_urls.txt", encoding='utf-8').readlines())

    # for open() in python 2
    # text_list = list(open(data_location + dataset + "/tweets_no_urls.txt").readlines())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_list)

    d = tokenizer.word_index
    vocab = []

    w = []
    n = []
    for k, v in d.items():
        w.append(k)
        n.append(v)

    for i in range(len(n)):
        vocab.append(w[n.index(i + 1)])

    seq = tokenizer.texts_to_sequences(text_list)
    text_matrix = tokenizer.texts_to_matrix(text_list, mode="tfidf")

    vocabulary_length = tokenizer.word_index.__len__()
    # vocabulary_length = text_matrix.shape[1]

    # max_seq_len = max(len(x) for x in s)
    # data = pad_sequences(s, padding="post", truncating="post")

    return seq, text_matrix, vocabulary_length, vocab


def load_wiki_data():
    text_list = list(open(data_location + "wiki" + "/tfidf.txt").readlines())
    text_list = [i.split('\t') for i in text_list]
    sequence = []
    vocabulary_length = 4973
    num_void_line = 0
    for num in range(len(text_list)):
        # sequence.append([])
        # vocabulary_length = len(text_list[num]) - 1
        tmp = []
        for i in range(3):
            tmp.append(np.ceil(float(text_list[num][i].strip())))
        # print(tmp
        if int(tmp[0]) != len(sequence) - 1:
            sequence.append([])

        for j in range(int(tmp[0]) - (len(sequence) - 1)):
            # print(int(tmp[0]) - (j + 1)
            num_void_line += 1
            sequence.append([])

        sequence[int(tmp[0])].append(int(tmp[1]))

    print('len(sequence)', len(sequence))
    print('len(num_void_line)', num_void_line)
    # print(sequence[:3]

    return sequence, vocabulary_length, len(sequence)


def load_character_sequence(dataset):
    text_list = list(open(data_location + dataset + "/tweets_no_urls.txt").readlines())

    # character_lib = list()

    # lowercase_alphabet = map(chr, range(97, 123))
    # uppercase_alphabet = map(chr, range(65, 91))
    # numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # special_characters = [' ', ',', '.', '?', '/', ':', ';', '\'', '"', '!', '@', '$', '%', '^', '&', '*', '(', ')', '-', '=']
    #
    # character_lib.extend(lowercase_alphabet)
    # character_lib.extend(uppercase_alphabet)
    # character_lib.extend(numbers)
    # character_lib.extend(special_characters)

    character_lib = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n "

    character_lib_length = len(character_lib)

    seq = list()
    for t in text_list:
        tmp = list()
        for i in t:
            i = str.lower(i)
            tmp.append(character_lib.index(i) + 1)
        seq.append(tmp)

    max_seq_len = max(len(x) for x in seq)

    print(max_seq_len)

    # character_data = pad_sequences(seq, maxlen=max_seq_len)

    return seq, max_seq_len, character_lib_length


def load_data(dataset, log, is_networked):
    if dataset == 'wiki':
        sequence, vocabulary_length = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, vocabulary_length = load_text_data(dataset)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        for idx in range(len(text_list)):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    max_seq_len = max(len(x) for x in sequence)

    # print('max_seq_len: ', max_seq_len
    log.write('\n' + 'max_seq_len: ' + str(max_seq_len))

    data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    linked_nodes = []
    node_pairs = []
    isolated_nodes = []

    if is_networked == 1:
        raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
        raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
        for idx in range(len(raw_node_pairs)):
            n0 = raw_node_pairs[idx][0]
            n1 = raw_node_pairs[idx][1].rstrip('\n')
            node_pairs.append([n0, n1])
            if n0 not in linked_nodes:
                linked_nodes.append(n0)
            if n1 not in linked_nodes:
                linked_nodes.append(n1)
        for node in node_ids:
            if node not in linked_nodes:
                isolated_nodes.append(node)
    else:
        isolated_nodes = list(node_ids)

    network_inf = (np.array(node_ids), node_pairs, linked_nodes, isolated_nodes)

    print("len(node_ids), len(node_pairs), len(linked_nodes), len(isolated_nodes) :")
    print(len(node_ids), len(node_pairs), len(linked_nodes), len(isolated_nodes))

    log.write('\n' + "len(node_ids), len(node_pairs), len(linked_nodes), len(isolated_nodes): ")
    log.write('\n' + str(len(node_ids)) + ' ' + str(len(node_pairs)) + ' '
              + str(len(linked_nodes)) + ' ' + str(len(isolated_nodes)))

    return data, labels, vocabulary_length, network_inf


def load_data1(dataset, log, is_networked, neg_link_ratio):
    if dataset == 'wiki':
        sequence, vocabulary_length = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, vocabulary_length = load_text_data(dataset)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        for idx in range(len(text_list)):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    max_seq_len = max(len(x) for x in sequence)

    # print('max_seq_len: ', max_seq_len
    log.write('\n' + 'max_seq_len: ' + str(max_seq_len))

    data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    linked_nodes = []
    node_pairs = []
    isolated_nodes = []
    num_pos_pairs = 0
    if is_networked == 1:
        raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
        raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
        for idx in range(len(raw_node_pairs)):
            n0 = raw_node_pairs[idx][0]
            n1 = raw_node_pairs[idx][1].rstrip('\n')
            node_pairs.append([n0, n1])
            if n0 not in linked_nodes:
                linked_nodes.append(n0)
            if n1 not in linked_nodes:
                linked_nodes.append(n1)

        num_pos_pairs = len(node_pairs)

        for node in node_ids:
            # np.random.seed(1)
            tmp_node = np.random.choice(node_ids, neg_link_ratio)
            # print('tmp_node: ', tmp_node
            tmp_node_list = tmp_node.tolist()
            for tmp_node in tmp_node_list:
                if ([node, tmp_node] not in node_pairs) & ([tmp_node, node] not in node_pairs) & (node != tmp_node):
                    node_pairs.append([node, tmp_node])
                    if node not in linked_nodes:
                        linked_nodes.append(node)
                    if tmp_node not in linked_nodes:
                        linked_nodes.append(tmp_node)

        for node in node_ids:
            if node not in linked_nodes:
                isolated_nodes.append(node)
    else:
        isolated_nodes = list(node_ids)

    network_inf = (np.array(node_ids), node_pairs, linked_nodes, isolated_nodes, num_pos_pairs)

    print("len(node_ids), len(node_pairs), len(linked_nodes), len(isolated_nodes) :")
    print(len(node_ids), len(node_pairs), len(linked_nodes), len(isolated_nodes))

    log.write('\n' + "len(node_ids), len(node_pairs), len(linked_nodes), len(isolated_nodes): ")
    log.write('\n' + str(len(node_ids)) + ' ' + str(len(node_pairs)) + ' '
              + str(len(linked_nodes)) + ' ' + str(len(isolated_nodes)))

    return data, labels, vocabulary_length, network_inf


def load_data2(dataset, log, is_networked, neg_link_ratio):
    if dataset == 'wiki':
        sequence, vocabulary_length = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, vocabulary_length = load_text_data(dataset)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]

        sequence = []
        vocabulary_length = 1
        for idx in range(len(text_list)):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1

            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    sequence[idx].append(int(word))

    data = np.array(sequence)

    print(data[0], np.sum(data[0]))
    print('len(data[0]): ', len(data[0]))
    print('vocabulary_length: ', vocabulary_length)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    linked_nodes = []
    node_pairs = []
    isolated_nodes = []
    num_pos_pairs = 0
    if is_networked == 1:
        raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
        raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
        for idx in range(len(raw_node_pairs)):
            n0 = raw_node_pairs[idx][0]
            n1 = raw_node_pairs[idx][1].rstrip('\n')
            node_pairs.append([n0, n1])
            if n0 not in linked_nodes:
                linked_nodes.append(n0)
            if n1 not in linked_nodes:
                linked_nodes.append(n1)

        num_pos_pairs = len(node_pairs)

        for node in node_ids:
            # np.random.seed(1)
            tmp_node = np.random.choice(node_ids, neg_link_ratio)
            tmp_node_list = tmp_node.tolist()
            for tmp_node in tmp_node_list:
                if ([node, tmp_node] not in node_pairs) & ([tmp_node, node] not in node_pairs) & (node != tmp_node):
                    node_pairs.append([node, tmp_node])
                    if node not in linked_nodes:
                        linked_nodes.append(node)
                    if tmp_node not in linked_nodes:
                        linked_nodes.append(tmp_node)

        for node in node_ids:
            if node not in linked_nodes:
                isolated_nodes.append(node)
    else:
        isolated_nodes = list(node_ids)

    network_inf = (np.array(node_ids), node_pairs, linked_nodes, isolated_nodes, num_pos_pairs)

    print("len(node_ids), len(node_pairs), len(linked_nodes), len(isolated_nodes) :")
    print(len(node_ids), len(node_pairs), len(linked_nodes), len(isolated_nodes))

    log.write('\n' + "len(node_ids), len(node_pairs), len(linked_nodes), len(isolated_nodes): ")
    log.write('\n' + str(len(node_ids)) + ' ' + str(len(node_pairs)) + ' '
              + str(len(linked_nodes)) + ' ' + str(len(isolated_nodes)))

    return data, labels, vocabulary_length, network_inf


def load_data3(dataset, log):
    if dataset == 'wiki':
        sequence, vocabulary_length = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, vocabulary_length = load_text_data(dataset)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    data_num = len(sequence)

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    node_seq = []
    for idx in range(data_num):
        node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        node_seq[n0].append(n1 + 1)
        node_seq[n1].append(n0 + 1)

    max_seq_len = max(len(x) for x in node_seq)

    print(max_seq_len)

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    data = (text_data, link_data)

    return data, labels, (vocabulary_length, data_num)


def data_transform(dataset):
    sequence, vocabulary_length = load_text_data(dataset)
    # data_num = len(sequence)

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    return text_data, labels, vocabulary_length


def load_cross_domain_data(dataset, log):
    if dataset == 'iphone' or dataset == 'movie':
        text_data1, labels1, vocabulary_length1 = data_transform(dataset)
        text_data2, labels2, vocabulary_length2 = data_transform('foreo')
        return [text_data1, text_data2], [labels1, labels2], [vocabulary_length1, vocabulary_length2]

    elif dataset == 'foreo':
        text_data1, labels1, vocabulary_length1 = data_transform(dataset)
        text_data2, labels2, vocabulary_length2 = data_transform('movie')
        return [text_data1, text_data2], [labels1, labels2], [vocabulary_length1, vocabulary_length2]


def load_data_from_bert(dataset, log):
    # data_num = len(sequence)
    #
    # max_seq_len = max(len(x) for x in sequence)
    #
    # text_data = pad_sequences(sequence, maxlen=max_seq_len)

    text_data = np.load(data_location + dataset + "/input_by_con_emb.npy")
    data_num = text_data.shape[0]

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    node_seq = []
    for idx in range(data_num):
        node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        node_seq[n0].append(n1 + 1)
        node_seq[n1].append(n0 + 1)

    max_seq_len = max(len(x) for x in node_seq)

    print(max_seq_len)

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    data = (text_data, link_data)
    # data = text_data

    return data, labels, data_num


def load_data3_for_sentiment(dataset, log):
    if dataset == 'wiki':
        sequence, vocabulary_length = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        text_list, sequence, vocabulary_length = load_text_data_for_sentiment(dataset)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    data_num = len(sequence)

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    node_seq = []
    for idx in range(data_num):
        node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        node_seq[n0].append(n1 + 1)
        node_seq[n1].append(n0 + 1)

    max_seq_len = max(len(x) for x in node_seq)

    print(max_seq_len)

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    data = (text_list, text_data, link_data)

    return data, labels, (vocabulary_length, data_num)


# it contains word embedding and node embedding based on SVD
def load_data4(dataset, log):
    if dataset == 'wiki':
        sequence, vocabulary_length, data_num = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, text_matrix, vocabulary_length = load_text_data2(dataset)
        data_num = len(text_matrix)
        print(text_matrix.shape)
        U, S, V = sp.linalg.svd(text_matrix.transpose(), full_matrices=False)
        print(U.shape, S.shape, V.shape)
        word_emb = U * S
    else:
        feature_mat = sp.loadtxt(data_location + dataset + "/feature.txt", delimiter='\t')
        print(feature_mat[:3][:3])
        print(feature_mat.shape)
        transformer = TfidfTransformer()
        word_mat_tfidf = transformer.fit_transform(feature_mat.transpose()).toarray()

        U,S,V = sp.linalg.svd(word_mat_tfidf, full_matrices=False)
        word_emb = U*S
        print(word_emb.shape)
        print(word_emb[:3,:5])

        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx])
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i)


    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    node_seq = []
    for idx in range(data_num):
        node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    row_idx = []
    col_idx = []
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        row_idx.append(n0)
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        col_idx.append(n1)
        node_seq[n0].append(n1)
        node_seq[n1].append(n0)

    max_seq_len = max(len(x) for x in node_seq)

    print(max_seq_len)

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    row_idx = np.array(row_idx)
    col_idx = np.array(col_idx)
    values = np.ones_like(row_idx)

    G = coo_matrix((values, (row_idx, col_idx)), shape=(data_num, data_num))
    # G = (G + G * G) / 2.0
    transformer1 = TfidfTransformer()
    G_tfidf = transformer1.fit_transform(G).toarray()

    U1, S1, V1 = sp.linalg.svd(G_tfidf, full_matrices=False)
    node_emb = U1 * S1

    data = (text_data, link_data)

    embedding_weights = (word_emb, node_emb)

    print('vocabulary_length: ', vocabulary_length)

    return data, labels, (vocabulary_length, data_num), embedding_weights


# it adds the links of length 2.
def load_data5(dataset, log):
    if dataset == 'wiki':
        sequence, vocabulary_length, data_num = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, vocabulary_length = load_text_data(dataset)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    node_seq = []
    node_seq_2 = [] # for links of length 2

    for idx in range(data_num):
        node_seq.append([])
        node_seq_2.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        node_seq[n0].append(n1 + 1)
        node_seq[n1].append(n0 + 1)

    for i in range(len(node_seq)):
        tmp = []
        for j in node_seq[i]:
            tmp = tmp + node_seq[j-1]

        for k in tmp:
            if k not in (node_seq[i] + node_seq_2[i]):
                node_seq_2[i].append(k)

    print(node_seq[:5])

    for i in range(len(node_seq)):
        node_seq[i] = node_seq[i] + node_seq_2[i]

    max_seq_len = max(len(x) for x in node_seq)

    print('max_seq_len: ', max_seq_len)
    # print('max_seq_len_2: ', max_seq_len_2

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    data = (text_data, link_data)

    return data, labels, (vocabulary_length, data_num)


# it adds the links of length 2.
def load_data6(dataset, log):
    if dataset == 'wiki':
        sequence, vocabulary_length, data_num = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, vocabulary_length = load_text_data(dataset)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    node_seq = []
    node_seq_2 = [] # for links of length 2

    for idx in range(data_num):
        node_seq.append([])
        node_seq_2.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        node_seq[n0].append(n1 + 1)
        node_seq[n1].append(n0 + 1)

    for i in range(len(node_seq)):
        tmp = []
        for j in node_seq[i]:
            tmp = tmp + node_seq[j-1]

        for k in tmp:
            if k not in (node_seq[i] + node_seq_2[i]):
                node_seq_2[i].append(k)

    print(node_seq[:5])

    max_seq_len = max(len(x) for x in node_seq)
    max_seq_len_2 = max(len(x) for x in node_seq_2)

    print('max_seq_len: ', max_seq_len)
    print('max_seq_len_2: ', max_seq_len_2)

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    link_data_2 = pad_sequences(node_seq_2, maxlen=max_seq_len_2)

    data = (text_data, link_data, link_data_2)

    return data, labels, (vocabulary_length, data_num)

# def load_wiki_data():
#     text_list = list(open(data_location + "wiki" + "/tfidf.txt").readlines())
#     text_list = [i.split('\t') for i in text_list]
#     sequence = []
#     vocabulary_length = 4973
#     num_void_line = 0
#     for num in range(len(text_list)):
#         # sequence.append([])
#         # vocabulary_length = len(text_list[num]) - 1
#         tmp = []
#         for i in range(3):
#             tmp.append(np.ceil(float(text_list[num][i].strip())))
#         # print(tmp
#         if int(tmp[0]) != len(sequence) - 1:
#             sequence.append([])
#
#         for j in range(int(tmp[0]) - (len(sequence) - 1)):
#             # print(int(tmp[0]) - (j + 1)
#             num_void_line += 1
#             sequence.append([])
#
#         for j in range(int(tmp[2])):
#             sequence[int(tmp[0])].append(int(tmp[1]))
#
#     print('len(sequence)', len(sequence)
#     print('len(num_void_line)', num_void_line
#     # print(sequence[:3]
#
#     return sequence, vocabulary_length, len(sequence)


def load_data7(dataset, nb_neighbor, log):
    if dataset == 'wiki':
        sequence, vocabulary_length, data_num = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, vocabulary_length = load_text_data(dataset)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    # sample some words
    for i in range(len(sequence)):
        if nb_neighbor < len(sequence[i]):
            sequence[i] = np.random.choice(sequence[i], nb_neighbor, replace=False).tolist()

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    node_seq = []
    for idx in range(data_num):
        node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        node_seq[n0].append(n1 + 1)
        node_seq[n1].append(n0 + 1)

    # sample some links
    for i in range(len(node_seq)):
        if nb_neighbor < len(node_seq[i]):
            node_seq[i] = np.random.choice(node_seq[i], nb_neighbor, replace=False).tolist()

    max_seq_len = max(len(x) for x in node_seq)

    print(max_seq_len)

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    data = (text_data, link_data)

    return data, labels, (vocabulary_length, data_num)


# for svm
def load_data8(dataset, log):
    if dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        text_list = list(open(data_location + dataset + "/tweets_no_urls.txt").readlines())

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text_list)

        text_matrix = tokenizer.texts_to_matrix(text_list, mode="tfidf")
        vocabulary_length = tokenizer.word_index.__len__()

    data_num = len(text_matrix)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    return text_matrix, labels, vocabulary_length


# for logistic regression
def load_data9(dataset, log):
    if dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        text_list = list(open(data_location + dataset + "/tweets_no_urls.txt", encoding='utf-8').readlines())

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text_list)

        text_matrix = tokenizer.texts_to_matrix(text_list, mode="tfidf")
        vocabulary_length = tokenizer.word_index.__len__()

    data_num = len(text_matrix)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = np.asarray(labels)

    return text_matrix, labels, vocabulary_length


# it contains word embedding and node embedding based on SVD
def load_data10(dataset, log):
    if dataset == 'wiki':
        sequence, vocabulary_length, data_num = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, text_matrix, vocabulary_length, vocab = load_text_data3(dataset)
        data_num = len(text_matrix)
        print(text_matrix.shape)
        print('len(vocab):', len(vocab))
        model = Word2Vec.load(data_location + dataset + "/w2v.bin")
        word_emb = np.array([[model.wv[w] for w in vocab]])
    else:
        feature_mat = sp.loadtxt(data_location + dataset + "/feature.txt", delimiter='\t')
        print(feature_mat[:3][:3])
        print(feature_mat.shape)
        transformer = TfidfTransformer()
        word_mat_tfidf = transformer.fit_transform(feature_mat.transpose()).toarray()

        U,S,V = sp.linalg.svd(word_mat_tfidf, full_matrices=False)
        word_emb = U*S
        print(word_emb.shape)
        print(word_emb[:3,:5])

        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx])
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i)


    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    node_seq = []
    for idx in range(data_num):
        node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    row_idx = []
    col_idx = []
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        row_idx.append(n0)
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        col_idx.append(n1)
        node_seq[n0].append(n1)
        node_seq[n1].append(n0)

    max_seq_len = max(len(x) for x in node_seq)

    print(max_seq_len)

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    row_idx = np.array(row_idx)
    col_idx = np.array(col_idx)
    values = np.ones_like(row_idx)

    G = coo_matrix((values, (row_idx, col_idx)), shape=(data_num, data_num))
    # G = (G + G * G) / 2.0
    transformer1 = TfidfTransformer()
    G_tfidf = transformer1.fit_transform(G).toarray()

    U1, S1, V1 = sp.linalg.svd(G_tfidf, full_matrices=False)
    node_emb = U1 * S1

    data = (text_data, link_data)

    embedding_weights = (word_emb, node_emb)

    print('vocabulary_length: ', vocabulary_length)

    return data, labels, (vocabulary_length, data_num), embedding_weights


def load_data11(dataset, log):
    if dataset == 'wiki':
        sequence, vocabulary_length = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, text_matrix, vocabulary_length, vocab = load_text_data3(dataset)
        data_num = len(text_matrix)
        print(text_matrix.shape)
        print('len(vocab):', len(vocab))
        # model = Word2Vec.load(data_location + dataset + "/w2v.bin")
        
        model = KeyedVectors.load_word2vec_format(data_location + "twitter_200d.txt")

        tmp = [np.random.uniform(-0.25, 0.25, 200).tolist()]

        for w in vocab:
            if w in model.wv.vocab:
                tmp.append(model.wv[w])
            else:
                tmp.append(np.random.uniform(-0.25, 0.25, 200).tolist())

        # tmp.extend([model.wv[w] for w in vocab])

        word_emb = np.array(tmp)
        print(word_emb.shape)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    data_num = len(sequence)

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    node_seq = []
    for idx in range(data_num):
        node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        node_seq[n0].append(n1 + 1)
        node_seq[n1].append(n0 + 1)

    max_seq_len = max(len(x) for x in node_seq)

    print(max_seq_len)

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    data = (text_data, link_data)

    return data, labels, (vocabulary_length, data_num), word_emb


# text matrix and node matrix are combined
def load_data12(dataset, log):
    if dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, text_matrix, vocabulary_length = load_text_data2(dataset)
        data_num = len(text_matrix)
        print(text_matrix.shape)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    # labels = to_categorical(np.asarray(labels))
    labels = np.asarray(labels)

    # node_seq = []
    # for idx in range(data_num):
    #     node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    row_idx = []
    col_idx = []
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        row_idx.append(n0)
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        col_idx.append(n1)
        # node_seq[n0].append(n1)
        # node_seq[n1].append(n0)

    # max_seq_len = max(len(x) for x in node_seq)
    #
    # print(max_seq_len
    #
    # link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    row_idx = np.array(row_idx)
    col_idx = np.array(col_idx)
    values = np.ones_like(row_idx)

    G = coo_matrix((values, (row_idx, col_idx)), shape=(data_num, data_num))
    # G = (G + G * G) / 2.0

    transformer1 = TfidfTransformer()
    G_tfidf = transformer1.fit_transform(G).toarray()

    # U1, S1, V1 = sp.linalg.svd(G_tfidf, full_matrices=False)
    data_low_dim = G_tfidf

    data = np.hstack((text_matrix, data_low_dim))

    print(data.shape)
    print("it is a test")
    # U1, S1, V1 = sp.linalg.svd(data, full_matrices=False)
    # data_low_dim = U1 * np.diag(S1)
    # print(data_low_dim.shape

    return data, labels, vocabulary_length


# text matrix and node matrix
def load_data13(dataset, log):
    if dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, text_matrix, vocabulary_length = load_text_data2(dataset)
        data_num = len(text_matrix)
        print(text_matrix.shape)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))
    labels = np.asarray(labels)

    # node_seq = []
    # for idx in range(data_num):
    #     node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    row_idx = []
    col_idx = []
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        row_idx.append(n0)
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        col_idx.append(n1)
        # node_seq[n0].append(n1)
        # node_seq[n1].append(n0)

    # max_seq_len = max(len(x) for x in node_seq)
    #
    # print(max_seq_len
    #
    # link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    row_idx = np.array(row_idx)
    col_idx = np.array(col_idx)
    values = np.ones_like(row_idx)

    G = coo_matrix((values, (row_idx, col_idx)), shape=(data_num, data_num))
    # G = (G + G * G) / 2.0

    transformer1 = TfidfTransformer()
    G_tfidf = transformer1.fit_transform(G).toarray()

    # U1, S1, V1 = sp.linalg.svd(G_tfidf, full_matrices=False)
    data_low_dim = G_tfidf

    data = (text_matrix, data_low_dim)

    # print(data.shape
    # print("it is a test"
    # U1, S1, V1 = sp.linalg.svd(data, full_matrices=False)
    # data_low_dim = U1 * np.diag(S1)
    # print(data_low_dim.shape

    return data, labels


# one group of data is word sequence, and another group is character sequence.
def load_data14(dataset, log):
    if dataset == 'wiki':
        sequence, vocabulary_length = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        text_list, sequence, vocabulary_length = load_text_data_for_sentiment(dataset)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    data_num = len(sequence)

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    seq, max_seq_len, character_lib_length = load_character_sequence(dataset)

    character_data = pad_sequences(seq, maxlen=max_seq_len)

    data = (text_data, character_data)

    return data, labels, (vocabulary_length, data_num)


def load_data15(dataset, log):
    if dataset == 'wiki':
        sequence, vocabulary_length = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        text_list, sequence, vocabulary_length = load_text_data_for_sentiment(dataset)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    data_num = len(sequence)

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        labels.append(int(label_list[idx][1].rstrip('\n')))

    labels = to_categorical(np.asarray(labels))

    seq, max_seq_len, character_lib_length = load_character_sequence(dataset)

    character_data = pad_sequences(seq, maxlen=max_seq_len)

    data = (character_data, text_data)

    return data, labels, (character_lib_length, data_num)


# load both intention and sentiment labels
def load_data16(dataset, log):
    word_emb = np.array([])

    if dataset == 'wiki':
        sequence, vocabulary_length = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, text_matrix, vocabulary_length, vocab = load_text_data3(dataset)
        data_num = len(text_matrix)
        print(text_matrix.shape)
        print('len(vocab):', len(vocab))
        # model = Word2Vec.load(data_location + dataset + "/w2v.bin")

        model = KeyedVectors.load_word2vec_format(data_location + "twitter_200d.txt")

        tmp = [np.random.uniform(-0.25, 0.25, 200).tolist()]

        for w in vocab:
            if w in model.wv.vocab:
                tmp.append(model.wv[w])
            else:
                tmp.append(np.random.uniform(-0.25, 0.25, 200).tolist())

        # tmp.extend([model.wv[w] for w in vocab])

        word_emb = np.array(tmp)
        print(word_emb.shape)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    data_num = len(sequence)

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    intention_labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        intention_labels.append(int(label_list[idx][1].rstrip('\n')))

    intention_labels = to_categorical(np.asarray(intention_labels))

    sentiment_label_list = list(open(data_location + dataset + "/sentiment_labels.txt").readlines())
    sentiment_label_list = [i.split('\t') for i in sentiment_label_list]
    sentiment_labels = []
    # node_ids = []
    for idx in range(len(sentiment_label_list)):
        # node_ids.append(sentiment_label_list[idx][0])
        sentiment_labels.append(int(sentiment_label_list[idx][1].rstrip('\n')))

    sentiment_labels = to_categorical(np.asarray(sentiment_labels))

    node_seq = []
    for idx in range(data_num):
        node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        node_seq[n0].append(n1 + 1)
        node_seq[n1].append(n0 + 1)

    max_seq_len = max(len(x) for x in node_seq)

    print(max_seq_len)

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    data = (text_data, link_data)

    return data, [intention_labels, sentiment_labels], (vocabulary_length, data_num), word_emb


# return character-level and word-level sequences
def load_data17(dataset, log):
    word_emb = np.array([])

    if dataset == 'wiki':
        sequence, vocabulary_length = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, text_matrix, vocabulary_length, vocab = load_text_data3(dataset)
        data_num = len(text_matrix)
        print(text_matrix.shape)
        print('len(vocab):', len(vocab))
        # model = Word2Vec.load(data_location + dataset + "/w2v.bin")

        model = KeyedVectors.load_word2vec_format(data_location + "twitter_200d.txt")

        tmp = [np.random.uniform(-0.25, 0.25, 200).tolist()]

        for w in vocab:
            if w in model.wv.vocab:
                tmp.append(model.wv[w])
            else:
                tmp.append(np.random.uniform(-0.25, 0.25, 200).tolist())

        # tmp.extend([model.wv[w] for w in vocab])

        word_emb = np.array(tmp)
        print(word_emb.shape)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    data_num = len(sequence)

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    intention_labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        intention_labels.append(int(label_list[idx][1].rstrip('\n')))

    intention_labels = to_categorical(np.asarray(intention_labels))

    node_seq = []
    for idx in range(data_num):
        node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        node_seq[n0].append(n1 + 1)
        node_seq[n1].append(n0 + 1)

    max_seq_len = max(len(x) for x in node_seq)

    print(max_seq_len)

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    seq, max_seq_len, character_lib_length = load_character_sequence(dataset)
    character_data = pad_sequences(seq, maxlen=max_seq_len)

    data = (text_data, character_data)

    return data, intention_labels, (vocabulary_length, character_lib_length), word_emb


# return word and node embedding
def load_data18(dataset, log):
    word_emb = np.array([])
    node_emb = np.array([[]])

    if dataset == 'wiki':
        sequence, vocabulary_length = load_wiki_data()
    elif dataset in ['iphone', 'iphone_large', 'xbox', 'foreo', 'movie']:
        sequence, text_matrix, vocabulary_length, vocab = load_text_data3(dataset)
        data_num = len(text_matrix)
        print(text_matrix.shape)
        print('len(vocab):', len(vocab))
        # model = Word2Vec.load(data_location + dataset + "/w2v.bin")

        u, s, v = sp.linalg.svd(text_matrix, full_matrices=False)
        # print(U.shape, S.shape, V.shape)
        node_emb = u * s

        model = KeyedVectors.load_word2vec_format(data_location + "twitter_200d.txt")

        tmp = [np.random.uniform(-0.25, 0.25, 200).tolist()]

        for w in vocab:
            if w in model.wv.vocab:
                tmp.append(model.wv[w])
            else:
                tmp.append(np.random.uniform(-0.25, 0.25, 200).tolist())

        # tmp.extend([model.wv[w] for w in vocab])

        word_emb = np.array(tmp)
        print(word_emb.shape)
    else:
        text_list = list(open(data_location + dataset + "/feature.txt").readlines())
        text_list = [i.split('\t') for i in text_list]
        sequence = []
        vocabulary_length = 1
        data_num = len(text_list)

        for idx in range(data_num):
            sequence.append([])
            vocabulary_length = len(text_list[idx]) - 1
            for i in range(vocabulary_length):
                word = text_list[idx][i].strip()
                if word != '':
                    if int(word) == 1:
                        sequence[idx].append(i + 1)

    data_num = len(sequence)

    max_seq_len = max(len(x) for x in sequence)

    text_data = pad_sequences(sequence, maxlen=max_seq_len)

    label_list = list(open(data_location + dataset + "/group.txt").readlines())
    label_list = [i.split('\t') for i in label_list]
    intention_labels = []
    node_ids = []
    for idx in range(len(label_list)):
        node_ids.append(label_list[idx][0])
        intention_labels.append(int(label_list[idx][1].rstrip('\n')))

    intention_labels = to_categorical(np.asarray(intention_labels))

    node_seq = []
    for idx in range(data_num):
        node_seq.append([])

    raw_node_pairs = list(open(data_location + dataset + "/graph.txt").readlines())
    raw_node_pairs = [i.split('\t') for i in raw_node_pairs]
    row_idx = []
    col_idx = []
    for idx in range(len(raw_node_pairs)):
        n0 = int(raw_node_pairs[idx][0])
        row_idx.append(n0)
        n1 = int(raw_node_pairs[idx][1].rstrip('\n'))
        col_idx.append(n1)
        node_seq[n0].append(n1+1)
        # node_seq[n1+1].append(n0+1)

    max_seq_len = max(len(x) for x in node_seq)

    print(max_seq_len)

    link_data = pad_sequences(node_seq, maxlen=max_seq_len)

    # row_idx = np.array(row_idx)
    # col_idx = np.array(col_idx)
    # values = np.ones_like(row_idx)
    #
    # g = coo_matrix((values, (row_idx, col_idx)), shape=(data_num, data_num))
    # # G = (G + G * G) / 2.0
    # transformer1 = TfidfTransformer()
    # g_tfidf = transformer1.fit_transform(g).toarray()
    #
    # u1, s1, v1 = sp.linalg.svd(g_tfidf, full_matrices=False)
    # node_emb = u1 * s1

    tmp = np.random.randn(1, data_num)
    node_emb = np.concatenate([tmp, node_emb], axis=0)

    data = (text_data, link_data)

    embedding_weights = (word_emb, node_emb)

    print('vocabulary_length: ', vocabulary_length)

    return data, intention_labels, (vocabulary_length, data_num), embedding_weights