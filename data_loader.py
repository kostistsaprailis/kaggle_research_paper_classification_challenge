import re
import os
import numpy as np
import pandas as pd
import networkx as nx

from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import normalize

np.random.seed(42)

BASE_DIR = 'data/'
EMBEDDING_DIM = 300
MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 240

class DataLoader(object):

    def __init__(self):
        """ Contructor"""
        self.embeddings_index = {}
        self.labels = None
        with open(os.path.join(BASE_DIR, 'glove.42B.300d.txt')) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
        print('Found %s word vectors.' % len(self.embeddings_index))
    
    def to_one_hot(self, labels, dimension=28):
        if not self.labels:
            self.labels = {}
        results = np.zeros((len(labels), dimension))
        num_label = 0
        for i, label in enumerate(labels):
            if label in self.labels:
                results[i, self.labels[label]] = 1.
            else:
                self.labels[label] = num_label
                results[i, self.labels[label]] = 1.
                num_label += 1
        return results

    def load_ids(self, file):
        # Read training data
        ids = []
        y = []
        with open(os.path.join(BASE_DIR, file), 'r') as f:
            next(f)
            for line in f:
                t = line.split(',')
                ids.append(t[0])
                y.append(t[1][:-1])
        return ids, y

    def create_graph(self, ids, all_ids):
        ret_seq = []
        for i in range(len(ids)):
            nodes = list(self.G.neighbors(ids[i]))
            seq = []
            for node in nodes:
                if node in all_ids:
                    seq.append(self.nodes_dict[node])
            ret_seq.append(seq)
        return ret_seq

    def clean_str(self, string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
        string = re.sub(r"\'s", " \'s", string) 
        string = re.sub(r"\'ve", " \'ve", string) 
        string = re.sub(r"n\'t", " n\'t", string) 
        string = re.sub(r"\'re", " \'re", string) 
        string = re.sub(r"\'d", " \'d", string) 
        string = re.sub(r"\'ll", " \'ll", string) 
        string = re.sub(r",", " , ", string) 
        string = re.sub(r"!", " ! ", string) 
        string = re.sub(r"\(", " \( ", string) 
        string = re.sub(r"\)", " \) ", string) 
        string = re.sub(r"\?", " \? ", string) 
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip()

    def load_data(self):
        # Load data about each article in a dataframe
        df = pd.read_csv(os.path.join(BASE_DIR, "node_information.csv")).fillna('')
        print(df.head())

        train_ids, y_train = self.load_ids("train.csv")
        validation_ids, y_val = self.load_ids("validation.csv")

        # Read test data
        test_ids = []
        with open(os.path.join(BASE_DIR, 'test.csv'), 'r') as f:
            next(f)
            for line in f:
                test_ids.append(line[:-2])

        n_train = len(train_ids)
        n_validation = len(validation_ids)
        n_test = len(test_ids)
        unique = np.unique(y_train)
        print("\nNumber of classes in train data: ", unique.size)
        unique = np.unique(y_val)
        print("\nNumber of classes in validation data: ", unique.size)

        train_authors = []
        validation_authors = []
        test_authors = []

        # Extract the authors, title and abstract of each training article from the dataframe
        train_text = []
        for i in train_ids:
            # train_text.append(df.loc[df['id'] == int(i)]['authors'].iloc[0] + ' ' + df.loc[df['id'] == int(i)]['title'].iloc[0] + ' ' + df.loc[df['id'] == int(i)]['abstract'].iloc[0])
            train_text.append(self.clean_str(df.loc[df['id'] == int(i)]['title'].iloc[0]) + ' ' + self.clean_str(df.loc[df['id'] == int(i)]['abstract'].iloc[0]))
            train_authors.append(df.loc[df['id'] == int(i)]['authors'].iloc[0])

        # Extract the authors, title and abstract of each validation article from the dataframe
        validation_text = []
        for i in validation_ids:
            validation_text.append(self.clean_str(df.loc[df['id'] == int(i)]['title'].iloc[0]) + ' ' + self.clean_str(df.loc[df['id'] == int(i)]['abstract'].iloc[0]))
            validation_authors.append(df.loc[df['id'] == int(i)]['authors'].iloc[0])

        # Extract the authors, title and abstract of each test article from the dataframe
        test_text = []
        for i in test_ids:
            test_text.append(self.clean_str(df.loc[df['id'] == int(i)]['title'].iloc[0]) + ' ' + self.clean_str(df.loc[df['id'] == int(i)]['abstract'].iloc[0]))
            test_authors.append(df.loc[df['id'] == int(i)]['authors'].iloc[0])


        temp = []
        for sentence in train_text:
            temp_sent = []
            for word in sentence.split():
                if word not in stopwords.words('english'):
                    temp_sent.append(word)
            temp.append(" ".join(temp_sent))
        train_text = temp

        temp = []
        for sentence in validation_text:
            temp_sent = []
            for word in sentence.split():
                if word not in stopwords.words('english'):
                    temp_sent.append(word)
            temp.append(" ".join(temp_sent))
        validation_text = temp

        temp = []
        for sentence in test_text:
            temp_sent = []
            for word in sentence.split():
                if word not in stopwords.words('english'):
                    temp_sent.append(word)
            temp.append(" ".join(temp_sent))
        test_text = temp


        # Create a directed graph
        self.G = nx.read_edgelist(os.path.join(BASE_DIR, 'Cit-HepTh.txt'), delimiter='\t', create_using=nx.DiGraph())

        print("Nodes: ", self.G.number_of_nodes())
        print("Edges: ", self.G.number_of_edges())

        self.nodes_dict = {}
        index = 0
        for node in self.G.nodes():
            self.nodes_dict[node] = index
            index += 1

        all_ids = []
        all_ids.extend(train_ids)
        all_ids.extend(validation_ids)
        all_ids.extend(test_ids)

        print("Max graph neight")

        print('Total articles:', len(self.nodes_dict.items()))

        y_train = self.to_one_hot(y_train)
        y_val = self.to_one_hot(y_val)

        full_text = train_text[:]
        full_text.extend(validation_text)
        full_text.extend(test_text)
        self.tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, char_level=False)
        self.tokenizer.fit_on_texts(full_text)
        sequences = self.tokenizer.texts_to_sequences(train_text)
        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        max_text_seq = 0
        for seq in sequences:
            if len(seq) > max_text_seq:
                max_text_seq = len(seq)
        train_text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, value=0.0)

        # indices = np.arange(train_text.shape[0])
        # np.random.shuffle(indices)
        # train_text = train_text[indices]
        # train_graph = train_graph[indices]
        # y_train = y_train[indices]

        sequences2 = self.tokenizer.texts_to_sequences(validation_text)
        word_index2 = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index2))
        validation_text = pad_sequences(sequences2, maxlen=MAX_SEQUENCE_LENGTH, value=0.0)
        for seq in sequences2:
            if len(seq) > max_text_seq:
                max_text_seq = len(seq)

        # indices = np.arange(validation_text.shape[0])
        # np.random.shuffle(indices)
        # validation_text = validation_text[indices]
        # validation_graph = validation_graph[indices]
        # y_val = y_val[indices]

        sequences3 = self.tokenizer.texts_to_sequences(test_text)
        word_index3 = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index3))
        test_text = pad_sequences(sequences3, maxlen=MAX_SEQUENCE_LENGTH, value=0.0)
        for seq in sequences3:
            if len(seq) > max_text_seq:
                max_text_seq = len(seq)

        print("Max_text_sequence:", max_text_seq)

        num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        authors = {'': 0}
        train_auth = []
        val_auth = []
        test_auth = []
        num_auths = 1
        max_auths = 0
        for i, auth in enumerate(train_authors):
            auths = auth.split(',')
            if len(auths) > max_auths:
                max_auths = len(auths)
            aus = []
            for a in auths:
                if a not in authors:
                    authors[a] = num_auths
                    num_auths += 1
                aus.append(authors[a])
            train_auth.append(aus)

        for i, auth in enumerate(validation_authors):
            auths = auth.split(',')
            if len(auths) > max_auths:
                max_auths = len(auths)
            aus = []
            for a in auths:
                if a not in authors:
                    authors[a] = num_auths
                    num_auths += 1
                aus.append(authors[a])
            val_auth.append(aus)

        for i, auth in enumerate(test_authors):
            auths = auth.split(',')
            if len(auths) > max_auths:
                max_auths = len(auths)
            aus = []
            for a in auths:
                if a not in authors:
                    authors[a] = num_auths
                    num_auths += 1
                aus.append(authors[a])
            test_auth.append(aus)
 
        print(max_auths)
        print(len(authors.items()))

        train_auth = np.array(pad_sequences(train_auth, padding='post', maxlen=max_auths, value=0.0))
        val_auth = np.array(pad_sequences(val_auth, padding='post', maxlen=max_auths, value=0.0))
        test_auth = np.array(pad_sequences(test_auth, padding='post', maxlen=max_auths, value=0.0))

        avg_neig_deg = nx.average_neighbor_degree(self.G, nodes=train_ids)
        train_graph_extra = np.zeros((n_train, 3))
        for i in range(n_train):
            train_graph_extra[i,0] = self.G.out_degree(train_ids[i])
            train_graph_extra[i,1] = self.G.in_degree(train_ids[i])
            train_graph_extra[i,2] = avg_neig_deg[train_ids[i]]
        train_graph_extra = normalize(train_graph_extra)

        avg_neig_deg = nx.average_neighbor_degree(self.G, nodes=validation_ids)
        validation_graph_extra = np.zeros((n_validation, 3))
        for i in range(n_validation):
            validation_graph_extra[i,0] = self.G.out_degree(validation_ids[i])
            validation_graph_extra[i,1] = self.G.in_degree(validation_ids[i])
            validation_graph_extra[i,2] = avg_neig_deg[validation_ids[i]]
        validation_graph_extra = normalize(validation_graph_extra)

        avg_neig_deg = nx.average_neighbor_degree(self.G, nodes=test_ids)
        test_graph_extra = np.zeros((n_train, 3))
        for i in range(n_test):
            test_graph_extra[i,0] = self.G.out_degree(test_ids[i])
            test_graph_extra[i,1] = self.G.in_degree(test_ids[i])
            test_graph_extra[i,2] = avg_neig_deg[test_ids[i]]
        test_graph_extra = normalize(test_graph_extra)

        train_graph = self.create_graph(train_ids, all_ids)
        validation_graph = self.create_graph(validation_ids, all_ids)
        test_graph = self.create_graph(test_ids, all_ids)

        max_seq = max(
            max([len(seq) for seq in train_graph]),
            max([len(seq) for seq in validation_graph]),
            max([len(seq) for seq in test_graph]),
        )

        print('Max Graph in/out degree:', max_seq)

        train_graph = np.array(pad_sequences(train_graph, padding='post', maxlen=max_seq, value=0.0))
        validation_graph = np.array(pad_sequences(validation_graph, padding='post', maxlen=max_seq, value=0.0))
        test_graph = np.array(pad_sequences(test_graph, padding='post', maxlen=max_seq, value=0.0))

        walk_index = {}
        with open(os.path.join(BASE_DIR, 'walk.embeddings')) as f:
            next(f)
            for line in f:
                values = line.split()
                node = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                walk_index[node] = coefs
        print('Found %s walk vectors.' % len(walk_index))
        train_graph_walk = np.zeros((n_train, 64))
        for i in range(n_train):
            train_graph_walk[i] = walk_index[train_ids[i]]
        validation_graph_walk = np.zeros((n_validation, 64))
        for i in range(n_validation):
            validation_graph_walk[i] = walk_index[validation_ids[i]]
        test_graph_walk = np.zeros((n_test, 64))
        for i in range(n_test):
            test_graph_walk[i] = walk_index[test_ids[i]]

        np.savez(os.path.join('npz', 'train.npz'),
                train_text=train_text,
                train_graph=train_graph,
                train_authors=train_auth,
                train_graph_extra=train_graph_extra,
                train_graph_walk=train_graph_walk,
                y_train=y_train,
                train_ids=train_ids)
        np.savez(os.path.join('npz', 'validation.npz'),
                validation_text=validation_text,
                validation_graph=validation_graph,
                validation_authors=val_auth,
                validation_graph_extra=validation_graph_extra,
                validation_graph_walk=validation_graph_walk,
                y_val=y_val,
                validation_ids=validation_ids)
        np.savez(os.path.join('npz', 'test.npz'),
                test_text=test_text,
                test_graph=test_graph,
                test_authors=test_auth,
                test_graph_extra=test_graph_extra,
                test_graph_walk=test_graph_walk,
                test_ids=np.array(test_ids))
        np.savez(os.path.join('npz', 'misc.npz'),
                embedding_matrix=embedding_matrix,
                labels=np.array(self.labels),
                num_words=np.array(num_words),
                max_num_authors=max_auths,
                num_authors=len(authors.items()))


if __name__ == '__main__':
    data_loader = DataLoader()
    data_loader.load_data()