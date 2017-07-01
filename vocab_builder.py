import numpy as np
import pandas as pd
import random
import json
import os
from collections import Counter
from word_cleaner import WordCleaner
import matplotlib.pyplot as plt


class VocabBuilder(object):
    def __init__(self, logger, data_sources, black_list_vocab=None, word_list_path=None,
                 subsample_threshold=1e-5,
                 word_length_threshold=3, tail_word_count_cutoff=10,
                 truncate_most_common=None):

        self.word_cleaner = WordCleaner(black_list_vocab)
        self.logger = logger
        self.data_sources = data_sources
        self.words = []
        self.stop_words = []
        self.word_list_path = word_list_path
        self.tail_word_count_cutoff = tail_word_count_cutoff
        self.word_length_threshold = word_length_threshold
        self.subsample_threshold = subsample_threshold
        self.truncate_most_common = truncate_most_common
        self.configure()

    def configure(self):
        ''' Setup contructor options '''
        if self.word_length_threshold is not None:
            self.logger.warn("Ignoring words shorter than {} characters".format(self.word_length_threshold))

        if self.truncate_most_common is not None and self.subsample_threshold is not None:
            self.logger.warn("sub sampling enabled and truncation is {}. disabling sub sampling".format(
                self.truncate_most_common))
            self.subsample_threshold = None

        loaded_from_file = self.__load_data_set()

        if self.word_list_path is not None and not loaded_from_file:
            self.__dump_to_disk()

        self.__create_lookup_tables()

        self.__filter_vocab()

        if self.subsample_threshold is not None:
            self.__subsample()

    def plot_distribution(self):
        counts = [c[1] for c in Counter(self.words).most_common(1000)]
        plt.hist(counts, bins=100)
        plt.draw()
        # plt.show()

    def __load_data_set(self):
        loaded_from_file = False
        if self.word_list_path is not None:
            if os.path.isfile(self.word_list_path):
                loaded_from_file = True
                with open(self.word_list_path, 'r') as f:
                    self.words = f.read().split()
                # self.words = Counter(np.loadtxt(self.word_list_path, comments="#", delimiter="\n", unpack=False, dtype=str))
            else:
                self.logger.warn("Word list file not found {}. loading from datasources".format(self.word_list_path))
                self.__load_datasources()
        else:
            self.__load_datasources()

        self.vocab = set(self.words)
        self.logger.info("unique words {} total dataset {}".format(len(self.vocab), len(self.words)))
        return loaded_from_file

    def __load_datasources(self):
        def append_words(file_name):
            if not file_name.endswith('.json'):
                raise AssertionError('Data Source file {} is not json'.format(file_name))

            self.logger.info('Loading json from {}'.format(file_name))
            df = pd.read_json(file_name)
            self.logger.debug('\tExtracting words from instructions...')
            self.words += self.__extract_words_from_dataframe(df, extract_keys=('instructions', 'ingredients'))

        for data_source in self.data_sources:
            if os.path.isdir(data_source):
                for root, directory, data_source_files in os.walk(data_source):
                    for ds_file in data_source_files:
                        append_words(os.path.join(root, ds_file))
            else:
                append_words(data_source)

        self.logger.info("Done loading vocabulary. {}".format(len(self.words)))

    def __dump_to_disk(self):
        self.logger.info("Persisting words to disk: {}".format(self.word_list_path))
        with open(self.word_list_path, 'w') as f:
            for w in self.words:
                try:
                    f.write("{}\n".format(w))
                except UnicodeEncodeError:
                    self.logger.info("unrecognized character {}".format(w))
        self.logger.info("\tDone saving vocabulary. at {}".format(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocab.txt')))

    def __extract_words_from_dataframe(self, df, extract_keys):
        words = []
        for index, row in df.transpose().iterrows():
            for extract_key in extract_keys:
                sentences = row[extract_key]
                if isinstance(sentences, str):
                    sentences = sentences.split('\n')
                for sentence in sentences:
                    words += self.word_cleaner.extract_words_from_sentence(sentence)
        return words

    def __filter_vocab(self):
        word_counter = Counter(self.words)
        nb_words = len(self.words)
        train_words = []
        if self.truncate_most_common is not None:
            self.logger.debug("\tFiltering {} most common words".format(self.truncate_most_common))
            most_commons = [tup[0] for tup in word_counter.most_common(self.truncate_most_common)]
            for w, count in word_counter.elements():
                if w in most_commons:
                    train_words.append(w)
        else:
            train_words = [w for w in word_counter.elements() if word_counter[w] > self.tail_word_count_cutoff]
            self.logger.debug("\tRejecting {} words with occurrence < {}".format(nb_words - len(train_words),
                                                                                 self.tail_word_count_cutoff))

        self.logger.debug("\tDone extracting words. kept {} out of {}".format(len(train_words), nb_words))
        return train_words

    def __subsample(self):
        self.logger.info("Subsampling...")
        word_counts = Counter(self.int_words)
        total_count = len(self.int_words)
        freqs = {w: count / total_count for w, count in word_counts.items()}
        p_drop = {w: 1 - np.sqrt(self.subsample_threshold / freqs[w]) for w in word_counts}

        self.int_words = [w for w in self.int_words if random.random() < (1 - p_drop[w])]
        self.logger.info("\tSubsampled {} words down to {}".format(total_count, len(self.int_words)))

    def __create_lookup_tables(self):
        self.logger.info("Creating lookup tables...")
        word_counts = Counter(self.words)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
        self.vocab_to_int = {word: ii for ii, word in self.int_to_vocab.items()}
        self.int_words = [self.vocab_to_int[word] for word in self.words]

        self.lookup = dict()
        self.lookup['int2vocab'] = self.int_to_vocab
        self.lookup['vocab2int'] = self.vocab_to_int
        self.lookup['int_words'] = self.int_words
        self.logger.debug('\tDone creating lookup tables')
