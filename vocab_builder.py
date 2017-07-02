import pandas as pd
import math
import os
from collections import Counter
from word_cleaner import WordCleaner
import matplotlib.pyplot as plt
from word_collection import WordCollection


class VocabBuilder(object):
    def __init__(self, logger, black_list_vocab=None, subsample_threshold=1e-5, tail_word_count_cutoff=10,
                 keep_most_common=None):

        self.word_col = WordCollection()
        self.logger = logger
        self.tail_word_count_cutoff = tail_word_count_cutoff
        self.keep_most_common = keep_most_common
        self.subsample_threshold = subsample_threshold

        if keep_most_common is not None and self.subsample_threshold is not None:
            self.logger.warn("sub sampling enabled and truncation is {}. disabling sub sampling".format(
                self.keep_most_common))
            self.subsample_threshold = None
        self.word_cleaner = WordCleaner(word_length_threshold=3, black_list_vocab=black_list_vocab)

    def get_lookups(self):
        word_counts = self.word_col.as_counter()
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_vocab = {}
        vocab_to_int = {}
        for ii, word in enumerate(sorted_vocab):
            int_to_vocab[ii] = word
            vocab_to_int[word] = ii

        int_words = [vocab_to_int[word] for word in self.word_col.items()]

        lookup = dict()
        lookup['int2vocab'] = int_to_vocab
        lookup['vocab2int'] = vocab_to_int
        lookup['int_words'] = int_words

        return lookup

    def save_word_collection(self, file_path):
        self.logger.info("Persisting words to disk: {}".format(file_path))
        with open(file_path, 'w') as f:
            for w in self.word_col.items():
                try:
                    f.write("{}\n".format(w))
                except UnicodeEncodeError:
                    self.logger.info("unrecognized character {}".format(w))

    def word_count_distribution(self, most_common=1000, plot=True):
        counts = [c[1] for c in Counter(self.word_col.items()).most_common(most_common)]
        if plot:
            plt.hist(counts, bins=100)
            plt.draw()
            # plt.show()

        return counts

    def build_from_word_list(self, file_path):
        if file_path is not None:
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    self.word_col.append(f.read().split())
            else:
                self.logger.warn("Word list file not found {}. loading from datasources".format(file_path))

        self.__normalize_word_collection()

    def build_from_json_sources(self, json_sources, keys):
        def append_words(file_name):
            if not file_name.endswith('.json'):
                raise AssertionError('Data Source file {} is not json'.format(file_name))

            self.logger.info('Loading json from {}'.format(file_name))
            df = pd.read_json(file_name)
            self.logger.debug('\tExtracting words from instructions...')
            self.word_col.append(self.__extract_words_from_dataframe(df, keys=keys))

        for json_source in json_sources:
            if os.path.isdir(json_source):
                for root, directory, data_source_files in os.walk(json_source):
                    for ds_file in data_source_files:
                        append_words(os.path.join(root, ds_file))
            else:
                append_words(json_source)

        self.__normalize_word_collection()

    '''
    Make sure the word_collection is cleaned up (downsampled, and infrequent words are cutoff
    '''
    def __normalize_word_collection(self):
        self.word_col.keep_most_common(keep_most_common=self.keep_most_common,
                                       tail_word_count_cutoff=self.tail_word_count_cutoff)
        self.word_col.subsample(self.subsample_threshold)

        self.logger.info("unique words {} total dataset {}".format(
            len(self.word_col.vocab()), len(self.word_col.items())))

    '''
    Retrieve word list from dataframe
    '''
    def __extract_words_from_dataframe(self, df, keys):
        words = []
        for index, row in df.transpose().iterrows():
            for extract_key in keys:
                sentences = row[extract_key]
                try:
                    if isinstance(sentences, float):
                        continue  # weird pandas thing that produces nan

                    if sentences is None:
                        continue

                    if isinstance(sentences, str):
                        sentences = sentences.split('\n')

                    for sentence in sentences:
                        words += self.word_cleaner.extract_words_from_sentence(sentence)
                except Exception as e:
                    self.logger.error(e)
        return words
