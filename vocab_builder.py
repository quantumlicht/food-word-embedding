import pandas as pd
import numpy as np
import random
from collections import Counter
import re
import os
import matplotlib.pyplot as plt
# from nltk.stem import WordNetLemmatizer

# class WordCleaner(object):
#     def __init__(self):
#         self.wnl = WordNetLemmatizer()
#
#     def clean(self, word):
#         pass
#
#     def __is_plural(self, word):
#         lemma = self.wnl.lemmatize(word, 'n')
#         return word is not lemma


class VocabBuilder(object):
    # TODO Stem words to remove pluralizations

    def __init__(self, logger, json_data_sources, black_list_vocab=None, dump_path=None, word_list_path=None,
                 subsample_threshold=1e-5,
                 word_length_threshold=3, tail_word_count_cutoff=10,
                 truncate_most_common=None):
        self.logger = logger
        self.json_data_sources = json_data_sources
        self.words = []

        self.dump_path = dump_path
        self.word_list_path = word_list_path

        self.tail_word_count_cutoff = tail_word_count_cutoff

        self.word_length_threshold = word_length_threshold
        if self.word_length_threshold is not None:
            logger.warn("Ignoring words shorter than {} characters".format(self.word_length_threshold))

        self.black_list_vocab = black_list_vocab
        if self.black_list_vocab is not None:
            self.logger.debug("loading black list from {}".format(self.black_list_vocab))
            self.stop_words = self.__load_stop_words()
        else:
            self.stop_words = []

        self.subsample_threshold = subsample_threshold
        self.truncate_most_common = truncate_most_common

        if self.truncate_most_common is not None and self.subsample_threshold is not None:
            logger.warn("subsampling enabled and truncation is {}. disabling subsampling".format(
                self.truncate_most_common))
            self.subsample_threshold = None

        self.__load_dataset()

        if self.dump_path is not None:
            self.__dump_to_disk()

    def plot_distribution(self):
        counts = [c[1] for c in Counter(self.words).most_common(1000)]
        plt.hist(counts, bins=100)
        plt.show()

    def __load_stop_words(self):
        with open(self.black_list_vocab) as f:
            return [line.strip() for line in f.readlines()]

    def __load_dataset(self):
        if self.word_list_path is not None:
            if os.path.isfile(self.word_list_path):
                self.words = Counter(np.loadtxt(self.word_list_path, comments="#", delimiter="\n", unpack=False))
            else:
                raise FileNotFoundError(self.word_list_path)
        else:
            self.__load_from_json()

        self.vocab = set(self.words)
        self.logger.info("unique words {} total dataset {}".format(len(self.vocab), len(self.words)))
        self.__create_lookup_tables()

        if self.subsample_threshold is not None:
            self.__subsample()

    def __load_from_json(self):
        for data_source in self.json_data_sources:
            self.logger.info('Loading dataframe from {}'.format(data_source))
            data_frame = pd.read_json(data_source)
            self.logger.debug('\tExtracting words...')
            self.words  += self.__extract_words(data_frame)

        self.logger.info("Done loading vocabulary. {}".format(len(self.words)))

    def __dump_to_disk(self):
        self.logger.info("Persisting words to disk: {}".format(self.dump_path))
        with open(self.dump_path, 'w') as f:
            for w in self.words:
                try:
                    f.write("{}\n".format(w))
                except UnicodeEncodeError:
                    self.logger.info("unrecognized character {}".format(w))
        self.logger.info("\tDone saving vocabulary. at {}".format(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocab.txt')))

    def __extract_words(self, df):
        regexp = r"[a-zA-ZèÈéÉâÂàÀôÔöÖçÇäÄ']+"
        word_counts = Counter()
        for instruction in df.loc['instructions']:
            if isinstance(instruction, str):
                for word in re.findall(regexp, instruction):
                    word = VocabBuilder.standardize(word)
                    if word.isdigit() or word == '':
                        continue
                    if len(word) < self.word_length_threshold:
                        continue
                    if self.black_list_vocab is not None and word in self.stop_words:
                        continue
                    word_counts[word] += 1
        return self.__filter_vocab(word_counts)

    def __filter_vocab(self, word_counts):
        nb_words = sum(word_counts.values())

        self.logger.debug("\tFound {} words in dataset".format(nb_words))
        train_words = []
        if self.truncate_most_common is not None:
            self.logger.debug("\tFiltering {} most common words".format(self.truncate_most_common))
            most_commons = sorted(word_counts, key=word_counts.get, reverse=True)[:self.truncate_most_common]
            for w in word_counts.elements():
                if w in most_commons:
                    train_words.append(w)
        else:
            train_words = [w for w in word_counts.elements() if word_counts[w] > self.tail_word_count_cutoff]
            self.logger.debug("\tRejecting {} words with occurrence < {}".format(nb_words - len(train_words),
                                                                                 self.tail_word_count_cutoff))

        self.logger.debug("\tDone extracting words. kept {} out of {}".format(len(train_words), nb_words))
        return train_words

    @staticmethod
    def standardize(w):
        w = w.replace('\ufb01', 'fi')
        w = w.replace('\ufb02', 'fl')
        w = w.replace('\u2153', '1/3')
        w = w.replace('\u2154', '2/3')
        w = w.replace('\u215b', '1/8')
        w = w.replace('\u00E9', 'e')
        w = w.replace('\u00E8', 'e')
        w = w.replace('\u00C8', 'E')
        w = w.replace('\u00C9', 'E')
        w = w.replace('\u00E6', 'ae')
        w = w.replace('\u00E0', 'a')
        w = w.replace('\u00C0', 'A')
        w = w.replace('\u00E0', 'a')
        w = w.replace('\u00E2', 'a')
        w = w.replace('\u00C2', 'A')
        w = w.replace('\u00E4', 'a')
        w = w.replace('\u00C4', 'A')
        w = w.replace('\u00E7', 'c')
        w = w.replace('\u00C7', 'C')
        w = w.replace('\u00F4', 'o')
        w = w.replace('\u00D4', 'O')
        w = w.replace('\u00F6', 'o')
        w = w.replace('\u00D6', 'O')
        w = w.replace('\u0153', 'oe')
        w = w.replace('\u0152', 'OE')
        w = w.replace('\'', '')
        return w.lower()

    def __subsample(self):
        self.logger.info("subsampling...")
        word_counts = Counter(self.int_words)
        total_count = len(self.int_words)
        freqs = {w: count / total_count for w, count in word_counts.items()}
        p_drop = {w: 1 - np.sqrt(self.subsample_threshold / freqs[w]) for w in word_counts}

        self.int_words = [w for w in self.int_words if random.random() < (1 - p_drop[w])]
        self.logger.info("\tsubsampled {} words down to {}".format(total_count, len(self.int_words)))

    def __create_lookup_tables(self):
        self.logger.info("creating lookup tables...")
        word_counts = Counter(self.words)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
        self.vocab_to_int = {word: ii for ii, word in self.int_to_vocab.items()}
        self.int_words = [self.vocab_to_int[word] for word in self.words]

        self.lookup = dict()
        self.lookup['int2vocab'] = self.int_to_vocab
        self.lookup['vocab2int'] = self.vocab_to_int
        self.lookup['int_words'] = self.int_words
        self.logger.info('\tDone creating lookup tables')
