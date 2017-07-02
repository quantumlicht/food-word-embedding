from collections import Counter
import random
import numpy as np

class WordCollection:
    def __init__(self, word_list=[]):
        self._words = word_list

    def vocab(self):
        return set(self._words)

    def append(self, word_list):
        self._words += word_list

    def items(self):
        return self._words

    def as_counter(self):
        return Counter(self._words)

    def subsample(self, subsample_threshold=None):
        if subsample_threshold is None:
            return

        word_counts = Counter(self._words)
        total_count = len(self._words)
        freqs = {w: count / total_count for w, count in word_counts.items()}
        p_drop = {w: 1 - np.sqrt(subsample_threshold / freqs[w]) for w in word_counts}

        self._words = [w for w in self._words if random.random() < (1 - p_drop[w])]

    def keep_most_common(self, keep_most_common=None, tail_word_count_cutoff=10):
        word_counter = Counter(self._words)
        train_words = []
        if keep_most_common is not None:
            most_commons = [tup[0] for tup in word_counter.most_common(keep_most_common)]
            for w, count in self._words:
                if w in most_commons:
                    train_words.append(w)
        else:
            train_words = [w for w in self._words if word_counter[w] > tail_word_count_cutoff]

        self._words = train_words
