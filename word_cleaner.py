from nltk.stem import WordNetLemmatizer
import re


class WordCleaner(object):

    def __init__(self, word_length_threshold=3, black_list_vocab=None, truncate_most_common=None):
        self.lemmatizer = WordNetLemmatizer()
        self.black_list_vocab = black_list_vocab
        self.word_length_threshold = word_length_threshold
        self.word_regex = r"[a-zA-ZèÈéÉâÂàÀôÔöÖçÇäÄ']+"

    def extract_words_from_sentence(self, sentence):
        words = []
        for word in re.findall(self.word_regex, sentence):
            word = WordCleaner._standardize(word)
            word = self._singularize(word)
            if word.isdigit() or word == '':
                continue
            if len(word) < self.word_length_threshold:
                continue
            if self.black_list_vocab is not None and word in self.black_list_vocab:
                continue
            words.append(word)
        return words

    def _singularize(self, word):
        return self.lemmatizer.lemmatize(word, 'n')

    @staticmethod
    def _standardize(w):
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
