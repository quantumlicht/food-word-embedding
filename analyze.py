import os
import sys
import numpy as np
import time
from collections import Counter

from word_embedding_model import WordEmbeddingModel
from vocab_builder import VocabBuilder
from experiment import Experiment
import logging


def get_experiment_logger(experiment_dir, log_level='INFO'):
    logger = logging.getLogger("")
    logger.setLevel(logging.__dict__[log_level])
    log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(log_format)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(experiment_dir, 'experiment.log'))
    fh.setFormatter(log_format)
    logger.addHandler(fh)
    return logger


def get_target(words, idx, win_size=5):
    """ Get a list of words in a window around an index. """
    r = np.random.randint(1, win_size + 1)
    start = idx - r if (idx - r) > 0 else 0
    stop = idx + r
    target_words = set(words[start:idx] + words[idx + 1:stop + 1])

    return list(target_words)


def n_batches(words, batch_size):
    return len(words) // batch_size


def get_batches(words, batch_size, win_size=5):
    """" Create a generator of word batches as a tuple (inputs, targets) """

    # only full batches
    words = words[:n_batches(words, batch_size) * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for ii in range(len(batch)):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, win_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y


def write_embedding_labels_to_disk(experiment_dir, words):
    metadata_file_path = os.path.join(experiment_dir, 'metadata.tsv')

    word_counts = Counter(words)
    with open(metadata_file_path, 'w') as f:
        f.write("Word\tFrequency\n")
        for word, count in word_counts.most_common():
            f.write("{}\t{}\n".format(word, count))
    return metadata_file_path


def main():
    root = os.path.dirname(os.path.realpath(__file__))
    experiments_folder = "board\\train"
    epochs = 1
    batch_size = 64
    window_size = 10
    save_every = 10000  # Save every x iteration
    log_every = 1000  # Log training stats every x iteration
    eval_similarity_every = 10000  # eval similarity every x iteration

    embedding_size = 100
    truncate_most_common = None  # None for no truncation

    experiment = Experiment(root, experiments_folder)
    os.makedirs(experiment.experiment_dir)

    restore = False  # Restore model from latest checkpoint

    logger = get_experiment_logger(experiment.experiment_dir, log_level='DEBUG')

    with open(os.path.join(root, 'blacklist.txt')) as f:
        blacklist_vocab = [line.strip() for line in f.readlines()]

    data_sources = ['./data']
    vocab_builder = VocabBuilder(logger, data_sources, black_list_vocab=blacklist_vocab,
                                 truncate_most_common=truncate_most_common, word_list_path=root+'/data/words.txt')
    vocab_builder.plot_distribution()

    n_batch = n_batches(vocab_builder.int_words, batch_size)

    logger.info("Losing {} words to obtain {} full batches".format(
        len(vocab_builder.int_words) - n_batch*batch_size,
        n_batch)
    )
    logger.info('Writing {} embedding labels to disk...'.format(len(vocab_builder.words)))
    embedding_metadata_path = write_embedding_labels_to_disk(experiment.experiment_dir, vocab_builder.words)
    logger.info("embedding labels stored at {}".format(embedding_metadata_path))

    logger.info("Generating model...")
    model = WordEmbeddingModel(logger, embedding_size, vocab_builder.lookup,
                               experiment.experiment_dir, embedding_metadata_path)

    if not restore:
        logger.info("Training...")
        iteration = 1
        loss = 0
        for e in range(1, epochs + 1):
            batches = get_batches(vocab_builder.int_words, batch_size, window_size)
            start = time.time()
            for x, y in batches:
                labels = np.array(y)[:, None]
                train_loss = model.run(iteration, x, labels)
                loss += train_loss

                if iteration % log_every == 0:
                    end = time.time()
                    logger.debug("Epoch {}/{}".format(e, epochs) +
                                 " Iteration: {}/{}".format(iteration, n_batch) +
                                 " Avg. Training loss: {:.5f}".format(loss / log_every) +
                                 " {:.5f} sec/batch".format((end - start) / log_every))
                    loss = 0
                    start = time.time()

                if iteration % eval_similarity_every == 0:
                    model.eval_similarity()

                iteration += 1
                if iteration % save_every == 0:
                    model.save(iteration)

        # final save
        logger.info("Training complete")
        model.save(iteration)

        model.extrapolate_relation(direction=('steak', 'pepper'), word_targets=('salmon', 'potatoes'))
    else:
        if experiment.restore_dir is not None:
            model.restore(experiment.last_restore_point, experiment.restore_dir)
            model.extrapolate_relation(direction=('steak', 'pepper'), word_targets=('salmon', 'potatoes'))
        else:
            logger.info("No model to restore")

if __name__ == "__main__":
    main()
