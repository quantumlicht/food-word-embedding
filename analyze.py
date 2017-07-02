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


def get_batches(words, batch_size, win_size=5):
    """" Create a generator of word batches as a tuple (inputs, targets) """

    def get_target(word_list, index, window_size=5):
        """ Get a list of words in a window around an index. """
        r = np.random.randint(1, window_size + 1)
        start = index - r if (index - r) > 0 else 0
        stop = index + r
        target_words = set(word_list[start:idx] + word_list[index + 1:stop + 1])

        return list(target_words)

    # only full batches
    n_batches = len(words) // batch_size
    words = words[:n_batches * batch_size]

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
    # batching params
    batch_size = 64
    window_size = 10

    # model params
    epochs = 1
    embedding_size = 100
    learning_rate = 0.001
    visualize_embedding = False

    # runner params
    save_every = 10000  # Save every x iteration
    log_every = 1000  # Log training stats every x iteration
    eval_similarity_every = 10000  # eval similarity every x iteration
    restore = True  # Restore model from latest checkpoint

    # Word embedding params
    word_list_path = './data/words.txt'
    keep_most_common = None  # None for no truncation

    # #########################################################
    experiment = Experiment(root, experiments_folder)
    os.makedirs(experiment.dir)

    logger = get_experiment_logger(experiment.dir, log_level='DEBUG')

    with open(os.path.join(root, 'blacklist.txt')) as f:
        blacklist_vocab = [line.strip() for line in f.readlines()]

    vocab_builder = VocabBuilder(logger, black_list_vocab=blacklist_vocab, keep_most_common=keep_most_common)

    if restore or os.path.isfile(word_list_path):
        vocab_builder.build_from_word_list(word_list_path)
    else:
        vocab_builder.build_from_json_sources(['./data'], keys=('instructions', 'ingredients'))
        vocab_builder.save_word_collection(word_list_path)

    vocab_builder.word_count_distribution()
    lookups = vocab_builder.get_lookups()

    n_batch = len(lookups['int_words']) // batch_size

    logger.info("Losing {} words to obtain {} full batches".format(
        len(lookups['int_words']) - n_batch*batch_size,
        n_batch)
    )

    logger.info("Generating model...")
    model = WordEmbeddingModel(logger, embedding_size, lookups,
                               experiment.dir)
    if visualize_embedding:
        logger.info('Writing {} embedding labels to disk...'.format(len(vocab_builder.word_col.items())))
        embedding_metadata_path = write_embedding_labels_to_disk(experiment.dir, vocab_builder.word_col.items())
        logger.info("\tembedding labels stored at {}".format(embedding_metadata_path))
        model.init_visualizer(os.path.join(embedding_metadata_path))

    if not restore:
        logger.info("Training...")
        iteration = 1
        loss = 0
        for e in range(1, epochs + 1):
            batches = get_batches(lookups['int_words'], batch_size, window_size)
            start = time.time()
            for x, y in batches:
                labels = np.array(y)[:, None]
                train_loss = model.run(iteration, x, labels, learning_rate)
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
        model.extrapolate_relation(direction=('steak', 'pepper'), word_targets=('salmon', 'potato'))
    else:
        if experiment.restore_dir is not None:
            model.restore(experiment.last_restore_point, experiment.restore_dir)
            model.extrapolate_relation(direction=('steak', 'pepper'), word_targets=('salmon', 'potato', 'banana'))
        else:
            logger.info("No model to restore")

if __name__ == "__main__":
    main()
