import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class WordEmbeddingModel:
    def __init__(self, logger, embedding_size, lookup, writer_dir, metadata_path):
        self.n_sampled = 100
        self.validation_set_size = 8

        self.logger = logger
        self.metadata_path = metadata_path
        self.embedding_size = embedding_size
        self.vocab2int = lookup['vocab2int']
        self.int2vocab = lookup['int2vocab']
        self.n_vocab = len(self.vocab2int)
        self.writer_dir = writer_dir

        self.validation_examples = self.__get_validation_examples()
        self.inputs, self.labels, self.cost, self.optimizer, self.embedding, self.learn_rate = self.__get_training_ops()
        self.normalized_embedding, self.similarity, self.validation_embedding = self.__get_similarity_ops()

        self.sess = tf.Session()

        self.saver = tf.train.Saver()
        self.merged = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.writer_dir, self.sess.graph)
        self.__setup_visualizer()  # should be after all variables are created
        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def __variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def visualize_tsne(self, nb_words, save_path=None):
        tsne = TSNE()
        embedding = self.sess.run(self.normalized_embedding)
        embed_tsne = tsne.fit_transform(embedding[:nb_words, :])

        fig, ax = plt.subplots(figsize=(14, 14))
        for idx in range(nb_words):
            plt.scatter(*embed_tsne[idx, :], color='steelblue')
            plt.annotate(self.int2vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

        if save_path is not None:
            fig.savefig(save_path)

    def __setup_visualizer(self):
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.normalized_embedding.name

        embedding.metadata_path = self.metadata_path
        projector.visualize_embeddings(self.summary_writer, config)

    def run(self, iteration, x, y, learning_rate):
        feed = {self.inputs: x,
                self.labels: y,
                self.learn_rate: learning_rate
                }
        train_loss, _, summary = self.sess.run([self.cost, self.optimizer, self.merged], feed_dict=feed)
        self.summary_writer.add_summary(summary, iteration)
        return train_loss

    def save(self, iteration):
        self.logger.info("saving model to {}".format(self.writer_dir))
        return self.saver.save(self.sess, os.path.join(self.writer_dir, "model.ckpt"), iteration)

    def restore(self, restore_file, restore_dir):
        self.logger.info("importing graph from {}".format(restore_file))
        self.saver = tf.train.import_meta_graph(restore_file)
        self.logger.info("restoring model from {}".format(restore_dir))
        self.saver.restore(self.sess, tf.train.latest_checkpoint(restore_dir))

    def eval_similarity(self, n_neighbors=8):
        self.__run_similarity(self.similarity, self.validation_examples, self.validation_set_size, n_neighbors)

    def __run_similarity(self, similarity, examples, size, n_neighbors=8):
        sim = self.sess.run(similarity)
        for i in range(size):
            valid_word = self.int2vocab[examples[i]]
            top_k = n_neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = self.int2vocab[nearest[k]]
                log = '%s %s,' % (log, close_word)
            self.logger.debug(log)

    def extrapolate_relation(self, direction, word_targets):
        relation_from_word_code = self.vocab2int[direction[0]]
        relation_to_word_code = self.vocab2int[direction[1]]
        target_word_codes = [self.vocab2int[target_word] for target_word in word_targets]

        embed_mat = self.sess.run(self.normalized_embedding)
        self.logger.info("embedding dimensions {}".format(embed_mat.shape))

        relations_embedding = tf.nn.embedding_lookup(
            self.normalized_embedding, np.array([relation_from_word_code, relation_to_word_code] + target_word_codes))

        relation_from_vec = relations_embedding[0]
        relation_to_vec = relations_embedding[1]
        target_vecs = relations_embedding[2:]
        concept_vec = relation_to_vec - relation_from_vec

        targets = []
        for i in range(len(word_targets)):
            targets.append(concept_vec + target_vecs[i])

        cosine_similarity = tf.matmul(targets, tf.transpose(self.normalized_embedding))

        # Add special target words in vocab
        last_target_code = -1
        for target in word_targets:
                self.int2vocab[last_target_code] = '{} - {} -> {}'.format(
                    direction[0], direction[1], target)
                last_target_code -= 1

        self.__run_similarity(cosine_similarity, list(range(-1, last_target_code, -1)), len(targets), n_neighbors=16)

    def __get_training_ops(self):
        with tf.name_scope('learning_rate'):
            learning_rate = tf.placeholder(tf.float32)

        with tf.name_scope('inputs'):
            inputs = tf.placeholder(tf.int32, [None], name='inputs')
        with tf.name_scope('labels'):
            labels = tf.placeholder(tf.int32, [None, None], name='labels')

        embedding = tf.Variable(tf.random_uniform((self.n_vocab, self.embedding_size), -1, 1), name='embedding')
        embed = tf.nn.embedding_lookup(embedding, inputs, name='lookup')

        with tf.name_scope('weights'):
            softmax_w = tf.Variable(tf.truncated_normal((self.n_vocab, self.embedding_size), stddev=0.1),
                                    name='weight')
            WordEmbeddingModel.__variable_summaries(softmax_w)

        with tf.name_scope('biases'):
            softmax_b = tf.Variable(tf.zeros(self.n_vocab), name='bias')
            WordEmbeddingModel.__variable_summaries(softmax_b)

        with tf.name_scope('loss'):
            # Calculate the loss using negative sampling
            # TODO: Compare results with NCE loss
            loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b,
                                              labels, embed,
                                              self.n_sampled, self.n_vocab, name='loss')
            cost = tf.reduce_mean(loss)
        tf.summary.scalar('cost', cost)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(cost)

        return inputs, labels, cost, optimizer, embedding, learning_rate

    def __get_validation_examples(self):
        validation_sample_window_size = 100
        # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent
        sample = random.sample(
            range(validation_sample_window_size), self.validation_set_size // 2)
        validation_examples = np.array(sample)

        sample = random.sample(
            range(self.n_vocab // 2, self.n_vocab // 2 + validation_sample_window_size),
            self.validation_set_size // 2)
        validation_examples = np.append(validation_examples, sample)
        return validation_examples

    def __get_similarity_ops(self):

        validation_dataset = tf.constant(self.validation_examples, dtype=tf.int32, name='validation_dataset')
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keep_dims=True), name='norm')

        # We use the cosine distance:
        # normalized_embedding = tf.nn.l2_normalize(embedding, dim=1, name='normalized_embedding')
        normalized_embedding = tf.divide(self.embedding, norm, name='normalized_embedding')

        validation_embedding = tf.nn.embedding_lookup(normalized_embedding, validation_dataset,
                                                      name='validation_embedding')
        similarity = tf.matmul(validation_embedding, tf.transpose(normalized_embedding), name='similarity')
        return normalized_embedding, similarity, validation_embedding
