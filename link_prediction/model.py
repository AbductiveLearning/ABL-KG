import math
from operator import is_
import timeit
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from dataset import KnowledgeGraph


class TransE:
    def __init__(self, kg: KnowledgeGraph,
                 embedding_dim, margin_value, score_func,
                 batch_size, learning_rate, n_generator, n_rank_calculator, threshold):
        self.threshold = threshold
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.margin_value = margin_value
        self.score_func = score_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_generator = n_generator
        self.n_rank_calculator = n_rank_calculator
        '''ops for training'''
        self.triple_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.triple_neg = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        self.margin = tf.placeholder(dtype=tf.float32, shape=[None])
        self.train_op = None
        self.loss = None
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.merge = None
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None
        self.energy_head_prediction = None
        self.energy_tail_prediction = None
        '''embeddings'''
        # bound = 6 / math.sqrt(self.embedding_dim)
        with tf.variable_scope('embedding'):
            # self.entity_embedding = tf.get_variable(name='entity',
            #                                         shape=[kg.n_entity, self.embedding_dim],
            #                                         initializer=tf.random_uniform_initializer(minval=-bound,
            #                                                                                   maxval=bound))
            self.entity_embedding = tf.get_variable('entity', shape=[kg.n_entity, self.embedding_dim],
                                                    initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram(name=self.entity_embedding.op.name, values=self.entity_embedding)
            # self.relation_embedding = tf.get_variable(name='relation',
            #                                           shape=[kg.n_relation, self.embedding_dim],
            #                                           initializer=tf.random_uniform_initializer(minval=-bound,
            #                                                                                     maxval=bound))
            self.relation_embedding = tf.get_variable('relation', shape=[kg.n_relation, self.embedding_dim],
                                                      initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram(name=self.relation_embedding.op.name, values=self.relation_embedding)

        with tf.name_scope('normalization'):
            self.entity_embedding = tf.nn.l2_normalize(self.entity_embedding, dim=1)
            # self.relation_embedding = tf.nn.l2_normalize(self.relation_embedding, dim=1)

        self.build_graph()
        self.build_eval_graph()

    def build_graph(self):
        with tf.name_scope('training'):
            distance_pos, distance_neg = self.infer(self.triple_pos, self.triple_neg)
            self.loss = self.calculate_loss(distance_pos, distance_neg, self.margin)
            tf.summary.scalar(name=self.loss.op.name, tensor=self.loss)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
            self.merge = tf.summary.merge_all()

    def build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction, \
            self.energy_head_prediction, self.energy_tail_prediction = self.evaluate(self.eval_triple)

    def infer(self, triple_pos, triple_neg, dropout=False):
        with tf.name_scope('lookup'):
            head_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 0])
            tail_pos = tf.nn.embedding_lookup(self.entity_embedding, triple_pos[:, 1])
            relation_pos = tf.nn.embedding_lookup(self.relation_embedding, triple_pos[:, 2])
            head_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 0])
            tail_neg = tf.nn.embedding_lookup(self.entity_embedding, triple_neg[:, 1])
            relation_neg = tf.nn.embedding_lookup(self.relation_embedding, triple_neg[:, 2])
        if dropout:
            with tf.name_scope('dropout'):
                keep_prob = 0.9
                head_pos = tf.nn.dropout(head_pos, keep_prob=keep_prob)
                tail_pos = tf.nn.dropout(tail_pos, keep_prob=keep_prob)
                relation_pos = tf.nn.dropout(relation_pos, keep_prob=keep_prob)
                head_neg = tf.nn.dropout(head_neg, keep_prob=keep_prob)
                tail_neg = tf.nn.dropout(tail_neg, keep_prob=keep_prob)
                relation_neg = tf.nn.dropout(relation_neg, keep_prob=keep_prob)
        with tf.name_scope('link'):
            distance_pos = head_pos + relation_pos - tail_pos
            distance_neg = head_neg + relation_neg - tail_neg
        return distance_pos, distance_neg

    def calculate_loss(self, distance_pos, distance_neg, margin):
        with tf.name_scope('loss'):
            if self.score_func == 'L1':  # L1 score
                score_pos = tf.reduce_sum(tf.abs(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.abs(distance_neg), axis=1)
            else:  # L2 score
                score_pos = tf.reduce_sum(tf.square(distance_pos), axis=1)
                score_neg = tf.reduce_sum(tf.square(distance_neg), axis=1)
            loss = tf.reduce_mean(tf.nn.relu(self.margin_value + score_pos - score_neg), name='max_margin_loss')
        return loss

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[0])
            tail = tf.nn.embedding_lookup(self.entity_embedding, eval_triple[1])
            relation = tf.nn.embedding_lookup(self.relation_embedding, eval_triple[2])
        with tf.name_scope('link'):
            distance_head_prediction = self.entity_embedding + relation - tail
            distance_tail_prediction = head + relation - self.entity_embedding
        with tf.name_scope('rank'):
            if self.score_func == 'L1':  # L1 score
                energy_head_prediction, idx_head_prediction = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(distance_head_prediction), axis=1), k=self.kg.n_entity)
                energy_tail_prediction, idx_tail_prediction = tf.nn.top_k(
                    tf.reduce_sum(tf.abs(distance_tail_prediction), axis=1), k=self.kg.n_entity)
            else:  # L2 score
                energy_head_prediction, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(
                    distance_head_prediction), axis=1), k=self.kg.n_entity)
                energy_tail_prediction, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(
                    distance_tail_prediction), axis=1), k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction, energy_head_prediction, energy_tail_prediction

    def launch_training(self, session, summary_writer):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue}).start()
        # print('-----Start training-----')
        start = timeit.default_timer()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        # print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            batch_loss, _, summary = session.run(fetches=[self.loss, self.train_op, self.merge],
                                                 feed_dict={self.triple_pos: batch_pos,
                                                            self.triple_neg: batch_neg,
                                                            self.margin: [self.margin_value] * len(batch_pos)})
            summary_writer.add_summary(summary, global_step=self.global_step.eval(session=session))
            epoch_loss += batch_loss
            n_used_triple += len(batch_pos)
            # print('[{:.3f}s] #triple: {}/{} triple_avg_loss: {:.6f}'.format(timeit.default_timer() - start,
            #                                                                 n_used_triple,
            #                                                                 self.kg.n_training_triple,
            #                                                                 batch_loss), end='\r')
        # print()
        print('epoch loss: {:.3f}, cost time: {:.3f}s'.format(epoch_loss, timeit.default_timer() - start))
        # print('-----Finish training-----')
        # self.check_norm(session=session)

    def launch_evaluation(self, session, is_test=False):
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        print('-----Start evaluation-----')
        start = timeit.default_timer()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        eval_triples = self.kg.validation_triples
        if is_test:
            eval_triples = self.kg.test_triples
        for eval_triple in eval_triples:
            idx_head_prediction, idx_tail_prediction, \
            energy_head_prediction, energy_tail_prediction = session.run(fetches=[self.idx_head_prediction,
                                                                                  self.idx_tail_prediction,
                                                                                  self.energy_head_prediction,
                                                                                  self.energy_tail_prediction],
                                                                         feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction,
                                   energy_head_prediction, energy_tail_prediction))
            n_used_eval_triple += 1
            # print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
            #                                                    n_used_eval_triple,
            #                                                    len(eval_triples)), end='\r')
        # print()
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        # print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        # print('-----All rank calculation accomplished-----')
        # print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_mr_raw = 0
        head_mrr_raw = 0
        head_hits1_raw = 0
        head_hits3_raw = 0
        head_hits5_raw = 0
        head_hits10_raw = 0
        tail_mr_raw = 0
        tail_mrr_raw = 0
        tail_hits1_raw = 0
        tail_hits3_raw = 0
        tail_hits5_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_mr_filter = 0
        head_mrr_filter = 0
        head_hits1_filter = 0
        head_hits3_filter = 0
        head_hits5_filter = 0
        head_hits10_filter = 0
        tail_mr_filter = 0
        tail_mrr_filter = 0
        tail_hits1_filter = 0
        tail_hits3_filter = 0
        tail_hits5_filter = 0
        tail_hits10_filter = 0

        new_predicated_triples = set()

        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter, \
            head, relation, predicted_tail = rank_result_queue.get()

            if predicted_tail is not None:
                new_predicated_triples.add((head, predicted_tail, relation))

            head_mr_raw += (head_rank_raw + 1)
            head_mrr_raw += 1 / (head_rank_raw + 1)
            if head_rank_raw < 1:
                head_hits1_raw += 1
            if head_rank_raw < 3:
                head_hits3_raw += 1
            if head_rank_raw < 5:
                head_hits5_raw += 1
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_mr_raw += (tail_rank_raw + 1)
            tail_mrr_raw += 1 / (tail_rank_raw + 1)
            if tail_rank_raw < 1:
                tail_hits1_raw += 1
            if tail_rank_raw < 3:
                tail_hits3_raw += 1
            if tail_rank_raw < 5:
                tail_hits5_raw += 1
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_mr_filter += (head_rank_filter + 1)
            head_mrr_filter += 1 / (head_rank_filter + 1)
            if head_rank_filter < 1:
                head_hits1_filter += 1
            if head_rank_filter < 3:
                head_hits3_filter += 1
            if head_rank_filter < 5:
                head_hits5_filter += 1
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_mr_filter += (tail_rank_filter + 1)
            tail_mrr_filter += 1 / (tail_rank_filter + 1)
            if tail_rank_filter < 1:
                tail_hits1_filter += 1
            if tail_rank_filter < 3:
                tail_hits3_filter += 1
            if tail_rank_filter < 5:
                tail_hits5_filter += 1
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_mr_raw /= n_used_eval_triple
        head_mrr_raw /= n_used_eval_triple
        head_hits1_raw /= n_used_eval_triple
        head_hits3_raw /= n_used_eval_triple
        head_hits5_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_mr_raw /= n_used_eval_triple
        tail_mrr_raw /= n_used_eval_triple
        tail_hits1_raw /= n_used_eval_triple
        tail_hits3_raw /= n_used_eval_triple
        tail_hits5_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            head_mr_raw, head_mrr_raw, head_hits1_raw, head_hits3_raw, head_hits5_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            tail_mr_raw, tail_mrr_raw, tail_hits1_raw, tail_hits3_raw, tail_hits5_raw, tail_hits10_raw))
        print('------Average------')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            (head_mr_raw + tail_mr_raw) / 2,
            (head_mrr_raw + tail_mrr_raw) / 2,
            (head_hits1_raw + tail_hits1_raw) / 2,
            (head_hits3_raw + tail_hits3_raw) / 2,
            (head_hits5_raw + tail_hits5_raw) / 2,
            (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_mr_filter /= n_used_eval_triple
        head_mrr_filter /= n_used_eval_triple
        head_hits1_filter /= n_used_eval_triple
        head_hits3_filter /= n_used_eval_triple
        head_hits5_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_mr_filter /= n_used_eval_triple
        tail_mrr_filter /= n_used_eval_triple
        tail_hits1_filter /= n_used_eval_triple
        tail_hits3_filter /= n_used_eval_triple
        tail_hits5_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            head_mr_filter, head_mrr_filter, head_hits1_filter, head_hits3_filter, head_hits5_filter,
            head_hits10_filter))
        print('-----Tail prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            tail_mr_filter, tail_mrr_filter, tail_hits1_filter, tail_hits3_filter, tail_hits5_filter,
            tail_hits10_filter))
        print('-----Average-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            (head_mr_filter + tail_mr_filter) / 2,
            (head_mrr_filter + tail_mrr_filter) / 2,
            (head_hits1_filter + tail_hits1_filter) / 2,
            (head_hits3_filter + tail_hits3_filter) / 2,
            (head_hits5_filter + tail_hits5_filter) / 2,
            (head_hits10_filter + tail_hits10_filter) / 2))
        print('-----Finish evaluation-----')

        return (head_hits1_filter + tail_hits1_filter) / 2, new_predicated_triples

    def semi_training(self, new_predicated_triples):
        self.kg.add_training_data(new_predicated_triples)

    # def launch_evaluation(self, session):
    #     eval_result_queue = mp.JoinableQueue()
    #     rank_result_queue = mp.Queue()
    #     print('-----Start evaluation-----')
    #     start = timeit.default_timer()
    #     for _ in range(self.n_rank_calculator):
    #         mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
    #                                                        'out_queue': rank_result_queue}).start()
    #     n_used_eval_triple = 0
    #     for eval_triple in self.kg.test_triples:
    #         idx_head_prediction, idx_tail_prediction = session.run(fetches=[self.idx_head_prediction,
    #                                                                         self.idx_tail_prediction],
    #                                                                feed_dict={self.eval_triple: eval_triple})
    #         eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
    #         n_used_eval_triple += 1
    #         print('[{:.3f}s] #evaluation triple: {}/{}'.format(timeit.default_timer() - start,
    #                                                            n_used_eval_triple,
    #                                                            self.kg.n_test_triple), end='\r')
    #     print()
    #     for _ in range(self.n_rank_calculator):
    #         eval_result_queue.put(None)
    #     print('-----Joining all rank calculator-----')
    #     eval_result_queue.join()
    #     print('-----All rank calculation accomplished-----')
    #     print('-----Obtaining evaluation results-----')
    #     '''Raw'''
    #     head_meanrank_raw = 0
    #     head_hits10_raw = 0
    #     tail_meanrank_raw = 0
    #     tail_hits10_raw = 0
    #     '''Filter'''
    #     head_meanrank_filter = 0
    #     head_hits10_filter = 0
    #     tail_meanrank_filter = 0
    #     tail_hits10_filter = 0
    #     for _ in range(n_used_eval_triple):
    #         head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
    #         head_meanrank_raw += head_rank_raw
    #         if head_rank_raw < 10:
    #             head_hits10_raw += 1
    #         tail_meanrank_raw += tail_rank_raw
    #         if tail_rank_raw < 10:
    #             tail_hits10_raw += 1
    #         head_meanrank_filter += head_rank_filter
    #         if head_rank_filter < 10:
    #             head_hits10_filter += 1
    #         tail_meanrank_filter += tail_rank_filter
    #         if tail_rank_filter < 10:
    #             tail_hits10_filter += 1
    #     print('-----Raw-----')
    #     head_meanrank_raw /= n_used_eval_triple
    #     head_hits10_raw /= n_used_eval_triple
    #     tail_meanrank_raw /= n_used_eval_triple
    #     tail_hits10_raw /= n_used_eval_triple
    #     print('-----Head prediction-----')
    #     print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_raw, head_hits10_raw))
    #     print('-----Tail prediction-----')
    #     print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_raw, tail_hits10_raw))
    #     print('------Average------')
    #     print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_raw + tail_meanrank_raw) / 2,
    #                                                      (head_hits10_raw + tail_hits10_raw) / 2))
    #     print('-----Filter-----')
    #     head_meanrank_filter /= n_used_eval_triple
    #     head_hits10_filter /= n_used_eval_triple
    #     tail_meanrank_filter /= n_used_eval_triple
    #     tail_hits10_filter /= n_used_eval_triple
    #     print('-----Head prediction-----')
    #     print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(head_meanrank_filter, head_hits10_filter))
    #     print('-----Tail prediction-----')
    #     print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format(tail_meanrank_filter, tail_hits10_filter))
    #     print('-----Average-----')
    #     print('MeanRank: {:.3f}, Hits@10: {:.3f}'.format((head_meanrank_filter + tail_meanrank_filter) / 2,
    #                                                      (head_hits10_filter + tail_hits10_filter) / 2))
    #     print('cost time: {:.3f}s'.format(timeit.default_timer() - start))
    #     print('-----Finish evaluation-----')
    #     return (head_hits10_filter + tail_hits10_filter) / 2

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction, \
                energy_head_prediction, energy_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                predicted_tail = None
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                # print(energy_tail_prediction[-1], energy_tail_prediction[0])
                # print(energy_tail_prediction)
                # if energy_tail_prediction[-1] < self.threshold:
                #     print(energy_tail_prediction[-1])
                predicted_tail = idx_tail_prediction[-1]
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter, head, relation, predicted_tail))
                in_queue.task_done()

    def check_norm(self, session):
        print('-----Check norm-----')
        entity_embedding = self.entity_embedding.eval(session=session)
        relation_embedding = self.relation_embedding.eval(session=session)
        entity_norm = np.linalg.norm(entity_embedding, ord=2, axis=1)
        relation_norm = np.linalg.norm(relation_embedding, ord=2, axis=1)
        print('entity norm: {} relation norm: {}'.format(entity_norm, relation_norm))
