import gc
import os
import math
import random
import time
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import openea.modules.train.batch as bat

from openea.modules.finding.evaluation import early_stop
from openea.modules.utils.util import task_divide
from openea.modules.base.optimizers import generate_optimizer
from openea.modules.utils.util import load_session
from openea.modules.base.initializers import init_embeddings
from openea.modules.base.losses import limited_loss, positive_loss
from openea.models.basic_model import BasicModel
from openea.modules.finding.evaluation import test
from openea.modules.finding.similarity import sim
from openea.modules.load.read import generate_sup_relation_triples
from openea.modules.bootstrapping.alignment_finder import find_alignment, mwgm_graph_tool, mwgm_igraph

from abduction import abduction_learning

from sklearn.metrics.pairwise import cosine_similarity

from utils import merge_new_alignment 
from utils import create_mined_rules, simplify_mined_rules


def neg_pair_margin_loss(e1, e2, margin=0.1):
    neg_distance = tf.reduce_sum(tf.square(e1 - e2), axis=1)
    loss = tf.reduce_sum(tf.nn.relu(tf.constant(margin) - neg_distance), name='neg_pair_margin_loss')
    return loss


class AlignE(BasicModel):

    def __init__(self):
        super().__init__()

    def init(self):
        self._define_variables()
        self._define_embed_graph()
        self._define_alignment_graph()
        self.session = load_session()
        tf.global_variables_initializer().run(session=self.session)

        # customize parameters
        assert self.args.init == 'normal'
        assert self.args.alignment_module == 'swapping'
        assert self.args.loss == 'limited'
        assert self.args.neg_sampling == 'truncated'
        assert self.args.optimizer == 'Adagrad'
        assert self.args.eval_metric == 'inner'
        assert self.args.loss_norm == 'L2'

        assert self.args.ent_l2_norm is True
        assert self.args.rel_l2_norm is True

        assert self.args.pos_margin >= 0.0
        assert self.args.neg_margin > self.args.pos_margin

        assert self.args.neg_triple_num > 1
        assert self.args.truncated_epsilon > 0.0
        assert self.args.learning_rate >= 0.01
        
        ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        self.kg1_ref_id_to_idx = {ref_ent1[i]: i for i in range(len(ref_ent1))}
        self.kg2_ref_id_to_idx = {ref_ent2[i]: i for i in range(len(ref_ent2))} 
        self.kg1_ref_idx_to_id = {v:k for k, v in self.kg1_ref_id_to_idx.items()}
        self.kg2_ref_idx_to_id = {v:k for k, v in self.kg2_ref_id_to_idx.items()}
        self.true_e1_to_e2 = {e1_id: e2_id for e1_id, e2_id in self.kgs.train_links + self.kgs.valid_links+ self.kgs.test_links}
        self.true_e2_to_e1 = {e2_id: e1_id for e1_id, e2_id in self.kgs.train_links + self.kgs.valid_links+ self.kgs.test_links}
        
        self.true_r1_to_r2 = {r1_id: self.kgs.kg2.relations_id_dict[r1_uri] for r1_uri, r1_id in self.kgs.kg1.relations_id_dict.items() if r1_uri in self.kgs.kg2.relations_id_dict.keys() }
        
        self.kg1_relations_uri_to_id = {r_uri: r_id for r_uri, r_id in self.kgs.kg1.relations_id_dict.items()}
        self.kg2_relations_uri_to_id = {r_uri: r_id for r_uri, r_id in self.kgs.kg2.relations_id_dict.items()}
        self.kg1_relations_id_to_uri = {r_id: r_uri for r_uri, r_id in self.kgs.kg1.relations_id_dict.items()}
        self.kg2_relations_id_to_uri = {r_id: r_uri for r_uri, r_id in self.kgs.kg2.relations_id_dict.items()}
        
        self.kg1_entities_uri_to_id = {e_uri: e_id for e_uri, e_id in self.kgs.kg1.entities_id_dict.items()}
        self.kg2_entities_uri_to_id = {e_uri: e_id for e_uri, e_id in self.kgs.kg2.entities_id_dict.items()}
        self.kg1_entities_id_to_uri = {e_id: e_uri for e_uri, e_id in self.kgs.kg1.entities_id_dict.items()}
        self.kg2_entities_id_to_uri = {e_id: e_uri for e_uri, e_id in self.kgs.kg2.entities_id_dict.items()}
           
    def _define_variables(self):
        with tf.variable_scope('relational' + 'embeddings'):
            self.ent_embeds = init_embeddings([self.kgs.entities_num, self.args.dim], 'ent_embeds',
                                              self.args.init, self.args.ent_l2_norm)
            self.rel_embeds = init_embeddings([self.kgs.relations_num, self.args.dim], 'rel_embeds',
                                              self.args.init, self.args.rel_l2_norm)

    def _define_embed_graph(self):
        with tf.name_scope('triple_placeholder'):
            self.pos_hs = tf.placeholder(tf.int32, shape=[None])
            self.pos_rs = tf.placeholder(tf.int32, shape=[None])
            self.pos_ts = tf.placeholder(tf.int32, shape=[None])
            self.neg_hs = tf.placeholder(tf.int32, shape=[None])
            self.neg_rs = tf.placeholder(tf.int32, shape=[None])
            self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        with tf.name_scope('triple_lookup'):
            phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
            prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
            pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
            nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
            nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
            nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)
        with tf.name_scope('triple_loss'):
            self.triple_loss = limited_loss(phs, prs, pts, nhs, nrs, nts,
                                            self.args.pos_margin, self.args.neg_margin,
                                            self.args.loss_norm, balance=self.args.neg_margin_balance)
            self.triple_optimizer = generate_optimizer(self.triple_loss, self.args.learning_rate,
                                                       opt=self.args.optimizer)

    def _define_alignment_graph(self):
        self.new_h = tf.placeholder(tf.int32, shape=[None])
        self.new_r = tf.placeholder(tf.int32, shape=[None])
        self.new_t = tf.placeholder(tf.int32, shape=[None])
        
        self.new_negative_e1 = tf.placeholder(tf.int32, shape=[None])
        self.new_negative_e2 = tf.placeholder(tf.int32, shape=[None])
        
        phs = tf.nn.embedding_lookup(self.ent_embeds, self.new_h)
        prs = tf.nn.embedding_lookup(self.rel_embeds, self.new_r)
        pts = tf.nn.embedding_lookup(self.ent_embeds, self.new_t)
        
        e1 = tf.nn.embedding_lookup(self.ent_embeds, self.new_negative_e1)
        e2 = tf.nn.embedding_lookup(self.ent_embeds, self.new_negative_e2)
        
        self.alignment_loss = positive_loss(phs, prs, pts, "L2") + neg_pair_margin_loss(e1, e2, margin=self.args.neg_pair_margin)
        
        self.alignment_optimizer = generate_optimizer(self.alignment_loss, self.args.learning_rate,
                                                      opt=self.args.optimizer)

    def _check_predict_pairs(self, pairs):
        predict_true = 0
        # print(self.true_e1_to_e2)
        # print(pairs)
        new_pairs = set()
        for e1_id, e2_id in pairs:
            if e1_id in self.true_e1_to_e2:
                if self.true_e1_to_e2[e1_id] == e2_id:
                    predict_true += 1
                new_pairs.add((e1_id, e2_id))
            elif e1_id in self.true_e2_to_e1:
                if self.true_e2_to_e1[e1_id] == e2_id:
                    predict_true += 1
                new_pairs.add((e2_id, e1_id))
            else:
                assert 0
        pairs = new_pairs
        if len(pairs) == 0:
            precision = 0    
        else:
            precision = predict_true / len(pairs)
        recall = predict_true / (len(self.kgs.valid_links) + len(self.kgs.test_links))
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        print("among {} entity pairs, predict true: {}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(len(pairs), predict_true, precision, recall, f1))
    
    def _check_predict_relation_pairs(self, pairs):
        predict_true = 0
        for r1_id, r2_id in pairs:
            if r1_id in self.true_r1_to_r2.keys() and self.true_r1_to_r2[r1_id] == r2_id:
                predict_true += 1
        if len(pairs) == 0:
            precision = 0
        else:
            precision = predict_true / len(pairs)
        recall = predict_true / len(self.true_r1_to_r2)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        print("among {} relation pairs, predict true: {}, precision: {:.3f}, recall: {:.3f}({}/{}), f1: {:.3f}".format(len(pairs), predict_true, precision, recall, predict_true, len(self.true_r1_to_r2), f1))
    
    def _obtain_ref_sim_mat(self):
        ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        embeds1 = tf.nn.embedding_lookup(self.ent_embeds, ref_ent1).eval(session=self.session)
        embeds2 = tf.nn.embedding_lookup(self.ent_embeds, ref_ent2).eval(session=self.session)
        ref_sim_mat = np.matmul(tf.nn.l2_normalize(embeds1, 1).eval(session=self.session), tf.nn.l2_normalize(embeds2, 1).eval(session=self.session).T)
        return ref_sim_mat

    def predict_alignment(self):
        print("########################### predict_alignment ################################")
        ref_sim_mat = self._obtain_ref_sim_mat()

        alignment_rest = find_alignment(ref_sim_mat, self.args.entity_score_threshold, self.args.find_topK)

        print("after filtering with entity score threshold, check alignment_rest quality:")
        self._check_predict_pairs([(self.kg1_ref_idx_to_id[e1_idx], self.kg2_ref_idx_to_id[e2_idx]) for e1_idx, e2_idx in alignment_rest])

        alignment_rest = mwgm_graph_tool(alignment_rest, ref_sim_mat)
        
        print("after 1 vs 1 constraint, check alignment_rest quality:")
        self._check_predict_pairs([(self.kg1_ref_idx_to_id[e1_idx], self.kg2_ref_idx_to_id[e2_idx]) for e1_idx, e2_idx in alignment_rest])
        
        predict_pairs = [(self.kg1_ref_idx_to_id[e1_idx], self.kg2_ref_idx_to_id[e2_idx]) for e1_idx, e2_idx in alignment_rest]
        print("################################################################################")
        return predict_pairs, ref_sim_mat

    def abduct_alignment(self, pre_abducted_pairs, ref_sim_mat, whether_use_mined_rule):
        print("########################### abduct_alignment ################################")

        sup_relation_uri_pairs = [(self.kg1_relations_id_to_uri[r1_id], self.kg2_relations_id_to_uri[r2_id]) for r1_id, r2_id in self.true_r1_to_r2.items()]
        post_abducted_pairs_set, inferred_entity_pairs_set, mined_rules = abduction_learning(self.args, self.kgs, sup_relation_uri_pairs, pre_abducted_pairs, ref_sim_mat, self.kg1_ref_id_to_idx, self.kg2_ref_id_to_idx, whether_use_mined_rule)
        conflict_pairs = list(set(pre_abducted_pairs) - set(post_abducted_pairs_set)) 
        print("minded rules usage: {}".format(whether_use_mined_rule))
        print("minded rules num: {}".format(len(mined_rules)))
        
        abducted_pairs = post_abducted_pairs_set | inferred_entity_pairs_set
        
        print("check post_abducted_pairs_set quality:")
        self._check_predict_pairs(post_abducted_pairs_set)
        
        print("check inferred_entity_pairs_set quality:")
        self._check_predict_pairs(inferred_entity_pairs_set)
        
        print("check abducted_pairs quality:")
        self._check_predict_pairs(abducted_pairs)
        
        print("check conflict_pairs quality:")
        self._check_predict_pairs(conflict_pairs)
        
        print("################################################################################")
        return abducted_pairs, conflict_pairs
   
    def predict_and_abduct(self):
        predict_pairs, ref_sim_mat = self.predict_alignment()
        
        abducted_pairs, negative_pairs = self.abduct_alignment(predict_pairs, ref_sim_mat, \
            whether_use_mined_rule=self.args.use_mined_rule)   
        return abducted_pairs, negative_pairs

    def train_new_alignment(self, new_positive_alignment, new_negative_alignment, batch_size):
        new_triples1, new_triples2 = generate_sup_relation_triples(new_positive_alignment,
                                                    self.kgs.kg1.rt_dict, self.kgs.kg1.hr_dict,
                                                    self.kgs.kg2.rt_dict, self.kgs.kg2.hr_dict)
        new_triples = new_triples1 | new_triples2
                
        start = time.time()
        epoch_loss = 0
        if batch_size > len(new_triples):
            batch_size = len(new_triples)
        triple_steps = len(new_triples) // batch_size
        new_triples = list(new_triples) 
        for i in range(triple_steps + 1):
            
            triple_batch = new_triples[i*batch_size: (i+1)*batch_size]
            
            alignment_fetches = {"loss": self.alignment_loss, "train_op": self.alignment_optimizer}
            neg_pair_batch = random.sample(new_negative_alignment, len(new_negative_alignment) // triple_steps)
            neg_pair_e1_batch = [e1 for e1, e2 in neg_pair_batch]
            neg_pair_e2_batch = [e2 for e1, e2 in neg_pair_batch]
           
            # neg_pair_batch = new_negative_alignment[i* (len(new_negative_alignment) // triple_steps): (i+1)* (len(new_negative_alignment) // triple_steps)]
            if len(neg_pair_batch) == 0 or self.args.use_new_negative_alignment == False:
                # neg_pair_batch = [[1],[2]]
                neg_pair_e1_batch = [1]
                neg_pair_e2_batch = [2]
            alignment_feed_dict = {self.new_h: [tr[0] for tr in triple_batch],
                                    self.new_r: [tr[1] for tr in triple_batch],
                                    self.new_t: [tr[2] for tr in triple_batch],
                                    self.new_negative_e1: neg_pair_e1_batch,
                                    self.new_negative_e2:neg_pair_e2_batch}
            rest = self.session.run(fetches=alignment_fetches, feed_dict=alignment_feed_dict)
            epoch_loss += rest["loss"]
        print("alignment_loss = {:.3f}, time = {:.3f} s".format(epoch_loss, time.time() - start))
        
    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        global_new_alignment_pairs = set()
        for i in range(1, self.args.max_epoch + 1):
            # training
            self.launch_training_1epo(i, triple_steps, steps_tasks, training_batch_queue, neighbors1, neighbors2)
            # validation
            if i >= self.args.start_valid and i % self.args.eval_freq == 0:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == self.args.max_epoch:
                    break
            # abductive learning
            if i >= self.args.abl_start and i % self.args.abl_freq == 0:
                abducted_pairs, negative_pairs = self.predict_and_abduct()
                if self.args.accumulate_new_alignment:
                    global_new_alignment_pairs = merge_new_alignment(global_new_alignment_pairs, abducted_pairs)
                    new_alignment_pairs = global_new_alignment_pairs
                else:
                    new_alignment_pairs = abducted_pairs
                if len(new_alignment_pairs) > 0:
                    self.train_new_alignment(new_alignment_pairs, negative_pairs, self.args.batch_size)
                    self.valid(self.args.stop_metric)
            # nn sampling
            if self.args.neg_sampling == 'truncated' and i % self.args.truncated_freq == 0:
                t1 = time.time()
                assert 0.0 < self.args.truncated_epsilon < 1.0
                neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
                neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
                if neighbors1 is not None:
                    del neighbors1, neighbors2
                gc.collect()
                neighbors1 = bat.generate_neighbours_single_thread(self.eval_kg1_useful_ent_embeddings(),
                                                                   self.kgs.useful_entities_list1,
                                                                   neighbors_num1, self.args.test_threads_num)
                neighbors2 = bat.generate_neighbours_single_thread(self.eval_kg2_useful_ent_embeddings(),
                                                                   self.kgs.useful_entities_list2,
                                                                   neighbors_num2, self.args.test_threads_num)
                ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
                print("\ngenerating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
                gc.collect()
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))

    def load(self):
     
        folder = os.path.join(self.args.output, self.args.embedding_module, self.args.training_data.strip('/').split('/')[-1], \
            self.args.dataset_division, self.args.load_model_dir)
        ent_embeds = np.load(os.path.join(folder,"ent_embeds.npy"))
        rel_embeds = np.load(os.path.join(folder,"rel_embeds.npy"))
        
        ref_ent1 = self.kgs.valid_entities1 + self.kgs.test_entities1
        ref_ent2 = self.kgs.valid_entities2 + self.kgs.test_entities2
        ent_embeds1 = ent_embeds[ref_ent1] 
        ent_embeds2 = ent_embeds[ref_ent2] 
        
        rel_embeds1 = rel_embeds[list(self.kgs.kg1.relations_id_dict.values())] 
        rel_embeds2 = rel_embeds[list(self.kgs.kg2.relations_id_dict.values())] 
    
        r1_list = list(self.kgs.kg1.relations_id_dict.values())
        r2_list = list(self.kgs.kg2.relations_id_dict.values())
