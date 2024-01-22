from tkinter import E
from dataset import KnowledgeGraph
from model import TransE
import abl

import tensorflow as tf
import argparse


def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--data_dir', type=str, default='./datasets/FB15K-237/')
    parser.add_argument('--embedding_dim', type=int, default=196)
    parser.add_argument('--margin_value', type=float, default=3.5)
    parser.add_argument('--score_func', type=str, default='L1')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_generator', type=int, default=8)
    parser.add_argument('--n_rank_calculator', type=int, default=8)
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/')
    parser.add_argument('--summary_dir', type=str, default='./summary/')
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--semi', type=bool, default=False)
    parser.add_argument('--abl', type=bool, default=False)
    parser.add_argument('--rule', type=str, default='rules.txt')
    parser.add_argument('--threshold', type=float, default=1.0)

    args = parser.parse_args()
    print(args)
    max_hit1 = 0.0
    drop_num = 0
    kg = KnowledgeGraph(data_dir=args.data_dir)
    kge_model = TransE(kg=kg, embedding_dim=args.embedding_dim, margin_value=args.margin_value,
                       score_func=args.score_func, batch_size=args.batch_size, learning_rate=args.learning_rate,
                       n_generator=args.n_generator, n_rank_calculator=args.n_rank_calculator, threshold=args.threshold)
    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    with tf.Session(config=sess_config) as sess:
        print('-----Initializing tf graph-----')
        tf.global_variables_initializer().run()
        print('-----Initialization accomplished-----')
        # kge_model.check_norm(session=sess)
        summary_writer = tf.summary.FileWriter(logdir=args.summary_dir, graph=sess.graph)
        for epoch in range(1, args.max_epoch+1):
            print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
            kge_model.launch_training(session=sess, summary_writer=summary_writer)
            if epoch % args.eval_freq == 0:
                is_test = False
                if args.semi:
                    is_test = True
                h1, new_data = kge_model.launch_evaluation(session=sess, is_test=is_test)
                if h1 < max_hit1:
                    drop_num += 1
                    if drop_num == 2 and epoch >= 250:
                        print("Early stop. Test:")
                        kge_model.launch_evaluation(session=sess, is_test=True)
                        exit(0)
                else:
                    max_hit1 = h1
                    drop_num = 0
                if args.semi and epoch >= 0:
                    if args.abl:
                        print("abl:")
                        test_triples, _, _ = abl.read_relation_triples(args.data_dir + "test.txt")
                        train_triples, _, _ = abl.read_relation_triples(args.data_dir + "train.txt")
                        valid_triples, _, _ = abl.read_relation_triples(args.data_dir + "valid.txt")
                        rules = abl.read_rules(args.data_dir + args.rule)
                        abl_triples = abl.infer_triples(train_triples | valid_triples, test_triples, rules)
                        abl_triples_ids = set()
                        for h, r, t in abl_triples:
                            abl_triples_ids.add((kg.entity_dict[h], kg.entity_dict[t], kg.relation_dict[r]))
                        kge_model.semi_training(abl_triples_ids)
                    else:
                        print("semi:")
                        kge_model.semi_training(new_data)


if __name__ == '__main__':
    main()
