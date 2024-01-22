import argparse

from openea.modules.args.args_hander import check_args, load_args
from openea.modules.load.kgs import read_kgs_from_folder, read_kgs_from_dbp_dwy
from openea.models.basic_model import BasicModel
from aligne import AlignE

from utils import str2bool

parser = argparse.ArgumentParser(description='ABL4EA')
parser.add_argument('--training_data', type=str, default='datasets/EN_FR_15K_V2/') 
parser.add_argument('--output', type=str, default='output/results/') 
parser.add_argument('--dataset_division', type=str, default='721_5fold/1/')

parser.add_argument('--embedding_module', type=str, default='AlignE',  choices=['AlignE', 'AliNet'])
parser.add_argument('--init', type=str, default='normal', choices=['normal', 'unit', 'xavier', 'uniform'])
parser.add_argument('--alignment_module', type=str, default='swapping', choices=['sharing', 'mapping', 'swapping'])
parser.add_argument('--search_module', type=str, default='greedy', choices=['greedy', 'global'])
parser.add_argument('--loss', type=str, default='limited', choices=['margin-based', 'logistic', 'limited'])
parser.add_argument('--neg_sampling', type=str, default='truncated', choices=['uniform', 'truncated'])

parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--loss_norm', type=str, default='L2')
parser.add_argument('--ent_l2_norm', type=bool, default=True)
parser.add_argument('--rel_l2_norm', type=bool, default=True)
parser.add_argument('--batch_size', type=int, default=5000) 

parser.add_argument('--pos_margin', type=float, default=0.01)
parser.add_argument('--neg_margin', type=float, default=2.0) 
parser.add_argument('--neg_margin_balance', type=float, default=0.2) 

parser.add_argument('--neg_triple_num', type=int, default=10)
parser.add_argument('--truncated_epsilon', type=float, default=0.9)
parser.add_argument('--truncated_freq', type=int, default=10)

parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, default='Adagrad', choices=['Adagrad', 'Adadelta', 'Adam', 'SGD'])
parser.add_argument('--batch_threads_num', type=int, default=4)
parser.add_argument('--test_threads_num', type=int, default=4)
parser.add_argument('--max_epoch', type=int, default=1000) 
parser.add_argument('--eval_freq', type=int, default=10)

parser.add_argument('--ordered', type=bool, default=True)
parser.add_argument('--top_k', type=list, default=[1, 5, 10, 50])
parser.add_argument('--csls', type=int, default=10)

parser.add_argument('--is_save', type=bool, default=False)
parser.add_argument('--eval_norm', type=bool, default=True)
parser.add_argument('--start_valid', type=int, default=0)
parser.add_argument('--stop_metric', type=str, default='mrr', choices=['hits1', 'mrr'])
parser.add_argument('--eval_metric', type=str, default='inner', choices=['inner', 'cosine', 'euclidean', 'manhattan'])

parser.add_argument('--entity_score_threshold', type=int, default=0.7) 
parser.add_argument('--find_topK', type=int, default=10) 
parser.add_argument('--abl_start', type=int, default=10)
parser.add_argument('--abl_freq', type=int, default=10)
parser.add_argument('--use_mined_rule', type=str2bool, default=True) 
parser.add_argument('--accumulate_new_alignment', type=bool, default=True) 
parser.add_argument('--use_new_negative_alignment', type=bool, default=True)

parser.add_argument('--neg_pair_margin', type=float, default=0.1) 

parser.add_argument('--load_model_dir', type=str, default=None) 

args = parser.parse_args()
print(args)


class ModelFamily(object):
    BasicModel = BasicModel

    AlignE = AlignE


def get_model(model_name):
    return getattr(ModelFamily, model_name)


if __name__ == '__main__':
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered)
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()

    if args.load_model_dir is not None:
        model.load() 
    else:
        model.run()
        model.test()
        model.save()
