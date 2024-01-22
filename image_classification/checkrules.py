import numpy as np
from multiprocessing import Pool
from functools import partial
import time

class CheckRules():
    def __init__(self, filename, name_rule_file = ''):
        self.rule_list = []
        self.name_rule_list = []
        if filename is None:
            return
        with open(filename) as fin:
            for rule in fin:
                self.rule_list.append(eval(rule))
        if len(name_rule_file) > 0:
            with open(name_rule_file) as fin:
                for rule in fin:
                    self.name_rule_list.append(eval(rule))
        

    def __satisfied(self, status, rule):
        ret = np.full(len(status), True)
        for r in rule:
            if r[0] == 1:
                ret &= status[:,r[1]] == r[2]
            else:
                ret &= status[:,r[1]] != r[2]
        return ret

    def get_rule_conf(self):
        conf_list = [rule[2] for rule in self.rule_list]
        return np.array(conf_list)

    def update_rule_by_idxs(self, idxs, conf_list = []):
        new_rule_list = []
        new_name_rule_list = []
        if len(conf_list) == 0:
            for idx in idxs:
                new_rule_list.append(self.rule_list[idx])
                if len(self.name_rule_list) > 0:
                    new_name_rule_list.append(self.name_rule_list[idx])
        else:
            assert(len(idxs)==len(conf_list))
            for idx, conf in zip(idxs, conf_list):
                new_rule_list.append((self.rule_list[idx][0], self.rule_list[idx][1], conf))
                if len(self.name_rule_list) > 0:
                    new_name_rule_list.append((self.name_rule_list[idx][0], self.name_rule_list[idx][1], conf))
        self.rule_list = new_rule_list
        self.name_rule_list = new_name_rule_list


    # def __satisfied(self, status, rule):
    #     for r in rule:
    #         if r[0] == 1 and status[r[1]] != r[2]:
    #             return False
    #         if r[0] == 0 and status[r[1]] == r[2]:
    #             return False
    #     return True

    # Ensure one hot encoding
    def judge(self, X, Y_onehot, log = False, idx2ml_dict = {}):
        X = np.array(X)
        Y_onehot = np.array(Y_onehot)
        consistency_list = np.zeros(len(X))
        XY = np.concatenate((X, Y_onehot), axis=1)
        for idx, rule in enumerate(self.rule_list):
            left, right, conf = rule
            sat_left_list = self.__satisfied(XY, left)
            sat_right_list = self.__satisfied(XY, right)
            unsat_list = sat_left_list & ~sat_right_list
            # print(sat_left_list, sat_right_list, unsat_list, (~unsat_list)*conf)
            if log:
                for x, y in zip(X[unsat_list], Y_onehot[unsat_list]):
                    if idx2ml_dict:
                        print("Violated rule %d"%(idx+1), end=' ')
                        for literal in left:
                            print("(%d,%s,%d)"%(literal[0],idx2ml_dict[literal[1]],literal[2]), end=' ')
                        print("->", end=' ')
                        for literal in right:
                            print("(%d,%s,%d)"%(literal[0],idx2ml_dict[literal[1]],literal[2]), end=' ')
                        print(conf, ": \t", x, y)
                    else:
                        print("Violated rule %d"%(idx+1), left, right, conf, ": \t", x, y)
            consistency_list += (~unsat_list)*(2*conf-1)#####conf
        return consistency_list
    
    # Return the confidence and violated samples a each rule
    def _eval_a_rule(self, rule, XY):
        left, right, _ = rule
        left_hold_cnt, total_hold_cnt = 0, 0
        sat_left_list = self.__satisfied(XY, left)
        sat_right_list = self.__satisfied(XY, right)
        left_hold_cnt = np.sum(sat_left_list)
        total_hold_cnt = np.sum(sat_left_list&sat_right_list)
        return left_hold_cnt, total_hold_cnt

    # Return the confidence and violated samples of each rule
    def eval_rules(self, X, Y):
        XY = np.concatenate((X, Y), axis=1)
        left_hold_cnt_list, total_hold_cnt_list = [], []
        for rule in self.rule_list:
            left_hold_cnt, total_hold_cnt = self._eval_a_rule(rule, XY)
            left_hold_cnt_list.append(left_hold_cnt)
            total_hold_cnt_list.append(total_hold_cnt)

        # partial_eval_a_rule = partial(self.eval_a_rule, XY=XY)
        # pool = Pool(processes = 50)
        # ret = pool.map(partial_eval_a_rule, self.rule_list)
        # pool.close()
        # pool.join()
        # left_hold_cnt_list, total_hold_cnt_list = [r[0] for r in ret], [r[1] for r in ret]

        conf_list = np.array(total_hold_cnt_list) / np.array(left_hold_cnt_list)
        vio_list = np.array(left_hold_cnt_list) - np.array(total_hold_cnt_list)
        return conf_list, vio_list


if __name__ == "__main__":
    check = CheckRules("zoo/rule_format.txt")
    status = np.array([0,0,1,0,0,1,1,1,1,1,1,0,4,0,0,0])
    cate = 5
    onehot1 = np.eye(7)[cate-1]
    cate = 6
    onehot2 = np.eye(7)[cate-1]
    print(check.judge([status,status], [onehot1,onehot2], True)) # 30 # 28

    # print(check.eval_rules([status,status], [onehot1,onehot2]))
