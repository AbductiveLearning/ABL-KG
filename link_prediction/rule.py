class Rule:
    def __init__(self, rule_body_left, rule_body_right, rule_body_middle, rule_head, conf):
        self.rule_body_left = rule_body_left
        self.rule_body_right = rule_body_right
        self.rule_body_middle = rule_body_middle
        self.rule_head = rule_head
        self.conf = conf
        assert len(self.rule_head) == 1
        assert len(self.rule_body_middle) == 1
        self.variables = set()
        for body in rule_body_left + rule_body_right + rule_body_middle:
            self.variables.add(body[0])
            self.variables.add(body[2])
            if body[1] != 'equalTo':
                assert type(body[1]) == int
        assert rule_head[0][0] in self.variables
        assert rule_head[0][1] == 'equalTo'
        assert rule_head[0][2] in self.variables
        # assert len(set([rule_head[0][0], rule_head[0][2], rule_body_middle[0][0], rule_body_middle[0][2]])) == 4
        
    def __str__(self):
        return "{}=>{},conf:{}".format(self.rule_body_left + self.rule_body_right + self.rule_body_middle, self.rule_head, self.conf)
    
    def __eq__(self, other_rule):
        if len(self.rule_body_left) == 1 and len(self.rule_body_right) == 1 and len(other_rule.rule_body_left) == 1 and len(other_rule.rule_body_right) == 1:
            if self.rule_body_left[0][1] == self.rule_body_right[0][1] and other_rule.rule_body_left[0][1] == other_rule.rule_body_right[0][1] and self.rule_body_left[0][1] == other_rule.rule_body_left[0][1]:
                this_variable_map = {}
                this_variable_map[self.rule_body_left[0][0]] = other_rule.rule_body_left[0][0]
                this_variable_map[self.rule_body_left[0][2]] = other_rule.rule_body_left[0][2]
                
                this_variable_map[self.rule_body_right[0][0]] = other_rule.rule_body_right[0][0]
                this_variable_map[self.rule_body_right[0][2]] = other_rule.rule_body_right[0][2]
                
                if this_variable_map[self.rule_body_middle[0][0]] == other_rule.rule_body_middle[0][0] and this_variable_map[self.rule_body_middle[0][2]] == other_rule.rule_body_middle[0][2]:
                    if this_variable_map[self.rule_head[0][0]] == other_rule.rule_head[0][0] and this_variable_map[self.rule_head[0][2]] == other_rule.rule_head[0][2]:
                        return True
        return False
        

class FuncRule(Rule):
    def __init__(self, rule_body_left, rule_body_right, rule_body_middle, rule_head, r1_func, r2_func):
        super(FuncRule, self).__init__(rule_body_left, rule_body_right, rule_body_middle, rule_head, r1_func * r2_func)
        self.r1_func = r1_func
        self.r2_func = r2_func
        assert len(self.rule_body_left) == 1
        assert len(self.rule_body_right) == 1
        

class Grounding:
    def __init__(self, rule, instantiation):
        assert rule.variables == instantiation.keys()
        self.rule = rule
        self.instantiation = instantiation
        self.inferred_entity_alignment = (instantiation[self.rule.rule_head[0][0]], instantiation[self.rule.rule_head[0][2]])
        self.inferred_by_entity_alignment = (instantiation[self.rule.rule_body_middle[0][0]], instantiation[self.rule.rule_body_middle[0][2]])
