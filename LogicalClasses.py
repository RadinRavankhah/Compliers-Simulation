class Terminal:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Terminal({self.name!r})"

    def __eq__(self, other):
        return isinstance(other, Terminal) and self.name == other.name

    def __hash__(self):
        return hash(('T', self.name))


class NonTerminal:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"NonTerminal({self.name!r})"

    def __eq__(self, other):
        return isinstance(other, NonTerminal) and self.name == other.name

    def __hash__(self):
        return hash(('N', self.name))
    
class Rule:
    def __init__(self, lhs: NonTerminal, rhs: list[Terminal | NonTerminal]):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} -> {' '.join(map(str, self.rhs))}"

    def __repr__(self):
        return f"Rule({self.lhs!r}, {self.rhs!r})"

    def __eq__(self, other):
        return isinstance(other, Rule) and self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self):
        return hash((self.lhs, tuple(self.rhs)))
    
class Grammar:
    def __init__(self, terminals: list[Terminal], non_terminals: list[NonTerminal], rules: list[Rule]):
        self.terminals = terminals
        self.non_terminals = non_terminals
        self.rules = rules

    def __str__(self):
        return "\n".join(map(str, self.rules))

class Dot:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __str__(self):
        return "."

    def __repr__(self):
        return "Dot()"

    def __eq__(self, other):
        return isinstance(other, Dot)

    def __hash__(self):
        return hash(('DOT',))

# convenient singleton reference used throughout:
DOT = Dot()

class LRRule:
    def __init__(self, lhs: NonTerminal, rhs: list[Terminal | NonTerminal | Dot]):
        self.lhs = lhs
        self.rhs = rhs
        
    def __str__(self):
        return f"{self.lhs} -> {' '.join(map(str, self.rhs))}"
    
    def make_from_rule(rule: Rule, dot_index_to_add: int):
        if dot_index_to_add > len(rule.rhs):
            raise ValueError("the index for dot cant be bigger than length of the rhs")
        
        lhs = rule.lhs
        rhs = [item for item in rule.rhs]
        rhs.insert(dot_index_to_add, Dot())
        
        return LRRule(lhs, rhs)

class State:
    def __init__(self, number: int, lr_rules: list[LRRule]):
        self.number = number
        self.lr_items = lr_rules
        
class GoTo:
    def __init__(self, from_state: State, to_state: State, character_item: Terminal | NonTerminal):
        self.from_state = from_state
        self.to_state = to_state
        self.character_item = character_item
                
class Parser:            
    def __init__(self, grammar: Grammar):
        self.grammar = grammar    # It's assumed that the first nonterminal in each grammar is always S
        local_s = [item for item in grammar.non_terminals if str(item.name) == "S"][0]
        local_s_prim = NonTerminal("S^")
        
        extended_nonterminals = [local_s_prim] + (grammar.non_terminals)
        print("________________")
        # for item in extended_nonterminals:
        #     print(item)
        
        # for item in [Rule(local_s_prim, [local_s])] + (grammar.rules):
        #     print(item)
        
        self.extended_grammar = Grammar(grammar.terminals, extended_nonterminals, [Rule(local_s_prim, [local_s])] + (grammar.rules))
        print(self.extended_grammar)

        self.states = self.create_items()
        for state in self.states:
            print(state)

    def create_items(self, lrrule: LRRule = None):
        lrrules = []
        
        # if lrrule == None:
        #     lrrule = LRRule.make_from_rule(self.extended_grammar.rules[0], 0)
        # else:
        #     for i in range(len(lrrule.rhs)):
        #         if type(lrrule.rhs[i]) == Dot:
        #             if i < len(lrrule.rhs) - 1:
        #                 lrrules.append(LRRule.make_from_rule(self.extended_grammar.rules[0], i + 1))
        #             else:
        #                 lrrules.append(LRRule.make_from_rule(self.extended_grammar.rules[0], i + 1))
                    
        
        
        # lrrules.append(lrrule)
        
        print("=============")
        i0_closures = self.find_closures([LRRule.make_from_rule(self.extended_grammar.rules[0], 0)])
        print(LRRule.make_from_rule(self.extended_grammar.rules[0], 0))
        for item in i0_closures:
            print(item)
            
        # for item in lrrules:
            # print(item)
            
        # lritem = LRItem()
        # lritem.rules_list = lrrules
        
        
        
        # return lrrules

    def find_closures(self, lrrules: list[LRRule]):
        closures_found = []
        for lrrule in lrrules:
            for item in lrrule.rhs:
                if type(item) == Dot and lrrule.rhs.index(item) < len(lrrule.rhs) - 1 and type(lrrule.rhs[lrrule.rhs.index(item) + 1]) == NonTerminal:
                    for rule in self.extended_grammar.rules:
                        if rule.lhs == lrrule.rhs[lrrule.rhs.index(item) + 1]:
                            closures_found.append(LRRule.make_from_rule(rule, 0))
                            closures_found.extend(self.find_closures([LRRule.make_from_rule(rule, 0)]))
        
        return closures_found
    
    def progress(self, state: State):
        pass

    def create_parse_table(self):
        pass
    
    def parse(self, string: str):
        pass
    
if __name__ == "__main__":
    S = NonTerminal("S")
    C = NonTerminal("C")
    A = NonTerminal("A")
    B = NonTerminal("B")
    
    a = Terminal("a")
    
    r1 = Rule(S, [C])
    r11 = Rule(S, [B])
    r2 = Rule(C, [A,B])
    r3 = Rule(A, [a])
    r4 = Rule(B, [a])
    
    # print(r1)
    # print(r2)
    # print(r3)
    # print(r4)
    
    terminals = [a]
    nonterminals = [S, C, A, B]
    rules = [r1, r11, r2, r3, r4]
    
    g1 = Grammar(terminals, nonterminals, rules)
    # print("----")
    print(g1)
    # print("------------------------")
    p1 = Parser(g1)
    
    # print(LRItem.make_from_rule(r1, 1))
    # print(LRItem.make_from_rule(r1, 0))
    # print(LRItem.make_from_rule(r1, 2))
    