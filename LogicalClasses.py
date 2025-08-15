class Terminal:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
class NonTerminal:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name
    
class Rule:
    def __init__(self, lhs: NonTerminal, rhs: list[Terminal | NonTerminal]):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f"{self.lhs} -> {' '.join(map(str, self.rhs))}"
    
class Grammar:
    def __init__(self, terminals: list[Terminal], non_terminals: list[NonTerminal], rules: list[Rule]):
        self.terminals = terminals
        self.non_terminals = non_terminals
        self.rules = rules

    def __str__(self):
        return "\n".join(map(str, self.rules))

class Dot:
    value = "."
    
    def __str__(self):
        return self.value

class LRItem:
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
        
        return LRItem(lhs, rhs)

class Parser:
    def __init__(self, grammar: Grammar):
        self.grammar = grammar    # It's assumed that the first nonterminal in each grammar is always S
        local_s = [item for item in grammar.non_terminals if str(item.name) == "S"][0]
        local_s_prim = NonTerminal("S^")
        
        extended_nonterminals = [local_s_prim] + (grammar.non_terminals)
        print("________________")
        print(item for item in extended_nonterminals)
        
        self.extended_grammar = Grammar(grammar.terminals, extended_nonterminals, [Rule(local_s_prim, local_s)] + (grammar.rules))
        print(self.grammar.rules)

        self.states = self.create_states()

    def create_states(self):
        for rule in self.extended_grammar.rules:
            for i in range(len(rule.rhs)):
                print(LRItem.make_from_rule(rule, i))
        
    
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
    r2 = Rule(C, [A,B])
    r3 = Rule(A, [a])
    r4 = Rule(B, [a])
    
    print(r1)
    print(r2)
    print(r3)
    print(r4)
    
    terminals = [a]
    nonterminals = [S, C, A, B]
    rules = [r1, r2, r3, r4]
    
    g1 = Grammar(terminals, nonterminals, rules)
    
    p1 = Parser(g1)
    
    print(LRItem.make_from_rule(r1, 1))
    print(LRItem.make_from_rule(r1, 0))
    print(LRItem.make_from_rule(r1, 2))
    