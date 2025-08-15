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
