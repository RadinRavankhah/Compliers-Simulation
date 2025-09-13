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

    def __repr__(self):
        return f"LRRule({self.lhs!r}, {self.rhs!r})"

    @staticmethod
    def make_from_rule(rule: Rule, dot_index_to_add: int):
        if dot_index_to_add < 0 or dot_index_to_add > len(rule.rhs):
            raise ValueError("the index for dot must be between 0 and len(rhs)")
        lhs = rule.lhs
        rhs = [item for item in rule.rhs]
        rhs.insert(dot_index_to_add, DOT)
        return LRRule(lhs, rhs)

    def dot_index(self):
        """Return the index of the Dot in rhs (or None if missing)."""
        for i, sym in enumerate(self.rhs):
            if isinstance(sym, Dot):
                return i
        return None

    def __eq__(self, other):
        return isinstance(other, LRRule) and self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self):
        return hash((self.lhs, tuple(self.rhs)))

class State:
    def __init__(self, number: int, lr_rules: list[LRRule]):
        self.number = number
        self.lr_items = lr_rules

    def __repr__(self):
        return f"State({self.number}, items={self.lr_items!r})"

    def __str__(self):
        return f"State {self.number}:\n" + "\n".join("  " + str(item) for item in self.lr_items)

class GoTo:
    def __init__(self, from_state: State, to_state: State, character_item: Terminal | NonTerminal):
        self.from_state = from_state
        self.to_state = to_state
        self.character_item = character_item
                
class Parser:
    def __init__(self, grammar: Grammar, debug: bool = False):
        """
        grammar: original grammar (non-augmented)
        debug: optional flag to print internal states (closures / transitions)
        """
        self.grammar = grammar
        self.debug = debug

        # choose start symbol as the first nonterminal (your comment said that's assumed)
        self.start_symbol = grammar.non_terminals[0]
        self.augmented_start = NonTerminal(self.start_symbol.name + "^")

        # build extended grammar: augmented start + original nonterminals/rules
        extended_nonterminals = [self.augmented_start] + grammar.non_terminals
        extended_rules = [Rule(self.augmented_start, [self.start_symbol])] + grammar.rules
        self.extended_grammar = Grammar(grammar.terminals, extended_nonterminals, extended_rules)

        # canonical collection (states) and transitions
        self.states: list[State] = []
        # transitions: map (state_number, symbol) -> state_number
        self.transitions: dict[tuple[int, Terminal | NonTerminal], int] = {}

        # build LR(0) canonical collection
        self._build_states()

        # compute FIRST/FOLLOW (FOLLOW used by SLR)
        self.first_sets = self.compute_first_sets()
        self.follow_sets = self.compute_follow_sets()

        # SLR parse table
        self.action_table = {}   # (state, terminal_name) -> ('s', j) | ('r', rule_index) | ('acc',)
        self.goto_table = {}     # (state, nonterminal_name) -> j
        self.create_parse_table()

        if self.debug:
            print("=== ACTION TABLE ===")
            for k, v in sorted(self.action_table.items()):
                print(k, ":", v)
            print("=== GOTO TABLE ===")
            for k, v in sorted(self.goto_table.items()):
                print(k, ":", v)

    # ---------- closure ----------
    def closure(self, items: set[LRRule]) -> set[LRRule]:
        closure = set(items)
        added = True
        while added:
            added = False
            new_items = set()
            for item in list(closure):
                idx = item.dot_index()
                if idx is None:
                    continue
                if idx < len(item.rhs) - 1:
                    X = item.rhs[idx + 1]
                    if isinstance(X, NonTerminal):
                        # for every production X -> gamma, add X -> . gamma
                        for rule in self.extended_grammar.rules:
                            if rule.lhs == X:
                                lr = LRRule.make_from_rule(rule, 0)
                                if lr not in closure:
                                    new_items.add(lr)
            if new_items:
                closure |= new_items
                added = True
        return closure

    # ---------- goto ----------
    def goto(self, items: set[LRRule], X: Terminal | NonTerminal) -> set[LRRule]:
        moved = set()
        for item in items:
            idx = item.dot_index()
            if idx is None:
                continue
            if idx < len(item.rhs) - 1 and item.rhs[idx + 1] == X:
                # create a copy of rhs and move dot one position to the right
                new_rhs = list(item.rhs)
                new_rhs[idx], new_rhs[idx + 1] = new_rhs[idx + 1], new_rhs[idx]
                moved.add(LRRule(item.lhs, new_rhs))
        return self.closure(moved) if moved else set()

    # ---------- build canonical collection ----------
    def _build_states(self):
        start_item = LRRule.make_from_rule(self.extended_grammar.rules[0], 0)
        I0 = self.closure({start_item})

        states = [I0]
        state_map = {frozenset(I0): 0}
        transitions = {}

        i = 0
        while i < len(states):
            I = states[i]
            # find all grammar symbols that appear immediately after a dot in I
            symbols_after_dot = set()
            for it in I:
                idx = it.dot_index()
                if idx is not None and idx < len(it.rhs) - 1:
                    symbols_after_dot.add(it.rhs[idx + 1])

            for X in symbols_after_dot:
                J = self.goto(I, X)
                if not J:
                    continue
                key = frozenset(J)
                if key not in state_map:
                    state_map[key] = len(states)
                    states.append(J)
                transitions[(i, X)] = state_map[key]
            i += 1

        # convert to State objects for nicer printing/use
        self.states = [State(num, list(itemset)) for num, itemset in enumerate(states)]
        self.transitions = transitions

        if self.debug:
            print("=== STATES ===")
            for s in self.states:
                print(s)
            print("=== TRANSITIONS ===")
            for k, v in sorted(transitions.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
                print(f"from state {k[0]} on {k[1]} -> state {v}")

    # ---------- FIRST sets (basic) ----------
    def compute_first_sets(self):
        terminals = self.grammar.terminals
        nonterminals = self.grammar.non_terminals
        rules = self.grammar.rules

        first = {nt: set() for nt in nonterminals}
        for t in terminals:
            first[t] = {t}

        # iterate until stable (this version assumes grammar without epsilon productions for simplicity)
        changed = True
        while changed:
            changed = False
            for rule in rules:
                A = rule.lhs
                rhs = rule.rhs
                # if rhs starts with a terminal, add it; if starts with nonterminal, add first(nonterminal)
                if not rhs:
                    continue
                first_sym = set()
                first_item = rhs[0]
                if isinstance(first_item, Terminal):
                    first_sym.add(first_item)
                else:
                    first_sym |= first[first_item]
                before = len(first[A])
                first[A] |= first_sym
                if len(first[A]) != before:
                    changed = True
        return first

    # ---------- FOLLOW sets ----------
    def compute_follow_sets(self):
        terminals = self.grammar.terminals
        nonterminals = self.grammar.non_terminals
        rules = self.grammar.rules

        follow = {nt: set() for nt in nonterminals}
        # end marker in follow of start symbol
        follow[self.start_symbol].add(Terminal('$'))

        changed = True
        while changed:
            changed = False
            for rule in rules:
                A = rule.lhs
                rhs = rule.rhs
                for i, B in enumerate(rhs):
                    if isinstance(B, NonTerminal):
                        beta = rhs[i+1:]
                        if not beta:
                            # add FOLLOW(A) to FOLLOW(B)
                            before = len(follow[B])
                            follow[B] |= follow[A]
                            if len(follow[B]) != before:
                                changed = True
                        else:
                            # FIRST(beta) - epsilon -> add to FOLLOW(B)
                            first_beta = set()
                            first_sym = beta[0]
                            if isinstance(first_sym, Terminal):
                                first_beta.add(first_sym)
                            else:
                                # nonterminal
                                first_beta |= self.first_sets.get(first_sym, set())
                            before = len(follow[B])
                            follow[B] |= first_beta
                            if len(follow[B]) != before:
                                changed = True
        return follow

    # ---------- SLR parse table ----------
    def create_parse_table(self):
        action = {}
        goto = {}

        for i, state in enumerate(self.states):
            for item in state.lr_items:
                idx = item.dot_index()
                if idx is None:
                    continue

                # Case 1: dot before a grammar symbol
                if idx < len(item.rhs) - 1:
                    a = item.rhs[idx + 1]
                    if isinstance(a, Terminal):
                        j = self.transitions.get((i, a))
                        if j is not None:
                            key = (i, a.name)
                            if key in action and action[key] != ('s', j):
                                # conflict - we simply print it for now (educational)
                                print("ACTION conflict at", key, "existing:", action[key], "new shift:", ('s', j))
                            action[key] = ('s', j)
                    else:
                        # nonterminal -> goto entry
                        j = self.transitions.get((i, a))
                        if j is not None:
                            goto[(i, a.name)] = j

                # Case 2: dot at the end -> reduce or accept
                else:
                    # check augmented accept
                    if item.lhs == self.augmented_start:
                        action[(i, '$')] = ('acc',)
                    else:
                        # produce reduction by finding the matching production index in extended_grammar.rules
                        prod_rhs = [sym for sym in item.rhs if not isinstance(sym, Dot)]
                        red_index = None
                        for idx_r, rule in enumerate(self.extended_grammar.rules):
                            if rule.lhs == item.lhs and rule.rhs == prod_rhs:
                                red_index = idx_r
                                break
                        if red_index is None:
                            raise RuntimeError("Could not find production for reduction: " + str(item))

                        # SLR: apply reduce on all terminals in FOLLOW(item.lhs)
                        follow_set = self.follow_sets.get(item.lhs, set())
                        for a in follow_set:
                            key = (i, a.name)
                            if key in action and action[key] != ('r', red_index):
                                print("ACTION conflict at", key, "existing:", action[key], "new reduce:", ('r', red_index))
                            action[key] = ('r', red_index)

        # store tables
        self.action_table = action
        self.goto_table = goto

    # ---------- parse using tables ----------
    def parse(self, input_str: str) -> bool:
        """Parse a space-separated string of terminal names, e.g. 'a a', return True on accept."""
        # tokenize input
        tokens = [Terminal(tok) for tok in input_str.split() if tok != ""]
        tokens.append(Terminal('$'))

        # stack of states and stack of symbols
        state_stack = [0]
        symbol_stack: list[Terminal | NonTerminal] = []

        ip = 0
        while True:
            state = state_stack[-1]
            a_name = tokens[ip].name
            act = self.action_table.get((state, a_name))
            if act is None:
                raise RuntimeError(f"Parse error at token {tokens[ip]!r} in state {state}")

            if act[0] == 's':  # shift
                j = act[1]
                symbol_stack.append(tokens[ip])
                state_stack.append(j)
                ip += 1
            elif act[0] == 'r':  # reduce
                r_idx = act[1]
                rule = self.extended_grammar.rules[r_idx]
                rhs_len = len(rule.rhs)
                # pop rhs_len symbols and states
                for _ in range(rhs_len):
                    if symbol_stack:
                        symbol_stack.pop()
                    state_stack.pop()
                # push LHS and goto
                symbol_stack.append(rule.lhs)
                t = state_stack[-1]
                goto_state = self.goto_table.get((t, rule.lhs.name))
                if goto_state is None:
                    raise RuntimeError("Goto missing after reduction")
                state_stack.append(goto_state)
            elif act[0] == 'acc':
                return True
            else:
                raise RuntimeError("Unknown action entry: " + str(act))


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
    