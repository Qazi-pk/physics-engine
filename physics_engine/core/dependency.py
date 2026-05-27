from collections import defaultdict

class DependencyGraph:
    def __init__(self):
        self.graph = defaultdict(set)

    def add_law(self, law):
        for s in law.symbols():
            self.graph[s].add(law)

    def laws_for(self, symbols):
        result = set()
        for s in symbols:
            result |= self.graph.get(s, set())
        return result
