class UnionFind:
    def __init__(self, elements):
        self.parent = {element: element for element in elements}
        self.rank = {element: 0 for element in elements}

    def find(self, element):
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])  # Path compression
        return self.parent[element]

    def union(self, element1, element2):
        root1 = self.find(element1)
        root2 = self.find(element2)
        
        if root1 != root2:
            # Union by rank
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

# Initialize Union-Find structure
uf = UnionFind(names)

max_edits = 2

# Merge clusters based on similarity
for name in names:
    similar_names = trie.within_edits(name, max_edits)
    for similar_name in similar_names:
        if similar_name != name:
            uf.union(name, similar_name)

# Collect clusters
from collections import defaultdict

clusters = defaultdict(set)
for name in names:
    root = uf.find(name)
    clusters[root].add(name)

# Print the resulting clusters
for cluster_rep, cluster in clusters.items():
    print(f"Cluster representative: {cluster_rep}")
    print(f"Names in cluster: {cluster}\n")
