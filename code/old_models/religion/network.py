import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def nx_test():
    # Create graph
    G = nx.Graph()
    
    # Add nodes to graph
    G.add_node(1)
    G.add_nodes_from([2, 3])
    G.add_nodes_from([(4, {"cultist": True}), (5, {"cultist": False})])
    # Create new graph H and add its nodes to G
    H = nx.path_graph(10)
    G.add_nodes_from(H)
    # Or make the whole graph H a single node in G
    G.add_node(H)
    
    # Edges
    G.add_edge(1, 2)  # Edge between node 1 and 2
    G.add_edges_from([(1, 2), (1, 3)])    
    # Edge bunch
    ebunch = (2, 3, {"crisis strength": 5})
    G.add_edges_from([ebunch])
    
    # Examine elements
    nodes_list = list(G.nodes)
    edges_list = list(G.edges)
    nbors_of_node_2 = list(G.neighbors(2))  # List the neighbour nodes of node 2
    degree_of_node_1 = G.degree[1]  # Degree of node 1
    nbunch_ex = G.edges([2, 3])  # Edges of node 2 and 3
    
    print("Nodes: \t", nodes_list)
    print("Edges: \t", edges_list)
    print("Nbors of node 2: \t", nbors_of_node_2)  
    print("Degree of node 1: \t", degree_of_node_1)
    print("nbuch 1 and 3: \t", nbunch_ex)
    
    # Create Erdos-Renyi graph
    er = nx.erdos_renyi_graph(n=20, p=0.15)
    
    # Draw graph
    subax1 = plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight="bold")
    subax2 = plt.subplot(122)
    nx.draw(er, with_labels=True, font_weight="bold")
    plt.show()


class ConversionPropagation():
    def __init__(self, N_tot: int, N_missionary: int, time_steps: int):
        self.N_tot = N_tot
        self.N_missionary = N_missionary
        self.time_steps = time_steps
        
        
        self.rng = np.random.default_rng()
        
        return 

    
    # Helper functions
    def _conversion_strength(self, graph, node):
        """
        For a given node, go through all adjacent nodes (i.e. connected nodes) and calculate the conversion strength of the target node
        NOTE need some sort of accounting for friends vs missionaries

        Args:
            graph (networkx graph): Current graph
            node (int): Target node

        Returns:
            float: Conversion strength of the target node
        """
        convert_crisis_strength = graph.nodes[node]["crisis_strength"]
        print("Convert : ", graph.nodes[node])
        conversion_strength = 0
        for nbor in graph.adj[node]:
            print("datadict ", graph.nodes[nbor])
            nbor_charisma = graph.nodes[nbor]["charisma"]
            conversion_strength += convert_crisis_strength * nbor_charisma
        
        return conversion_strength
    
    
    def _remove_add_edges(self):
        
        return

    
    def _crisis_strength_random_walk(self):
        
        return
    
    
    def _convert_charisma_update(self):
        
        return
    

    def _check_who_converts(self):
        
        return

    def initial_network(self):
        # Erdos-Renyi    
        G = nx.erdos_renyi_graph(n=self.N_tot, p=0.1, seed=42)
        # G = nx.path_graph(n=self.N_tot)
        
        # Attributes and initial values
        attributes_dict = {"missionary": False, 
                           "crisis_strength": 0,
                           "charisma": 0}        
        for key in attributes_dict:
            nx.set_node_attributes(G, attributes_dict[key], key)
        # Loop over nodes and make the first N_missionary to cultists and the rest to non-cultists
        for node in G.nodes:
            if node < self.N_missionary: 
                G.nodes[node]["missionary"] = True
                G.nodes[node]["charisma"] = self.rng.random()  # Charisma random number between 0 and 1. NOTE might change
            else:
                G.nodes[node]["crisis_strength"] = self.rng.random()  # NOTE might change
        
        return G
        

    def update(self):
        
        self._remove_add_edges()
        self._crisis_strength_random_walk()
        self._convert_charisma_update()
        self._check_who_converts()
        
        
        return


    def evolve(self):
        G = self.initial_network()
        
        for i in range(1, self.time_steps):
            self.update()
        
        return 
    
    
    def plot_graph(self, graph):
        subax1 = plt.subplot(121)
        nx.draw(graph, with_labels=True, font_weight="bold")
        plt.show()
        
        
    def test(self):
        G = self.initial_network()
        conversion_strength_node_1 = self._conversion_strength(G, node=5)
        
        print("Conversion strength node 1: ", conversion_strength_node_1)
            

if __name__ == "__main__":
    CultClass = ConversionPropagation(N_tot = 10, N_missionary=5, time_steps=10)
    
    # G = CultClass.initial_network()
    # CultClass.plot_graph(G)
    CultClass.test()
    