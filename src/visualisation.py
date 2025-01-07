import networkx as nx
import matplotlib.pyplot as plt

def visualize_causal_graph(df, causation_agent):
    causal_graph = causation_agent.build_causal_graph(df)  # Assumes this method returns a list of tuples [(cause, effect), ...]
    if not causal_graph:
        print("No causal relationships detected.")
        return

    G = nx.DiGraph()
    G.add_edges_from(causal_graph)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
    plt.title("Causal Graph")
    plt.show()