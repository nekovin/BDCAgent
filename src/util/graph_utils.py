import networkx as nx
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

def visualize_causal_graph(G: nx.DiGraph, output_path: str = None):
    """
    Visualize a networkx DiGraph using matplotlib and optionally save it.
    
    Args:
        G: networkx DiGraph object
        output_path: Optional path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Create layout
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=2000, alpha=0.6)
    
    # Get edge weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Normalize edge weights for visualization
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 2 + 1 
                            for w in edge_weights]
    else:
        normalized_weights = []
    
    # Draw edges with weights determining thickness
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          width=normalized_weights,
                          arrowsize=20)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    # Add edge labels (weights)
    edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' 
                  for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.title("Causal Relationship Graph")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Graph saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def export_graph_json(G: nx.DiGraph, output_path: str):
    """
    Export the graph structure to JSON for visualization in web interfaces.
    
    Args:
        G: networkx DiGraph object
        output_path: Path to save the JSON file
    """
    # Create layout
    pos = nx.spring_layout(G)
    
    # Convert node positions to list
    nodes = [{"id": node, 
              "x": float(pos[node][0] * 500 + 400),  # Scale and center
              "y": float(pos[node][1] * 300 + 300)} 
             for node in G.nodes()]
    
    # Convert edges to list
    links = [{"source": u, 
              "target": v, 
              "weight": float(G[u][v]["weight"])} 
             for u, v in G.edges()]
    
    # Create graph data structure
    graph_data = {
        "nodes": nodes,
        "links": links
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"Graph data exported to {output_path}")

def save_causal_graph(G: nx.DiGraph, base_path: str):
    """
    Save causal graph in multiple formats.
    
    Args:
        G: networkx DiGraph object
        base_path: Base path for saving files (without extension)
    """
    # Create base path if it doesn't exist
    Path(base_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save visualization as PNG
    visualize_causal_graph(G, f"{base_path}.png")
    
    # Export graph data as JSON
    export_graph_json(G, f"{base_path}.json")
    
    # Save graph in GraphML format for later loading
    nx.write_graphml(G, f"{base_path}.graphml")