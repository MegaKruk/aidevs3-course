import networkx as nx
from pyvis.network import Network
import os


def visualize_lightrag_graph(graphml_path="./lightrag_data/graph_chunk_entity_relation.graphml"):
    if not os.path.exists(graphml_path):
        print(f"Error: Graph file not found at {graphml_path}")
        return

    # 1. Load the graph
    print(f"Loading graph from {graphml_path}...")
    G = nx.read_graphml(graphml_path)

    # 2. Initialize PyVis Network
    # '1000px' height, '100%' width, dark mode, directed
    net = Network(height='1000px', width='100%', bgcolor='#222222', font_color='white', directed=True)

    # 3. Customize nodes based on their attributes
    for node, attrs in G.nodes(data=True):
        # LightRAG usually stores 'entity_type' or similar
        label = node
        title = f"Type: {attrs.get('entity_type', 'Unknown')}\nDescription: {attrs.get('description', 'N/A')}"

        # Color coding for the "Wow" factor
        node_type = attrs.get('entity_type', '').lower()
        color = '#97c2fc'  # Default blue

        if 'person' in node_type:
            color = '#ffcc00'  # Gold for people
        elif 'organization' in node_type:
            color = '#00ffcc'  # Teal for companies
        elif 'concept' in node_type:
            color = '#ff99cc'  # Pink for tech/concepts
        elif 'event' in node_type:
            color = '#99ff99'  # Green for events/projects

        net.add_node(node, label=label, title=title, color=color)

    # 4. Add edges
    for source, target, attrs in G.edges(data=True):
        description = attrs.get('description', '')
        net.add_edge(source, target, title=description, color='#ffffff', arrowStrikethrough=False)

    # 5. Set physics for a nice "floating" effect
    net.force_atlas_2based()

    output_file = "knowledge_graph.html"
    net.show(output_file, notebook=False)
    print(f"Success! Open {output_file} in your browser to see the graph.")

    # 6. Pretty printing to terminal
    G = nx.read_graphml("./lightrag_data/graph_chunk_entity_relation.graphml")

    print(f"Knowledge Graph Summary:")
    print(f"Total Entities: {G.number_of_nodes()}")
    print(f"Total Relationships: {G.number_of_edges()}")

    print("\nTop 10 Entities & Types:")
    for i, (node, attrs) in enumerate(list(G.nodes(data=True))[:10]):
        print(f"- {node} ({attrs.get('entity_type', 'N/A')})")


if __name__ == "__main__":
    visualize_lightrag_graph()
