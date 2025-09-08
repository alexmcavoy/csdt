import numpy as np
import networkx as nx


def create_large_graph(graph_type, n, **kwargs):
    generators = {
        "erdos_renyi": lambda: nx.erdos_renyi_graph(n, kwargs.get("p", 0.5)),
        "barabasi_albert": lambda: nx.barabasi_albert_graph(n, kwargs.get("m", 2)),
        "watts_strogatz": lambda: nx.watts_strogatz_graph(
            n, kwargs.get("k", 4), kwargs.get("p", 0.3)
        ),
    }

    if graph_type not in generators:
        raise ValueError(f"Unknown graph type: {graph_type}")

    G = generators[graph_type]()

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            G.add_edge(list(components[i])[0], list(components[i + 1])[0])

    return nx.to_numpy_array(G, dtype=int)

def create_test_graphs():
    return {
        "triangle": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        "path": np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]),
        "square": np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]]),
        "star": np.array(
            [[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
        ),
        "complete4": np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]),
    }
