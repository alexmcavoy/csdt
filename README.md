# eta3

I focused on efficiency, so please edit according to the SI! Happy to do this together but I wanted to get you the first stab at the code.

## Installation

```bash
pip install -e .
```

## Usage

```python
import eta3
from eta3.utils import create_large_graph

# Create test graph
adj_matrix = create_large_graph('erdos_renyi', n=30, p=0.4)

# Solve eta system
result = eta3.solve_eta_system(adj_matrix)

# Access results
eta_ij = result['eta_ij']
eta_ijk = result['eta_ijk']
```