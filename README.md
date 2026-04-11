# eta3

This is package gives methods for calculating terms of the form $\eta^{\left(m\right)}=\sum_{i,j=1}^{N}\pi_{i}p_{ij}^{\left(m\right)}\eta_{ij}$ and $\eta^{\left(m,n\right)}=\sum_{i,j,k=1}^{N}\pi_{i}p_{ij}^{\left(m\right)}p_{jk}^{\left(n\right)}\eta_{ijk}$, where the $\eta$ terms are defined by $\eta_{i}=0$ for $i=1,\dots ,N$,

$$\eta_{ij} = \frac{1}{2} + \frac{1}{2}\sum_{\ell =1}^{N} p_{i\ell}\eta_{\ell j} + \frac{1}{2}\sum_{\ell =1}^{N} p_{j\ell}\eta_{i\ell}$$

for $i,j=1,\dots ,N$ with $i\neq j$, and

$$\eta_{ijk} = \frac{1}{3} + \frac{1}{3}\sum_{\ell =1}^{N} p_{i\ell}\eta_{\ell jk} + \frac{1}{3}\sum_{\ell =1}^{N} p_{j\ell}\eta_{i\ell k} + \frac{1}{3}\sum_{\ell =1}^{N} p_{k\ell}\eta_{ij\ell}$$

for $i,j,k=1,\dots ,N$ with $i\neq j\neq k\neq i$.

## Installation

```bash
pip install -e .
```

## Example

```python
import eta3
from eta3.utils import create_large_graph

# create test graph
adj_matrix = create_large_graph('erdos_renyi', n=30, p=0.4)

# solve eta system
result = eta3.solve_eta_system(adj_matrix)

# access results
eta_ij = result['eta_ij']
eta_ijk = result['eta_ijk']
```

## Verifying symbolic claims in the paper

```
pip install sympy
python scripts/selection_condition.py
```

## Reproducing network calculations in the paper

```
python scripts/sigma_for_networks.py
```