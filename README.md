# PhyloInfer

The goal of this package is to develop efficient approximate inference methods for Bayesian phylogenetics. So far, 
we have implemented a probablistic path Hamiltonian Monte Carlo method for Bayesian phylogenetic inference.


# Prerequisites

PhyloInfer is a python package that is built on ETE toolkit and biopython

* [ete3](http://etetoolkit.org)
* [Bio](http://biopython.org)

# Examples

The following example shows how to simulate data from a given phylogeny tree with $50$ tips

```python
# load modules
import phyloinfer as pinf
import numpy as np

# simulate Data
pden = np.array([.1,.2,.3,.4])
kappa = 2

# decompose the rate matrix (HKY model)
D, U, beta, rate_matrix = pinf.rateM.decompHKY(pden, kappa)

# sample a random tree from the prior
ntips = 50
true_tree = pinf.Tree()
true_tree.populate(ntips)
true_tree.unroot()
pinf.tree.init(true_tree, branch='random')

data = pinf.data.treeSimu(true_tree, D, U, beta, pden, 1000)
```


