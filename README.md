# PhyloInfer

The goal of this package is to develop efficient approximate inference methods for Bayesian phylogenetics. So far, 
we have implemented a probablistic path Hamiltonian Monte Carlo method for Bayesian phylogenetic inference.


# Prerequisites

PhyloInfer is a python package that is built on ETE toolkit and biopython

* [ete3](http://etetoolkit.org)
* [Bio](http://biopython.org)

# Examples

In what follows, we presents one simple example on simulated data. First, we simulate data from a given phylogeny tree with 50 tips

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

Now, you may want to take a look at the negative log-posterior or the log-likelihood of the true tree

```python
L = pinf.Loglikelihood.initialCLV(true_tree, data)
true_branch = pinf.branch.get(true_tree)
print "The negative log-posterior of the true tree: {}".format(pinf.Logposterior.Logpost(true_tree, true_branch, D, U, beta, pden, L))
print "The log-likelihood of the true tree: {}".format(pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, D, U, beta, pden, L))
```

Next, we sample a starting tree from the prior

```python
init_tree = pinf.Tree()
init_tree.populate(ntips)
init_tree.unroot()
pinf.tree.init(init_tree, branch='random')
```

Again, you may want to see its negative log-posterior or log-likelihood

```python
init_branch = pinf.branch.get(init_tree)
print "The negative log-posterior of the init tree: {}".format(pinf.Logposterior.Logpost(init_tree, init_branch, D, U, beta, pden, L))
print "The log-likelihood of the init tree: {}".format(pinf.Loglikelihood.phyloLoglikelihood(init_tree, init_branch, D, U, beta, pden, L))
```

Now, we are ready to run ppHMC to sample from the posterior!!!

```python
samp_res = pinf.phmc.hmc(init_tree, init_branch, (pden,kappa), data, 100, 0.0005, 100, subModel='HKY', burned=0.2, adap_stepsz_rate = 0.6)
```

