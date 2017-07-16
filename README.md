# PhyloInfer

The goal of this package is to develop efficient approximate inference methods for Bayesian phylogenetics. So far, 
we have implemented a [probablistic path Hamiltonian Monte Carlo](https://arxiv.org/abs/1702.07814) method for Bayesian phylogenetic inference.


# Prerequisites and Installation

PhyloInfer is a python package that is built on ETE toolkit and biopython

* [ete3](http://etetoolkit.org)
* [Bio](http://biopython.org)

To install, simple run
```sh
pip install phyloinfer
```
Please install pyTQDist if you want to use the related features.

# Examples

In what follows, we present some simple examples on simulated data and real data. 

## Simulated Dataset

First, we simulate data from a given phylogeny tree with 50 tips

```python
# load modules
import phyloinfer as pinf
import numpy as np

# set model parameters
pden = np.array([.25,.25,.25,.25])

# decompose the rate matrix (JC model)
D, U, U_inv, rate_matrix = pinf.rateM.decompJC()

# sample a random tree from the prior
ntips = 50
true_tree = pinf.tree.create(ntips, branch='random')

# simulate Data
data = pinf.data.treeSimu(true_tree, D, U, U_inv, pden, 1000)
```

Now, you may want to take a look at the negative log-posterior or the log-likelihood of the true tree

```python
L = pinf.Loglikelihood.initialCLV(data)
true_branch = pinf.branch.get(true_tree)
print "The negative log-posterior of the true tree: {}".format(pinf.Logposterior.Logpost(true_tree, true_branch, D, U, U_inv, pden, L))
print "The log-likelihood of the true tree: {}".format(pinf.Loglikelihood.phyloLoglikelihood(true_tree, true_branch, D, U, U_inv, pden, L))
```

Next, we sample a starting tree from the prior

```python
init_tree = pinf.tree.create(ntips, branch='random')
```

Again, you may want to see its negative log-posterior or log-likelihood

```python
init_branch = pinf.branch.get(init_tree)
print "The negative log-posterior of the init tree: {}".format(pinf.Logposterior.Logpost(init_tree, init_branch, D, U, U_inv, pden, L))
print "The log-likelihood of the init tree: {}".format(pinf.Loglikelihood.phyloLoglikelihood(init_tree, init_branch, D, U, U_inv, pden, L))
```

Now, we are ready to run ppHMC to sample from the posterior!!!

```python
samp_res = pinf.phmc.hmc(init_tree, init_branch, (pden, 1.0), data, 100, 0.001, 100, subModel='JC', surrogate=True,  burnin_frac=0.2, adap_stepsz_rate = 0.4, delta=0.002, monitor_event=True, printfreq=50)
```

## Primates Dataset

Load primates data set

```python
data, taxon = pinf.data.loadData('../datasets/primates.nex','nexus')
```

Again, initialize the tree from the prior

```python
ntips = len(taxon)
init_tree = pinf.tree.create(ntips, branch='random')
init_branch = pinf.branch.get(init_tree)
```

Run ppHMC to sample from the posterior

```python
samp_res = pinf.phmc.hmc(init_tree, init_branch, (pden, 1.0), data, 100, 0.004, 100, subModel='JC', surrogate=True, burnin_frac=0.5, delta=0.008, adap_stepsz_rate=0.8, printfreq=20)
```

For more details, see the notebooks in examples.
