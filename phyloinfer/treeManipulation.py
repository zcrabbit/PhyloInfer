import numpy as np
from phyloinfer import Tree



# Tree initialization: set the node name to the sequence order
def init(tree, branch=None, name='all', scale=0.1, display=False, return_map=False):
    if return_map: idx2node = {}
    i, j = 0, len(tree)
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            if name != 'interior':
                node.name, i = i, i+1
            else:
                node.name = int(node.name)
        else:
            node.name, j = j, j+1
        if not node.is_root():
            if isinstance(branch, basestring) and branch=='random':
                node.dist = np.random.exponential(scale)
            elif branch is not None:
                node.dist = branch[node.name]
        else:
            node.dist = 0.0
            
        if return_map: idx2node[node.name] = node
        if display:
            print node.name, node.dist
        
    if return_map: return idx2node
    
    
def create(ntips, branch='random', scale=0.1):
    tree = Tree()
    tree.populate(ntips)
    tree.unroot()
    init(tree, branch=branch, scale=scale)
    
    return tree
    

def namenum(tree, taxon):
    taxon2idx = {}
    j = len(taxon)
    for i, name in enumerate(taxon):
        taxon2idx[name] = i
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            assert type(node.name) is str, "The taxon name should be strings"
            node.name = taxon2idx[node.name]
        else:
            node.name, j = j, j+1
    

def nametaxon(tree, taxon):
    for node in tree.traverse("postorder"):
        if node.is_leaf():
            assert type(node.name) is int, "The taxon name should be integers"
            node.name = taxon[node.name]            


def is_internal(node):
    return not node.is_leaf() and not node.is_root()


# obtain a dictionary of nodes
def idx2nodeMAP(tree):
    idx2node = {}
    for node in tree.traverse("postorder"):
        idx2node[node.name] = node
    return idx2node


# perform NNI on interior branches
def NNI(node, include=False):
    if node.is_root() or node.is_leaf():
        print "Can not perform NNI on root or leaf branches!"
    else:
        if include:
            neighboor = np.random.randint(3)
            xchild = node.get_sisters()[0]
            parent = node.up
            if neighboor == 1:
                ychild = node.children[0]
            if neighboor == 2:
                ychild = node.children[1]
            if neighboor:
                xchild.detach()
                ychild.detach()
                parent.add_child(ychild)
                node.add_child(xchild)
                return 1
        else:
            neighboor = np.random.randint(2)
            xchild = node.get_sisters()[0]
            parent = node.up
            ychild = node.children[neighboor]
            xchild.detach()
            ychild.detach()
            parent.add_child(ychild)
            node.add_child(xchild)
            return 1
    
    return 0

                        
def copy(tree, branch=None):
    copy_tree = tree.copy('newick')
    for node in copy_tree.traverse('postorder'):
        node.name = int(node.name)
        if not node.is_root() and branch:
            node.dist = branch[node.name]
                
    return copy_tree                    