import numpy as np
# tree branch length manipulations

def update(tree, dbranch):
    for node in tree.traverse("postorder"):
        if not node.is_root():
            node.dist += dbranch[node.name]

def get(tree):
    # branch = [0] * (2*len(tree)-3)
    branch = np.zeros(2*len(tree)-3)
    for node in tree.traverse("postorder"):
        if not node.is_root():
            branch[node.name] = node.dist
    return branch

def set(tree,branch):
    for node in tree.traverse("postorder"):
        if not node.is_root():
            node.dist = branch[node.name]