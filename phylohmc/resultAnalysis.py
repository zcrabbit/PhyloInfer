import matplotlib.pyplot as plt
import numpy as np
import re
from Bio import Phylo
from cStringIO import StringIO
from .treeManipulation import init
from phylohmc import Tree

rx = {
    'tbr-approx': re.compile('(?<=a_TBR:\s)\d+'),
    'tbr': re.compile('(?<=d_TBR = )\d+'),
    'replug': re.compile('(?<=d_R = )\d+'),
    'uspr': re.compile('(?<=d_USPR = )\d+')
}

def treeStats(sampled_tree, target_tree_dict=None):
    if not target_tree_dict:
        tree_samp_stats = {}
        for tree in sampled_tree:
            seen = 0
            for exist_tree in tree_samp_stats:
                if tree.robinson_foulds(exist_tree,unrooted_trees=True)[0] == 0:
                    seen = 1
                    tree_samp_stats[exist_tree] += 1
                    break
            if not seen:
                tree_samp_stats[tree] = 1
    else:
        tree_samp_stats = {tree:0.0 for tree in target_tree_dict}
        for tree in sampled_tree:
            for target in tree_samp_stats:
                if tree.robinson_foulds(target_tree_dict[target],unrooted_trees=True)[0] == 0:
                    tree_samp_stats[target] += 1
                    break
        

    tree_samp_count = len(sampled_tree)*1.0
    for tree_samp in tree_samp_stats:
        tree_samp_stats[tree_samp] /= tree_samp_count
    return sorted(tree_samp_stats.items(), key=lambda x:x[1], reverse=True)


def savePara(sampled_branch, filename):
    import uuid
    branch_count = len(sampled_branch[0])
    with open(filename,'w') as output_file:
        output_file.write('[ID:{}]\n'.format(uuid.uuid4()))
        output_file.write('nIter' + '\t' + '\t'.join(['length[{}]'.format(i) for i in range(branch_count)]) + '\n')
        for i, branch in enumerate(sampled_branch):
            output_file.write(str(i) + '\t' + '\t'.join([str(branch) for branch in sampled_branch[i]]) + '\n')

            
def readPara(filename):
    with open(filename,'r') as readin_file:
        id_line = readin_file.readline()
        ID = ''.join(re.split('\[|\]|ID:',id_line)).strip()
        name_line = readin_file.readline()
        names = name_line.strip('\n').split('\t')
        stats_dict = {name:[] for name in names}
        
        while True:
            line = readin_file.readline()
            if line == "": 
                break
            for i, stats in enumerate(line.strip('\n').split('\t')):
                stats_dict[names[i]].append(float(stats))
            
    return stats_dict, names, ID 


def saveTree(sampled_tree, filename, tree_format):
    if type(sampled_tree) is not list:
        sampled_tree = [sampled_tree]
        
    with open(filename,'w') as output_file:
        for tree in sampled_tree:
            tree_newick = tree.write(format=tree_format)
            output_file.write(tree_newick + '\n')
    

def uspr(t1, t2, metric):
    import subprocess as sp
    t1 = t1.write(format=9)
    t2 = t2.write(format=9)
    
    p = sp.Popen(['./uspr', '--' + metric], stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    p.stdin.write(t1 + '\n' + t2)
    p.stdin.close()
    p.wait()
    return int(rx[metric].search(p.stdout.read().decode()).group(0))


def mcmc_treeprob(filename, data_type, truncate=10):
    mcmc_samp_tree_stats = Phylo.parse(filename, data_type)
    mcmc_samp_tree_dict = {}
    mcmc_samp_tree_name = []
    mcmc_samp_tree_wts = []
    num_hp_tree = 0
    for tree in mcmc_samp_tree_stats:
        handle = StringIO()
        Phylo.write(tree, handle,'newick')
        mcmc_samp_tree_dict[tree.name] = Tree(handle.getvalue().strip())
        init(mcmc_samp_tree_dict[tree.name],name='interior')
        handle.close()
        mcmc_samp_tree_name.append(tree.name)
        mcmc_samp_tree_wts.append(tree.weight)
        num_hp_tree += 1
        if num_hp_tree >= truncate: break
    
    return mcmc_samp_tree_dict, mcmc_samp_tree_name, mcmc_samp_tree_wts
    
        