
# Data Manipulation

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
import numpy as np

nucnames = ['A','G','C','T']

def loadData(filename,data_type):
    data = []
    id_seq = []
    for seq_record in SeqIO.parse(filename,data_type):
        id_seq.append(seq_record.id)
        data.append(list(seq_record.seq))

    return data, id_seq

def saveData(data, id_seq, filename, data_type):
    my_seq = []
    for i,seq in enumerate(data):
        my_seq.append(SeqRecord(Seq(''.join(seq),generic_dna),id=str(id_seq[i]),description=''))
    
    with open(filename,"w") as output_file:
        SeqIO.write(my_seq, output_file, data_type)

# sampler for a discrete distribution
def statSamp(pden, idx=False):
    cumsum = pden.cumsum()
    u = np.random.uniform()
    for i in range(4):
        if u <= cumsum[i]:
            if idx:
                return i
            else:
                return nucnames[i]

# simulate taxon sequences given the tree topology and rate matrices
def treeSimu(tree, D, U, U_inv, pden, ndata, missrate=0.0):
    ntips = len(tree)
    pt_matrix = [np.zeros((4,4)) for i in range(2*ntips-3)]
    
    # run tree traversal to acquire the transition matrices
    for node in tree.traverse("postorder"):
        if not node.is_root():
            # pt_matrix[node.name] = np.transpose(np.linalg.lstsq(U.T,
            #                     np.dot(np.diag(np.exp(D*node.dist)),U.T))[0])
            pt_matrix[node.name] = np.dot(U, np.dot(np.diag(np.exp(D*node.dist)), U_inv))
    
    simuData = []
    status = [''] * (2*ntips-2) 
    for run in range(ndata):
        for node in tree.traverse("preorder"):
            if node.is_root():
                status[node.name] = statSamp(pden,idx=True)
            else:
                status[node.name] = statSamp(pt_matrix[node.name][status[node.up.name]],idx=True)
            
        simuData.append([nucnames[i] if np.random.uniform() > missrate else '-' for i in status[:ntips]])
    
    return np.transpose(simuData)
    