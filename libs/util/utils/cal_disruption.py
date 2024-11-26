import numpy as np 
from scipy import sparse
import torch
import logging
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial 



def calculate_disruptiveness_(i,references_dict_,citations_dict_):
    ascdnt = references_dict_[i]# reference/ asc
    dscdnt = citations_dict_[i]# citation  
    
    n_j = 0
    n_i=0
    n_k = set()
        
    for d in dscdnt:
        if references_dict_[d] & ascdnt:
            n_j+=1
        else:
            n_i+=1  

    for a in ascdnt:
        n_k = n_k.union(citations_dict_[a])  

    n_k = n_k - dscdnt

    n_k = len(n_k)

    if n_i+n_j+n_k ==0:
        return 0
    else:
        
        return (n_i-n_j)/(n_i+n_j+n_k)
            
            



def calculate_disruptiveness(references_dict,citations_dict):

    disruption_list = []
    p = Pool(10)
    func = partial(calculate_disruptiveness_,references_dict_= references_dict,citations_dict_ = citations_dict)
    print("starting")
    results = list(tqdm(p.imap(func, range(len(references_dict)) ), total = len(references_dict)))
    print("finished")
    return np.array(results)


