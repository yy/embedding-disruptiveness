import scipy
import node2vecs
from utils.cal_disruption import calculate_disruptiveness_
import torch
import numpy as np
import pickle
import os
import argparse
import configparser
import sys
import logging
from tqdm import tqdm
import pickle 
from multiprocessing import Pool
from functools import partial 
from pathlib import Path


if __name__ == "__main__":
    
    NETWORK = sys.argv[1]
  


    net = scipy.sparse.load_npz(NETWORK)
    net = net.tocoo()
    edge_list = list(zip(net.row,net.col))

    reference_dict = {i:set() for i in tqdm(range(net.shape[0]))}
    citation_dict = {i:set() for i in tqdm(range(net.shape[0]))}
    

    
    for i in tqdm(edge_list):
        reference_dict[i[0]].add(i[1])
        citation_dict[i[1]].add(i[0])

 
    path = Path(NETWORK)
    parent = path.parent.absolute()
    reference_dict_path = os.path.join(parent, "reference_dict.pkl")
    citation_dict_path = os.path.join(parent, "citation_dict.pkl")

    with open(reference_dict_path, 'wb') as handle:
        pickle.dump(reference_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(citation_dict_path, 'wb') as handle:
        pickle.dump(citation_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

