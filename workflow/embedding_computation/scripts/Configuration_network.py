"""
Purpose: 
    Making a random citation network while keeping the agegap
Input:
    - -n citation network with sparse network datatype
    
Ouput:
    - random network with sparse network datatype

Author: Munjung Kim
"""

import gc
import scipy
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse
import networkx as nx
import configparser
import sys
import pandas as pd
from tqdm import tqdm
from random import shuffle
from utils.data_loading import load_papers 
from multiprocessing import Pool

def randomized_edgelist(edge_list_to_delete):
    if len(edge_list_to_delete)<2:
        return edge_list_to_delete
    row_before, col_before = list(zip(*edge_list_to_delete))
    row_before = list(row_before)
    col_before = list(col_before)
        
    shuffle(col_before)
        
    new_edgelist = list(zip(*[row_before,col_before]))
        
    return new_edgelist


if __name__ == "__main__":
    
    logging.basicConfig(filename = 'Configuration_network_agegap.log',level=logging.INFO, format='%(asctime)s %(message)s')
    
    logging.info(sys.argv[2])

    directory = os.path.dirname(sys.argv[3])
    if not os.path.exists(directory):
        os.makedirs(directory)


    
    net = scipy.sparse.load_npz(sys.argv[1])
    net = net.tocoo()
    edge_list = list(zip(net.row,net.col))

    node_df = pd.read_csv(sys.argv[2])
    logging.info("load nodedf")

    years_comb_edge_list = {(j,i) : [] for i in range( int(min(node_df["year"])),int(max(node_df["year"])+1 ) )for j in range(i,int(max(node_df["year"])+1 ))  }
    
    years_comb_edge_list_reverse = {
        (i, j): []
        for i in range(int(min(node_df["year"])), int(max(node_df["year"]) + 1))
        for j in range(i, int(max(node_df["year"]) + 1))
        if i != j  # Exclude cases where i == j
                    }
            
    
    logging.info("start  making year combination dictionary")
    nan_combination = []
    for u,v in tqdm(edge_list):
        year_comb = (node_df["year"][u],node_df["year"][v])
        
        if pd.isna(node_df["year"][u]) or pd.isna(node_df["year"][v]):
            nan_combination.append((u,v))
            continue
        
    
        if year_comb not in years_comb_edge_list:
            years_comb_edge_list_reverse[year_comb].append((u,v))
        else:
            years_comb_edge_list[year_comb].append((u,v))
    total_years_combination =  {**years_comb_edge_list, **years_comb_edge_list_reverse}
       
    logging.info("start  randomizing the citation network")
    del years_comb_edge_list
    del years_comb_edge_list_reverse
    del node_df



    p = Pool(20)
    
    print("starting")
    results = list(tqdm(p.imap_unordered(randomized_edgelist, total_years_combination.values()), total = len(total_years_combination)))
    del total_years_combination
    new_edge_list = [i for j in results for i in j] + nan_combination

    del results
    
    new_rows, new_cols = list(zip(*new_edge_list))
    del new_edge_list
    ones = np.ones(len(new_rows))
    new_net = scipy.sparse.csr_matrix((ones, (new_rows, new_cols)),shape = net.shape)
    

    logging.info('Finish making a configuration model.')
    logging.info('Saving a configuration model.')

    
    
        
        
    scipy.sparse.save_npz(sys.argv[3],new_net)
    
    
    
    
    
    





