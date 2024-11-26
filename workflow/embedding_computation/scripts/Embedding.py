# -*- coding: utf-8 -*-
"""
Purpose: 
    Training embedding space of citation network
Input:
    argv (list): Command-line arguments

    - argv[0]: The script name
    - argv[1]: Path to the citation network file
    - argv[2]: Dimension of the embedding vectors
    - argv[3]: Size of the window
    - argv[4]: Device name where the in-vectors are stored
    - argv[5]: Device name where the out-vectors are stored and the training process will run
    - argv[6]: Name of the citation network. This parameter indicates the type of modified citation network based on the original citation network. For example, 'random' represents a randomly rewired citation network.
    - argv[7]: Value of q for the biased random walk
    - argv[8]: Epoch size
    - argv[9]: Batch size
    - argv[10]: Save directory
Ouput:
    - in_vec.npy : .npy file of in-vectors 
    - out_vec.npy : .npy file of out-vectors

Author: Munjung Kim
"""


import scipy
import sys
# import utils
import node2vecs
import logging
import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
import os
import argparse
import pandas as pd
import torch.nn as nn


if __name__ == "__main__":
    logging.basicConfig(filename = 'Embedding.log', level = logging.INFO, format='%(asctime)s %(message)s')
    

    # DATA_DIR = '/home/munjkim/SoS/Disruption/data'

    NET = sys.argv[1] # citation network file. The type is .npz
    DIM = int(sys.argv[2]) #Dimension of the embedding
    WIN = int(sys.argv[3]) #window size of the node2vec
    DEV1 = sys.argv[4] # Device to use for in-vectors
    DEV2 = sys.argv[5] # Device to use for out-vectors
    NAME = sys.argv[6] # Name of the network (Choose one between 'original', 'original/Restricted_{year}', 'random/random_i', 'destroyed/destroyed_i')
    Q =int(sys.argv[7]) # the value of q for the biased random walk
    EPOCH = int(sys.argv[8]) # the number of epochs
    BATCH = int(sys.argv[9]) # the sie of batches
    DATA_DIR = sys.argv[10]
    MODEL_FOLDER = f"{DIM}_{WIN}_q_{Q}_ep_{EPOCH}_bs_{BATCH}_embedding"
    
    SAVE_DIR = os.path.join(DATA_DIR,NAME,MODEL_FOLDER)

    logging.info('Arg Parse Success.')
    logging.info(NET)
    logging.info(SAVE_DIR)

    net = scipy.sparse.load_npz(NET)

    sampler = node2vecs.RandomWalkSampler(net, walk_length = 160)


    noise_sampler = node2vecs.utils.node_sampler.ConfigModelNodeSampler(ns_exponent=1.0)
    noise_sampler.fit(net)

    n_nodes = net.shape[0]

    dim =DIM
    logging.info("Dimension: "+str(dim))


    
    logging.info('gpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = DEV1 + ',' + DEV2
    model = node2vecs.Word2Vec(vocab_size = n_nodes, embedding_size= dim, padding_idx = n_nodes)

    loss_func = node2vecs.Node2VecTripletLoss(n_neg=1)

    

    

    dataset = node2vecs.TripletDataset(
        adjmat=net,
        window_length=WIN,
        num_walks = 25,
        noise_sampler=noise_sampler,
        padding_id=n_nodes,
        buffer_size=1e4,
        context_window_type="right", # we can limit the window to cover either side of center words. `context_window_type="double"` specifies a context window that extends both left and right of a focal node. context_window_type="left" or ="right" specifies that the window extends left or right, respectively.
        epochs=EPOCH, # number of epochs
        negative=1, # number of negative node per context
        p = 1, # (inverse) weight for the probability of backtracking walks 
        q = Q, # (inverse) weight for the probability of depth-first walks 
        walk_length = 160 # Length of walks
    )
    
    logging.info('Start training: Dim'+str(DIM) + '_Win'+str(WIN))

    node2vecs.train(
        model=model,
        dataset=dataset,
        loss_func=loss_func,
        batch_size=BATCH,
        learning_rate=1e-3,
        num_workers=10,
    )
    model.eval()

    logging.info('Finish training')

    in_vec = model.ivectors.weight.data.cpu().numpy()[:n_nodes, :] # in_vector
    out_vec = model.ovectors.weight.data.cpu().numpy()[:n_nodes, :] # out_vector
    SAVE_DIR_IN = os.path.join(SAVE_DIR ,"in.npy")
    SAVE_DIR_OUT = os.path.join(SAVE_DIR ,"out.npy")

    np.save(SAVE_DIR_IN,in_vec)
    np.save(SAVE_DIR_OUT,out_vec)
