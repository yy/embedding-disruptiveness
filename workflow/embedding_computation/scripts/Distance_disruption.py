"""
Purpose: 
    Calculate distance and disruption index
Input:
    argv (list): Command-line arguments

    - argv[0] : The script name
    - argv[1] : Measure index type: you can choose between `disruption` (previous measure) and `distance`.
    - argv[2] : path to the in-vector numpy file
    - argv[3] : path to the out-vector numpy file
    - argv[4] : path to the network file
    - argv[5] : device name to calculate the distance

Ouput:
    - Disruption_{embedding_file_namees}.npy : .npy file that contains disruption index of each paper
    - Distance.npy : .npy file that contains distance index of each paper

Author: Munjung Kim
"""  
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



if __name__ == "__main__":
    
    MEASURE = sys.argv[1]

    
    
    if 'disruption' in MEASURE:
        
        REF_DICT = sys.argv[2] #pkl file of reference list
        CIT_DICT = sys.argv[3]
        NETWORK = sys.argv[4]
        DEVICE = sys.argv[5]

        RESTRICT = sys.argv[6]

        logging.basicConfig(filename = 'Disruption.log',level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info('Start loading reference dictionary')
        
        
        with open(REF_DICT, 'rb') as f:
            net_coo_dict_reference = pickle.load(f)
        with open(CIT_DICT, 'rb') as f:
            net_coo_dict_citation = pickle.load(f)
                
        logging.info('Start calculating the disruptiveness function')
        number_of_edges = len(net_coo_dict_citation)

        
        
        disruption_result = []
        

        for i in tqdm(range(number_of_edges )):
            ascdnt = net_coo_dict_reference[i]# reference/ asc
            dscdnt = net_coo_dict_citation[i]# citation  
            
            n_j = 0
            n_i=0
            n_k = set()
                
            for d in dscdnt:
                if net_coo_dict_reference[d] & ascdnt:
                    n_j+=1
                else:
                    n_i+=1  

            for a in ascdnt:
                n_k = n_k.union(net_coo_dict_citation[a])  

            n_k = n_k - dscdnt

            n_k = len(n_k)

            
            if MEASURE == 'disruption_nok':
                if n_i+n_j ==0:
                    disruption_result.append(0)
                else:
                    disruption_result.append( (n_i-n_j)/(n_i+n_j))
                    
            else:
                if n_i+n_j+n_k ==0:
                    disruption_result.append(0)
                else:
                    disruption_result.append( (n_i-n_j)/(n_i+n_j+n_k))

            del n_i
            del n_k
            del n_j
            del dscdnt
            del ascdnt

        
        del net_coo_dict_citation
        del net_coo_dict_reference
        


        

        # di = calculate_disruptiveness(net_coo_dict_reference,net_coo_dict_citation)
    
        
        NET_FOLDER = os.path.abspath(os.path.join(NETWORK, os.pardir))
        if MEASURE == 'disruption_5':
            SAVE_DIR = os.path.join(NET_FOLDER,'disruption_5.npy')
        elif MEASURE == 'disruption_nok':
            SAVE_DIR = os.path.join(NET_FOLDER,'disruption_nok.npy')
        elif MEASURE == 'disruption':
            SAVE_DIR = os.path.join(NET_FOLDER,'disruption.npy')
        np.save(SAVE_DIR,np.array(disruption_result))

    elif MEASURE =='distance':
        EMBEDDING_IN = sys.argv[2]
        EMBEDDING_OUT = sys.argv[3]
        NETWORK = sys.argv[4]
        DEVICE = sys.argv[5]

        RESTRICT = sys.argv[6]
        
        logging.basicConfig(filename = 'Distance.log',level=logging.INFO, format='%(asctime)s %(message)s')

    
        # net = scipy.sparse.load_npz(NETWORK)
    
        in_vec = np.load(EMBEDDING_IN)
        out_vec = np.load(EMBEDDING_OUT)
        
        EMBEDDING_FOLDER = os.path.abspath(os.path.join(EMBEDDING_IN, os.pardir))
        
        in_vec_torch = torch.from_numpy(in_vec).to(DEVICE)
        out_vec_torch = torch.from_numpy(out_vec).to(DEVICE)

        n = len(out_vec_torch)

        distance= []
        
        batch_size = int(n/2000) + 1
        
        logging.info('Starting calculating the distances')

        for i in tqdm(range(2000)):
            X = in_vec_torch[i*batch_size: (i+1)*batch_size]
            Y = out_vec_torch[i*batch_size: (i+1)*batch_size]
            numerator = torch.diag(torch.matmul(X,torch.transpose(Y,0,1)))
            norms_X = torch.sqrt((X * X).sum(axis=1))
            norms_Y = torch.sqrt((Y * Y).sum(axis=1))

            denominator = norms_X*norms_Y


            cs = 1 - torch.divide(numerator, denominator)
            distance.append(cs.tolist())
        
        distance_lst =  np.array([dis for  sublist in distance for dis in sublist])
        
        
        logging.info('Saving the files.')
        
        SAVE_DIR = os.path.join(EMBEDDING_FOLDER,'distance.npy')
        np.save(SAVE_DIR, distance_lst)


    elif MEASURE == 'multistep' :

        logging.basicConfig(filename = 'Multistep_Disruption.log',level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info('Start loading reference dictionary')

        NET_FOLDER = os.path.abspath(os.path.join(NETWORK, os.pardir))

        SAVE_DIR = os.path.join(NET_FOLDER,'disruption_twostep.npy')

        logging.info(SAVE_DIR)


        with open('/data/sg/munjkim/wos/original_1960_2019/references_dict.pkl', 'rb') as f:
                net_coo_dict_reference = pickle.load(f)
                
        logging.info('Start loading citation dictionary')

        with open('/data/sg/munjkim/wos/original_1960_2019/citations_dict.pkl', 'rb') as f:
                net_coo_dict_citation = pickle.load(f)
        
        logging.info('Start calculating the disruptiveness function')
        
        
        number_of_edges = len(net_coo_dict_citation)

        
        
        disruption_result = []
        
        
#         for i in tqdm(range(number_of_edges )):
            
            
#             ascdnt = list(net_coo_dict_reference[i])# reference/ asc
#             for asd in net_coo_dict_reference[i]:
#                 # for asdasd in net_coo_dict_reference[asd]:
#                 ascdnt.extend(list(net_coo_dict_reference[asd]))
#             ascdnt = set(ascdnt)
#             dscdnt =  list(net_coo_dict_citation[i])# citation  
#             for dsd in net_coo_dict_citation[i]:
#                 # for dsddsd in net_coo_dict_citation[dsd]:
#                 dscdnt.extend(list(net_coo_dict_citation[dsd]))
#             dscdnt = set(dscdnt)
            
#             n_j = 0
#             n_i=0
#             n_k = set()
                
#             for d in dscdnt:
#                 if net_coo_dict_reference[d] & ascdnt:
#                     n_j+=1
#                 else:
#                     n_i+=1  

#             for a in ascdnt:
#                 n_k = n_k.union(net_coo_dict_citation[a])  

#             n_k = n_k - dscdnt

#             n_k = len(n_k)

#             if n_i+n_j+n_k ==0:
#                 disruption_result.append(0)
#             else:
                
#                 disruption_result.append( (n_i-n_j)/(n_i+n_j+n_k))

#             del n_i
#             del n_k
#             del n_j
#             del dscdnt
#             del ascdnt
            
#         if i %60000 ==0:
#             np.save(SAVE_DIR,np.array(disruption_result))
            

        
#         del net_coo_dict_citation
#         del net_coo_dict_reference
        


        

        # di = calculate_disruptiveness(net_coo_dict_reference,net_coo_dict_citation)
        
        
        
        def calculate_disruption(i, net_coo_dict_reference, net_coo_dict_citation):
            logging.info('Start loading citation dictionary')


            logging.info('Start calculating the disruptiveness function')

            ascdnt = list(net_coo_dict_reference[i])
            for asd in net_coo_dict_reference[i]:
                ascdnt.extend(list(net_coo_dict_reference[asd]))
            ascdnt = set(ascdnt)

            dscdnt = list(net_coo_dict_citation[i])
            for dsd in net_coo_dict_citation[i]:
                dscdnt.extend(list(net_coo_dict_citation[dsd]))
            dscdnt = set(dscdnt)

            n_j = 0
            n_i = 0
            n_k = set()

            for d in dscdnt:
                if net_coo_dict_reference[d] & ascdnt:
                    n_j += 1
                else:
                    n_i += 1

            for a in ascdnt:
                n_k = n_k.union(net_coo_dict_citation[a])

            n_k = n_k - dscdnt
            n_k = len(n_k)

            if n_i + n_j + n_k == 0:
                result = 0
            else:
                result = (n_i - n_j) / (n_i + n_j + n_k)

            del n_i, n_k, n_j, dscdnt, ascdnt, net_coo_dict_citation

            return result
        
        
        
        with Pool(processes=20) as pool:
            partial_calculate_disruption = partial(calculate_disruption, net_coo_dict_reference=net_coo_dict_reference, net_coo_dict_citation=net_coo_dict_citation)
            disruption_result = list(tqdm(pool.imap(partial_calculate_disruption, range(number_of_edges)), total=number_of_edges))

    
        
        
        
        np.save(SAVE_DIR,np.array(disruption_result))
    
    
   
    
        
    
        
        
    
        
        
    
    
    