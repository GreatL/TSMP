import pandas as pd
import os
import pickle
import networkx as nx
import numpy as np
import itertools

np.random.seed(0)

def split_file_to_list(fl_path):
    """Reads the data from a file and returns a list."""
    a = []
    with open(fl_path, 'r') as f:
        for ff in f.readlines():
            a.append(int(ff))
    return a

def find_tri_common_neighbor(G):
    """Finds and returns all triangles (2-clique) in the graph G."""
    neighbor_dict = {n: set(G.neighbors(n)) for n in G.nodes()}
    all_triangle = set()
    for node1, node2 in G.edges():
        common_neighbors = neighbor_dict[node1].intersection(neighbor_dict[node2])
        for common_node in common_neighbors:
            triangle = tuple(sorted([node1, common_node, node2]))
            all_triangle.add(triangle)
    return all_triangle

def split_train_probe(nv_lis, sp_lis, tm_lis, start_ratio, end_ratio, test_ratio, k):
    """Splits the simplicial data into open/closed triangles, corresponding skeleton networks, and dataset labels based on time. 
    start_ratio represents the starting timestamp for dataset splitting, 
    end_ratio represents the ending timestamp for dataset splitting, 
    test_ratio represents the ending timestamp for the test set, 
    k represents the order of the prediction, and 2 represents three nodes."""
    simplices_0_60, simplices_60_80 = split_data(nv_lis, sp_lis, tm_lis, start_ratio, end_ratio, test_ratio)
    
    train_closed_trg, train_edge = set(), set()
    for simp in simplices_0_60:
        if len(simp) == 2:
            train_edge.add(tuple(sorted(simp)))
        elif len(simp) > 2:
            for trig in itertools.combinations(simp, k):
                train_closed_trg.add(tuple(sorted(trig)))
            for edg in itertools.combinations(simp, 2):
                train_edge.add(tuple(sorted(edg)))
    
    G_train = nx.Graph()
    G_train.add_edges_from(train_edge)
    train_node = set(G_train.nodes())
    all_triangle_train = find_tri_common_neighbor(G_train)
    train_open_trg = list(all_triangle_train ^ train_closed_trg)

    closed_trg_60_80 = set()
    for simp in simplices_60_80:
        if len(simp) > 2:
            for trig in itertools.combinations(simp, k):
                if tuple(sorted(trig)) not in train_closed_trg:
                    if len(set(trig) & train_node) == k:
                        closed_trg_60_80.add(tuple(sorted(trig)))

    y_train = [1 if i in closed_trg_60_80 else 0 for i in train_open_trg]

    return G_train, train_open_trg, list(train_closed_trg), y_train

def split_data(nv_lis, sp_lis, tm_lis, start_ratio, end_ratio, test_ratio):
    """Splits data into training and testing sets based on the timestamps."""
    old_zip, new_zip = set(), set()
    old_simplices, new_simplices = [], []
    
    # Split data based on time
    start_time = int(np.round(np.percentile(tm_lis, start_ratio)))
    end_time = int(np.round(np.percentile(tm_lis, min(end_ratio, 100))))

    curr_ind = 0
    for (nv, time) in zip(nv_lis, tm_lis):
        end_ind = curr_ind + nv
        if (time >= start_time) & (time <= end_time):
            simp_zip = tuple([time, tuple(sorted(sp_lis[curr_ind:end_ind]))])
            old_zip.add(simp_zip)
        curr_ind += nv
    if old_zip:
        _, old_simplices = zip(*old_zip)

    # Split testing set based on time
    start_time = end_time + 1
    end_time = int(np.round(np.percentile(tm_lis, min(end_ratio + test_ratio, 100))))

    curr_ind = 0
    for (nv, time) in zip(nv_lis, tm_lis):
        end_ind = curr_ind + nv
        if (time >= start_time) & (time <= end_time):
            simp_zip = tuple([time, tuple(sorted(sp_lis[curr_ind:end_ind]))])
            new_zip.add(simp_zip)

        curr_ind += nv
    if new_zip:
        _, new_simplices = zip(*new_zip)

    return old_simplices, new_simplices

if __name__ == "__main__":
    dataset_list = ['coauth-DBLP','coauth-MAG-Geology','congress-bills','tags-math-sx','tags-stack-overflow','threads-math-sx','threads-stack-overflow']
    # 'email-Enron', 'email-Eu', 'NDC-classes', 'NDC-substances', 'contact-primary-school', 'contact-high-school', 'coauth-MAG-History', 'DAWN', 'threads-ask-ubuntu', 'tags-ask-ubuntu'
    
    for dataset in dataset_list:
        print(dataset)
        fl_nm_nv = './datasets/' + dataset + '/' + dataset + '-nverts.txt'
        fl_nm_sp = './datasets/' + dataset + '/' + dataset + '-simplices.txt'
        fl_nm_tm = './datasets/' + dataset + '/' + dataset + '-times.txt'

        nv_lis = split_file_to_list(fl_nm_nv)  # the number of vertices within each simplex
        sp_lis = split_file_to_list(fl_nm_sp)  # the nodes comprising the simplices
        tm_lis = split_file_to_list(fl_nm_tm)  # the timestamps for each simplex
    
        G_train, x_train_trg, train_closed, y_train = split_train_probe(nv_lis, sp_lis, tm_lis, 0, 60, 20, 3)
        G_test, x_test_trg, test_closed, y_test = split_train_probe(nv_lis, sp_lis, tm_lis, 0, 80, 20, 3)
        
        # Print statistics of training and testing data
        print('Training set sample count:', len(x_train_trg))
        print('Testing set sample count:', len(x_test_trg))
        print('Training set closed triangles count:', sum(y_train))
        print('Training set open triangles count:', len(y_train) - sum(y_train))
        print('Testing set closed triangles count:', sum(y_test))
        print('Testing set open triangles count:', len(y_test) - sum(y_test))
        print('--------')

        # Save processed dataset
        save_dir = './processing_dataset/' + dataset
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        with open(save_dir + '/trg_open_train.pickle', 'wb') as f:
            pickle.dump(x_train_trg, f)
        with open(save_dir + '/y_train.pickle', 'wb') as f:
            pickle.dump(y_train, f)
        with open(save_dir + '/trg_open_test.pickle', 'wb') as f:
            pickle.dump(x_test_trg, f)
        with open(save_dir + '/y_test.pickle', 'wb') as f:
            pickle.dump(y_test, f)
        with open(save_dir + '/G_train.pickle', 'wb') as f:
            pickle.dump(G_train, f)
        with open(save_dir + '/G_test.pickle', 'wb') as f:
            pickle.dump(G_test, f)
        with open(save_dir + '/trg_closed_train.pickle', 'wb') as f:
            pickle.dump(train_closed, f)
        with open(save_dir + '/trg_closed_test.pickle', 'wb') as f:
            pickle.dump(test_closed, f)