import numpy as np
import pandas as pd
from find_motifs_new import *
import os.path
import pickle
from scipy.stats import gmean, hmean

np.random.seed(0)

# Function to compute motif numbers (M1-M16) for the given graph and closed set
def compute_motif_number(G, train_closed):
    edge_list = [tuple(sorted(e)) for e in G.edges()]
    
    # Create a dictionary of neighbors for each node
    neighbors_dict = {}
    for node in G.nodes():
        neighbors_dict[node] = set(G.neighbors(node))

    closed_set = set(train_closed)
    
    # List of motif functions to be computed
    #motifs_function = [compute_M1_M3, compute_M4_M5, compute_M6_M8, compute_M12_M16]
    motifs_function = [compute_M1_M3, compute_M4_M5, compute_M6_M8, compute_M12_M16]
    motif_number = []

    # Iterate over each motif function to compute motif counts
    for motif_fun in motifs_function:
        print(motif_fun)
        motifs = motif_fun(edge_list, closed_set, neighbors_dict)
        for i in range(motifs.shape[1]):
            motif_number.append(motifs[:, i])

    # Organize motif counts into a dictionary where each edge is mapped to its motif counts
    motif_number = list(map(list, zip(*motif_number)))
    motif_number = dict(zip(edge_list, motif_number))
    
    return motif_number

# Function to construct the feature set based on motif numbers for training and testing
def construct_x(x_trg, G, train_closed, filename):

    save_dir = 'split_dataset/' + dataset + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Compute motif numbers for the graph
    motif_number = compute_motif_number(G, train_closed)

    # Define the number of features for each edge
    feature_number = 25  # 3 for each edge, 4 for the 4 motifs
    df2 = []

    # For each target edge in the training set, retrieve corresponding motif numbers
    for r in range(len(x_trg)):
        edge1 = tuple(sorted((x_trg[r][0], x_trg[r][1])))
        edge2 = tuple(sorted((x_trg[r][0], x_trg[r][2])))
        edge3 = tuple(sorted((x_trg[r][1], x_trg[r][2])))

        df2.append(motif_number[edge1] + motif_number[edge2] + motif_number[edge3])

    # Create a DataFrame for the feature set with edges as index and motif counts as columns
    train_df = pd.DataFrame(df2, index=x_trg, columns=[str(i) for i in range(3 * feature_number)])
    train_df.index.name = 'index'

    # Compute mean, geometric mean, and harmonic mean for the motif counts
    df_mean = pd.DataFrame()
    for i in range(feature_number):
        df = train_df.iloc[:, [i, i + feature_number, i + 2 * feature_number]]
        df_mean[str(i) + '_swa'] = df.mean(axis=1)  # Simple average
        df_mean[str(i) + '_swg'] = df.apply(gmean, axis=1)  # Geometric mean
        df_mean[str(i) + '_swh'] = df.apply(hmean, axis=1)  # Harmonic mean

    # Save the computed mean values to a pickle file
    with open(save_dir + filename + '_mean.pickle', 'wb') as f:
        pickle.dump(df_mean, f)

# Main function to load data, compute features, and save them for training/testing
if __name__ == "__main__":
    dataset_list = ['coauth-DBLP']
    # 'email-Enron', 'email-Eu', 'NDC-classes', 'NDC-substances', 'contact-primary-school', 'contact-high-school'
    # 'coauth-MAG-History', 'DAWN', 'threads-ask-ubuntu', 'tags-ask-ubuntu'
    # 'coauth-DBLP' 20天'coauth-MAG-Geology','congress-bills','tags-math-sx','tags-stack-overflow','threads-math-sx','threads-stack-overflow'
    for dataset in dataset_list:
        print(dataset)
        save_dir = './processing_dataset/' + dataset
        
        # Load data for training and testing
        with open(save_dir + '/trg_open_train.pickle', 'rb') as f:
            x_train_trg = pickle.load(f)
        with open(save_dir + '/y_train.pickle', 'rb') as f:
            y_train = pickle.load(f)
        with open(save_dir + '/trg_open_test.pickle', 'rb') as f:
            x_test_trg = pickle.load(f)
        with open(save_dir + '/y_test.pickle', 'rb') as f:
            y_test = pickle.load(f)
        with open(save_dir + '/G_train.pickle', 'rb') as f:
            G_train = pickle.load(f)
        with open(save_dir + '/G_test.pickle', 'rb') as f:
            G_test = pickle.load(f)
        with open(save_dir + '/trg_closed_train.pickle', 'rb') as f:
            train_closed = pickle.load(f)
        with open(save_dir + '/trg_closed_test.pickle', 'rb') as f:
            test_closed = pickle.load(f)

        print('--------')

        # Construct features for the training and testing sets
        x_train = construct_x(x_train_trg, G_train, train_closed, filename='train')
        x_test = construct_x(x_test_trg, G_test, test_closed, filename='test')