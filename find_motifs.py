import networkx as nx
import numpy as np
import itertools

def compute_M1_M3(edge_list, train_closed, neighbors_dict):
    result_r = np.zeros((len(edge_list), 3), dtype=int)
    for e,(n1, n2) in enumerate(edge_list):
        nei1 = neighbors_dict[n1]
        nei2 = neighbors_dict[n2]
        result_r[e,0] = len(nei1 ^ nei2) #M1

        motif_3_all = {
            (elem, n1, n2) if elem < n1 else
            (n1, elem, n2) if elem <= n2 else
            (n1, n2, elem)
            for elem in nei1 & nei2
        }
        result_r[e, 2] = len(motif_3_all & train_closed) #M3
        result_r[e, 1] = len(motif_3_all)- result_r[e, 2] #M2

    return result_r

def compute_M4_M5(x_edges, train_closed,neighbors_dict):
    result_r = np.zeros((len(x_edges), 3), dtype=int)
    for e,(n1, n2) in enumerate(x_edges):
        nei1 = neighbors_dict[n1]
        nei2 = neighbors_dict[n2]
        nei_x=nei1-nei2
        nei_y=nei2-nei1
        
        sorted_combinations = set()
        for x in nei_x:
            for y in nei_y:
                if x < y:
                    sorted_combinations.add((x, y))
                else:
                    sorted_combinations.add((y, x))

        has_edge = len( sorted_combinations & set(x_edges) )
        no_edge = len(sorted_combinations) - has_edge

        result_r[e,0] = no_edge #M4-1

        result_r[e,2] = has_edge #M5
        
        for n in (nei_x | nei_y):
            second_nei=neighbors_dict[n]-(nei1 | nei2 )
            result_r[e,1] += len(second_nei)  #M4-2

    return result_r

def compute_M6_M8(x_edges, train_closed,neighbors_dict):
    
    result_r = np.zeros((len(x_edges), 11), dtype=int)
    x_edges_set=set(x_edges)
    for e,(x, y) in enumerate(x_edges):
        nei_x = neighbors_dict[x]
        nei_y = neighbors_dict[y]
        nei_x_only=set(sorted(nei_x - nei_y))
        nei_y_only=set(sorted(nei_y - nei_x ))
        nei_x_y=nei_x | nei_y

        combin_x = set(itertools.combinations(nei_x_only, 2))
        combin_y = set(itertools.combinations(nei_y_only, 2))

        has_edge_1= combin_x & x_edges_set
        has_edge_2 = combin_y & x_edges_set

        result_r[e, 0] = len(combin_x)-len(has_edge_1)+len(combin_y)-len(has_edge_2) #M6

        has_tri_1 = {
            (x, n1, n2) if x < n1 else
            (n1, x, n2) if x <= n2 else
            (n1, n2, x)
            for n1,n2 in has_edge_1
        } & train_closed
        has_tri_2 = {
            (y, n1, n2) if y < n1 else
            (n1, y, n2) if y <= n2 else
            (n1, n2, y)
            for n1,n2 in has_edge_2
        } & train_closed

        result_r[e, 1]=len(has_edge_1)-len(has_tri_1) +len(has_edge_2)-len(has_tri_2) #M7-1
        result_r[e, 4]=len(has_tri_1) + len(has_tri_2) #M8-1
        
        first=len(nei_x_only) + len(nei_y_only)
        for n in (nei_x & nei_y):
            second=len(neighbors_dict[n] - nei_x_y)

            z1 = neighbors_dict[n] & nei_x_only
            z2 = neighbors_dict[n] & nei_y_only
            has_tri_z=(set([tuple(sorted((x, n, z))) for z in z1]) | set([tuple(sorted((y, n, z))) for z in z2])) & train_closed
        

            if n <= x:
                temp_tri=(n, x, y)
            elif n <= y:
                temp_tri=(x, n, y)
            else:
                temp_tri=(x, y, n)
            if temp_tri in train_closed:
                
                result_r[e, 5] += first #M8-2
                result_r[e, 6] += second #M8-3
                result_r[e, 10] += len(has_tri_z) #M11-1
                result_r[e, 9] += len(z1)+len(z2)-len(has_tri_z) #M10-2
            else: 
                result_r[e, 2] += first  #M7-2
                result_r[e, 3] += second #M7-3
                result_r[e, 8] += len(has_tri_z) #M10-1
                result_r[e, 7] += len(z1)+len(z2)-len(has_tri_z) #M9-1

    return result_r

def compute_M12_M16(x_edges, train_closed,neighbors_dict):
    x_edges_set=set(x_edges)
    result_r = np.zeros((len(x_edges), 8), dtype=int)
    for e,(x,y) in enumerate(x_edges):
        nei1,nei2 = neighbors_dict[x],neighbors_dict[y]
        for z,w in itertools.combinations(nei1 & nei2, 2):
            three_node_combinations = len(set(itertools.combinations(sorted({x, y, z, w}), 3)) & train_closed)
            if three_node_combinations <=2:
                if (z,w) in x_edges_set:
                    result_r[e, three_node_combinations+3 ] += 1  #M12-M14
                else:
                    result_r[e, three_node_combinations] += 1 #M9-2,M10-3,M11-2
            else:
                result_r[e, three_node_combinations+3 ] += 1  #M15-M16

    return result_r