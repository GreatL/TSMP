# structB_plus_temporal_lr_multi.py
# 方案 B：triangle 环境特征（三条边的 triangle count）、可选拼接 tri-level temporal stats + LR
# 多次运行统计 AP/AUC 的 mean±std

import os
import pickle
import argparse
import numpy as np
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler


def load_triangles_graph_labels(dataset, split):
    base = f'processing_dataset/{dataset}'
    with open(os.path.join(base, f'trg_open_{split}.pickle'), 'rb') as f:
        tris = pickle.load(f)
    with open(os.path.join(base, f'G_{split}.pickle'), 'rb') as f:
        G = pickle.load(f)
    with open(os.path.join(base, f'y_{split}.pickle'), 'rb') as f:
        y = pickle.load(f)
    return tris, G, np.array(y, dtype=np.int64)


def load_tri_temporal_stats(dataset, split):
    path = f'temporal_stats/{dataset}/tri_stats_{split}.pickle'
    if not os.path.exists(path):
        print(f'[WARN] {path} not found, temporal stats disabled for {split}.')
        return None
    with open(path, 'rb') as f:
        tri_stats = pickle.load(f)
    # seg_sums 维度
    seg_dim = None
    for ts in tri_stats:
        if ts is not None:
            seg_dim = len(ts['seg_sums'])
            break
    if seg_dim is None:
        seg_dim = 0

    feats = []
    for ts in tri_stats:
        if ts is None:
            base = np.zeros(8, dtype=np.float32)
            if seg_dim > 0:
                vec = np.zeros(8 + seg_dim, dtype=np.float32)
            else:
                vec = base
        else:
            base = np.array([
                ts['count_sum'],
                ts['count_mean'],
                ts['count_max'],
                ts['last_max'],
                ts['last_range'],
                ts['first_max'],
                ts['first_range'],
                ts['span_mean']
            ], dtype=np.float32)
            if seg_dim > 0:
                vec = np.concatenate([base, ts['seg_sums'].astype(np.float32)])
            else:
                vec = base
        feats.append(vec)
    return np.stack(feats, axis=0)  # [N, D_time]


def normalize_time_feats(T):
    mins = T.min(axis=0)
    maxs = T.max(axis=0)
    return (T - mins) / (maxs - mins + 1e-12)


def under_sample(X, y, ratio=0.33, seed=0):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res


def compute_edge_triangle_counts(G):
    """
    在 skeleton G 上，对每条边 e 计算 tri_edge[e]:
      -> e 作为子边参与多少个 triangle。
    使用邻居交集实现：对于每条边 (u,v)，CN(u,v) 的每一个 w
    与 (u,v) 组成一个 triangle。
    """
    tri_edge = {}
    # 为了避免重复多次求交集，先缓存邻居
    neighbor_dict = {n: set(G.neighbors(n)) for n in G.nodes()}
    for u, v in G.edges():
        cn = neighbor_dict[u] & neighbor_dict[v]
        tri_edge[(u, v)] = len(cn)
    return tri_edge


def build_structB_features(tris, G):
    """
    对每个 τ=(i,j,k) 构造结构 B 特征:
      三条边 (i,j),(j,k),(i,k) 的 triangle count 统计：
        tri_edge_sum, tri_edge_mean, tri_edge_max, tri_edge_min
    """
    tri_edge = compute_edge_triangle_counts(G)

    feats = []
    for tri in tris:
        i, j, k = tri
        e1 = tuple(sorted((i, j)))
        e2 = tuple(sorted((j, k)))
        e3 = tuple(sorted((i, k)))
        vals = []
        for e in (e1, e2, e3):
            u, v = e
            v_cnt = tri_edge.get((u, v), tri_edge.get((v, u), 0))
            vals.append(v_cnt)
        vals = np.array(vals, dtype=np.float32)
        s = vals.sum()
        m = vals.mean()
        mx = vals.max()
        mn = vals.min()
        feat = np.array([s, m, mx, mn], dtype=np.float32)
        feats.append(feat)
    return np.stack(feats, axis=0)  # [N,4]


def run_once(dataset, use_temporal, seed=0):
    np.random.seed(seed)

    # 结构窗：train/test 各自从 processing_dataset 读 G_train/G_test
    tris_tr, G_tr, y_tr = load_triangles_graph_labels(dataset, 'train')
    tris_te, G_te, y_te = load_triangles_graph_labels(dataset, 'test')

    # 结构 B 特征
    X_tr_struct = build_structB_features(tris_tr, G_tr)
    X_te_struct = build_structB_features(tris_te, G_te)

    if use_temporal:
        T_tr = load_tri_temporal_stats(dataset, 'train')
        T_te = load_tri_temporal_stats(dataset, 'test')
        if T_tr is None or T_te is None:
            use_temporal = False

    if use_temporal:
        T_tr_norm = normalize_time_feats(T_tr)
        T_te_norm = normalize_time_feats(T_te)
        X_tr = np.concatenate([X_tr_struct, T_tr_norm], axis=1)
        X_te = np.concatenate([X_te_struct, T_te_norm], axis=1)
    else:
        X_tr, X_te = X_tr_struct, X_te_struct

    # 欠采样 + LR
    X_tr_bal, y_tr_bal = under_sample(X_tr, y_tr, ratio=0.33, seed=seed)
    clf = LogisticRegression(solver='liblinear', penalty='l2', max_iter=1000)
    clf.fit(X_tr_bal, y_tr_bal)

    y_score = clf.predict_proba(X_te)[:, 1]
    if y_te.sum() == 0:
        ap, auc = 0.0, np.nan
    else:
        ap  = average_precision_score(y_te, y_score)
        try:
            auc = roc_auc_score(y_te, y_score)
        except ValueError:
            auc = np.nan
    return ap, auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='email-Enron')
    parser.add_argument('--use_temporal', action='store_true')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    ds   = args.dataset
    useT = args.use_temporal

    print(f'===== StructB (triangle env) + {"Temporal" if useT else "None"} LR multi-run Dataset: {ds} =====')

    aps, aucs = [], []
    for k in range(args.runs):
        seed_k = args.seed + k
        ap, auc = run_once(ds, useT, seed=seed_k)
        aps.append(ap); aucs.append(auc)
        print(f'  Run {k}: AP={ap:.4f}, AUC={auc:.4f}')

    aps  = np.array(aps)
    aucs = np.array(aucs)
    ap_mean  = float(np.nanmean(aps))
    ap_std   = float(np.nanstd(aps, ddof=1)) if args.runs > 1 else 0.0
    auc_mean = float(np.nanmean(aucs))
    auc_std  = float(np.nanstd(aucs, ddof=1)) if args.runs > 1 else 0.0

    print(f'AP mean={ap_mean:.4f}, std={ap_std:.4f}')
    print(f'AUC mean={auc_mean:.4f}, std={auc_std:.4f}')
