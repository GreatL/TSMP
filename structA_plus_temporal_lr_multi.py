# structA_plus_temporal_lr_multi.py
# 结构特征方案 A（边权 + 节点度 + common neighbors） + tri-level temporal stats + LR
# 多次运行输出 AP/AUC mean±std

import os
import pickle
import argparse
import numpy as np
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler


# ------------------------
# 读取 open triangles / skeleton / labels
# ------------------------

def load_triangles_graph_labels(dataset, split):
    base = f'processing_dataset/{dataset}'
    with open(os.path.join(base, f'trg_open_{split}.pickle'), 'rb') as f:
        tris = pickle.load(f)  # list of (i,j,k)
    with open(os.path.join(base, f'G_{split}.pickle'), 'rb') as f:
        G = pickle.load(f)     # networkx.Graph (当前是无权骨架)
    with open(os.path.join(base, f'y_{split}.pickle'), 'rb') as f:
        y = pickle.load(f)
    return tris, G, np.array(y, dtype=np.int64)


# ------------------------
# 读取 tri-level temporal stats
# ------------------------

def load_tri_temporal_stats(dataset, split):
    path = f'temporal_stats/{dataset}/tri_stats_{split}.pickle'
    if not os.path.exists(path):
        print(f'[WARN] {path} not found, temporal features disabled for {split}.')
        return None

    with open(path, 'rb') as f:
        tri_stats = pickle.load(f)  # list，与 open_tris 对齐

    # 确定 seg_sums 维度
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

    tri_time = np.stack(feats, axis=0)  # [N, D_time]
    return tri_time


def normalize_time_feats(T):
    mins = T.min(axis=0)
    maxs = T.max(axis=0)
    return (T - mins) / (maxs - mins + 1e-12)


# ------------------------
# 结构特征方案 A: 边权 + 节点度 + common neighbors
# ------------------------

def compute_edge_weights_and_cn(G):
    """
    G 当前是无权 skeleton（由 processing_dataset 生成），
    不包含边权。为方案 A，我们需要基于原始 simplices 重新统计：
      - edge weight: 共现次数
      - common neighbors: 可以在当前 G 上按度量法计算
    但为了不改原数据结构，这里只针对 G 计算 CN 和度；
    edge weight 这一步简化为使用 G 中的 'multiplicity' 近似：
      - 如果你有带权 skeleton，可以在这里替换为真实权重。
    目前 G 中没有 weight 属性，我们设每条边权重=1，
    w_sum/w_mean/w_max 等将退化为“边数量的简单函数”，
    仍然可用于验证时间特征的重要性。
    """
    # 节点度
    degrees = dict(G.degree())
    # common neighbors: for any edge (u,v), cn_uv = |N(u)∩N(v)|
    cn_dict = {}
    neighbor_dict = {n: set(G.neighbors(n)) for n in G.nodes()}
    for u, v in G.edges():
        cn = neighbor_dict[u] & neighbor_dict[v]
        cn_dict[(u, v)] = len(cn)
    return degrees, cn_dict


def build_structA_features(tris, G):
    """
    对每个 triangle τ=(i,j,k) 构造结构特征方案 A:
      - 边权: 这里先用 weight=1 简化（如果你有真实权重，可在此替换）
      - 节点度: d_i, d_j, d_k
      - common neighbors: 对三条边的 cn_uv 统计
    返回 struct_feats: ndarray [N, D_struct]
    """
    degrees, cn_dict = compute_edge_weights_and_cn(G)

    feats = []
    for tri in tris:
        i, j, k = tri
        # 三条边
        e1 = tuple(sorted((i, j)))
        e2 = tuple(sorted((j, k)))
        e3 = tuple(sorted((i, k)))
        edges = [e1, e2, e3]

        # 边权 (当前简化为 weight=1)
        ws = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        w_sum  = ws.sum()
        w_mean = ws.mean()
        w_max  = ws.max()
        w_min  = ws.min()

        # 节点度
        ds = np.array([
            degrees.get(i, 0),
            degrees.get(j, 0),
            degrees.get(k, 0)
        ], dtype=np.float32)
        deg_sum  = ds.sum()
        deg_mean = ds.mean()
        deg_max  = ds.max()
        deg_min  = ds.min()

        # common neighbors per edge
        cns = []
        for e in edges:
            u, v = e
            cn_val = cn_dict.get((u, v), cn_dict.get((v, u), 0))
            cns.append(cn_val)
        cns = np.array(cns, dtype=np.float32)
        cn_sum  = cns.sum()
        cn_mean = cns.mean()
        cn_max  = cns.max()

        feat_tau = np.array([
            w_sum, w_mean, w_max, w_min,
            deg_sum, deg_mean, deg_max, deg_min,
            cn_sum, cn_mean, cn_max
        ], dtype=np.float32)
        feats.append(feat_tau)

    struct_feats = np.stack(feats, axis=0)  # [N, 11]
    return struct_feats


# ------------------------
# 训练 / 评估
# ------------------------

def under_sample(X, y, ratio=0.33, seed=0):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res


def run_once(dataset, use_temporal, seed=0):
    np.random.seed(seed)

    # 1) 载入训练/测试三元组、图和标签
    tris_tr, G_tr, y_tr = load_triangles_graph_labels(dataset, 'train')
    tris_te, G_te, y_te = load_triangles_graph_labels(dataset, 'test')

    # 2) 结构特征方案 A
    X_tr_struct = build_structA_features(tris_tr, G_tr)
    X_te_struct = build_structA_features(tris_te, G_te)

    # 3) 时间特征
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
        X_tr = X_tr_struct
        X_te = X_te_struct

    # 4) 欠采样 + LR
    X_tr_bal, y_tr_bal = under_sample(X_tr, y_tr, ratio=0.33, seed=seed)

    clf = LogisticRegression(
        solver='liblinear',
        penalty='l2',
        max_iter=1000
    )
    clf.fit(X_tr_bal, y_tr_bal)

    y_score = clf.predict_proba(X_te)[:, 1]

    if y_te.sum() == 0:
        ap = 0.0
        auc = np.nan
    else:
        ap  = average_precision_score(y_te, y_score)
        try:
            auc = roc_auc_score(y_te, y_score)
        except ValueError:
            auc = np.nan
    return ap, auc


# ------------------------
# 主函数：多次运行统计 mean±std
# ------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='email-Enron')
    parser.add_argument('--use_temporal', action='store_true',
                        help='是否拼接 triangle-level temporal stats')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    ds   = args.dataset
    useT = args.use_temporal

    print(f'===== StructA (edge+deg+CN) + {"Temporal" if useT else "None"} LR multi-run Dataset: {ds} =====')

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
