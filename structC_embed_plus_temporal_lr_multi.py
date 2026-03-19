# structC_embed_plus_temporal_lr_multi.py
# 方案 C：PyG Node2Vec 节点嵌入结构特征 + tri-level temporal stats + LR，多次运行

import os
import pickle
import argparse
import random
import numpy as np
import networkx as nx
import torch

from torch_geometric.nn import Node2Vec
from torch_geometric.utils import from_networkx

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
    return np.stack(feats, axis=0)


def normalize_time_feats(T):
    mins = T.min(axis=0)
    maxs = T.max(axis=0)
    return (T - mins) / (maxs - mins + 1e-12)


def train_pyg_node2vec(G_nx, dim=64, walk_length=40, context_size=20,
                       walks_per_node=10, p=1.0, q=1.0,
                       batch_size=128, lr=0.01, epochs=10, device='cpu', seed=0):
    """
    在 skeleton G_nx 上用 PyG Node2Vec 训练节点 embedding。
    返回 emb_dict[node_id] -> np.array(dim,)
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    H = G_nx.copy()
    mapping = {node: idx for idx, node in enumerate(H.nodes())}
    inv_mapping = {idx: node for node, idx in mapping.items()}
    H = nx.relabel_nodes(H, mapping)
    data = from_networkx(H)

    model = Node2Vec(
        edge_index=data.edge_index,
        embedding_dim=dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=p, q=q,
        num_negative_samples=1,
        sparse=True
    ).to(device)

    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            pos_rw = pos_rw.to(device)
            neg_rw = neg_rw.to(device)
            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        avg_loss = total_loss / len(loader)
        if epoch % 2 == 0 or epoch == 1:
            print(f'Node2Vec epoch {epoch:03d} | loss={avg_loss:.4f}')

    model.eval()
    with torch.no_grad():
        z = model.embedding.weight.cpu().numpy()  # [N, dim]

    emb_dict = {}
    for idx, node in inv_mapping.items():
        emb_dict[node] = z[idx]
    return emb_dict


def build_embed_struct_features(tris, emb_dict):
    """
    对每个 τ=(i,j,k)，构造基于 embedding 的结构特征:
      - h_i, h_j, h_k
      - h_mean = (h_i+h_j+h_k)/3
      - h_max = elementwise max(h_i,h_j,h_k)
      -> concat 得到结构特征
    """
    feats = []
    d = len(next(iter(emb_dict.values())))
    for tri in tris:
        i, j, k = tri
        nodes = sorted([i, j, k])
        i_s, j_s, k_s = nodes
        h_i = emb_dict.get(i_s, np.zeros(d, dtype=np.float32))
        h_j = emb_dict.get(j_s, np.zeros(d, dtype=np.float32))
        h_k = emb_dict.get(k_s, np.zeros(d, dtype=np.float32))
        h_mean = (h_i + h_j + h_k) / 3.0
        h_max  = np.maximum(np.maximum(h_i, h_j), h_k)
        feat = np.concatenate([h_i, h_j, h_k, h_mean, h_max]).astype(np.float32)
        feats.append(feat)
    return np.stack(feats, axis=0)


def under_sample(X, y, ratio=0.33, seed=0):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res


def run_once(dataset, use_temporal, seed=0,
             dim=64, walk_length=40, context_size=20,
             walks_per_node=10, p=1.0, q=1.0,
             batch_size=128, lr=0.01, epochs=10):

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 读 train/test 三元 + G_train/G_test + y
    tris_tr, G_tr, y_tr = load_triangles_graph_labels(dataset, 'train')
    tris_te, G_te, y_te = load_triangles_graph_labels(dataset, 'test')

    # 2) 在 G_tr 上训练 Node2Vec embedding
    print('  Training Node2Vec on G_train...')
    emb_tr = train_pyg_node2vec(
        G_tr, dim=dim, walk_length=walk_length,
        context_size=context_size, walks_per_node=walks_per_node,
        p=p, q=q, batch_size=batch_size, lr=lr, epochs=epochs,
        device=device, seed=seed
    )

    # 3) 构造结构特征（同一 emb 用于 train/test）
    X_tr_struct = build_embed_struct_features(tris_tr, emb_tr)
    X_te_struct = build_embed_struct_features(tris_te, emb_tr)

    # 4) 时间特征
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

    # 5) 欠采样 + LR
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
    parser.add_argument('--runs', type=int, default=3,
                        help='Node2Vec 较慢，先少跑几次')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--walk_length', type=int, default=40)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    ds   = args.dataset
    useT = args.use_temporal

    print(f'===== StructC (Node2Vec embed) + {"Temporal" if useT else "None"} LR multi-run Dataset: {ds} =====')

    aps, aucs = [], []
    for k in range(args.runs):
        seed_k = args.seed + k
        ap, auc = run_once(ds, useT, seed=seed_k,
                           dim=args.dim,
                           walk_length=args.walk_length,
                           context_size=args.context_size,
                           walks_per_node=args.walks_per_node,
                           p=args.p, q=args.q,
                           batch_size=args.batch_size,
                           lr=args.lr, epochs=args.epochs)
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
