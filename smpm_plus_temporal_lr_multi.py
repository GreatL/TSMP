# smpm_plus_temporal_lr_multi.py
# SMPM motif 特征 + triangle-level temporal stats + Logistic Regression，多次运行统计 mean±std

import os
import pickle
import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler


def load_smpm_features(dataset, split):
    path = f'split_dataset/{dataset}/{split}_mean.pickle'
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'rb') as f:
        df = pickle.load(f)
    df_norm = df.apply(lambda a: (a - a.min()) / (a.max() - a.min() + 1e-12))
    X = df_norm.values.astype(np.float32)
    return X


def load_labels(dataset, split):
    path = f'processing_dataset/{dataset}/y_{split}.pickle'
    with open(path, 'rb') as f:
        y = pickle.load(f)
    return np.array(y, dtype=np.int64)


def load_tri_temporal_stats(dataset, split):
    path = f'temporal_stats/{dataset}/tri_stats_{split}.pickle'
    if not os.path.exists(path):
        print(f'[WARN] {path} not found, temporal features disabled for {split}.')
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

    tri_time = np.stack(feats, axis=0)
    return tri_time


def normalize_time_feats(T):
    mins = T.min(axis=0)
    maxs = T.max(axis=0)
    return (T - mins) / (maxs - mins + 1e-12)


def under_sample(X, y, ratio=0.33, seed=0):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=seed)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res


def run_once(dataset, use_temporal, seed):
    np.random.seed(seed)
    # 1) load
    X_train_smpm = load_smpm_features(dataset, 'train')
    X_test_smpm  = load_smpm_features(dataset, 'test')
    y_train = load_labels(dataset, 'train')
    y_test  = load_labels(dataset, 'test')

    if use_temporal:
        tri_time_train = load_tri_temporal_stats(dataset, 'train')
        tri_time_test  = load_tri_temporal_stats(dataset, 'test')
        if tri_time_train is None or tri_time_test is None:
            use_temporal = False

    if use_temporal:
        tri_time_train_norm = normalize_time_feats(tri_time_train)
        tri_time_test_norm  = normalize_time_feats(tri_time_test)
        X_train = np.concatenate([X_train_smpm, tri_time_train_norm], axis=1)
        X_test  = np.concatenate([X_test_smpm,  tri_time_test_norm],  axis=1)
    else:
        X_train = X_train_smpm
        X_test  = X_test_smpm

    # 2) undersample + LR
    X_train_bal, y_train_bal = under_sample(X_train, y_train, ratio=0.33, seed=seed)

    clf = LogisticRegression(
        solver='liblinear',
        penalty='l2',
        max_iter=1000
    )
    clf.fit(X_train_bal, y_train_bal)

    y_score = clf.predict_proba(X_test)[:, 1]
    if y_test.sum() == 0:
        ap = 0.0
        auc = np.nan
    else:
        ap = average_precision_score(y_test, y_score)
        try:
            auc = roc_auc_score(y_test, y_score)
        except ValueError:
            auc = np.nan
    return ap, auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='email-Enron')
    parser.add_argument('--use_temporal', action='store_true')
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()

    ds = args.dataset
    print(f'===== SMPM+{"Temporal" if args.use_temporal else "None"} LR multi-run Dataset: {ds} =====')

    aps, aucs = [], []
    for k in range(args.runs):
        ap, auc = run_once(ds, args.use_temporal, seed=k)
        aps.append(ap); aucs.append(auc)
        print(f'  Run {k}: AP={ap:.4f}, AUC={auc:.4f}')

    aps = np.array(aps)
    aucs = np.array(aucs)

    ap_mean  = float(np.nanmean(aps))
    ap_std   = float(np.nanstd(aps, ddof=1))
    auc_mean = float(np.nanmean(aucs))
    auc_std  = float(np.nanstd(aucs, ddof=1))

    print(f'AP mean={ap_mean:.4f}, std={ap_std:.4f}')
    print(f'AUC mean={auc_mean:.4f}, std={auc_std:.4f}')
