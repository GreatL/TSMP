# extract_temporal_stats.py
# 从原始 simplices + 时间戳中提取边、节点、三元组的时序统计信息，供后续模型使用
#
# 输入：
#   datasets/{dataset}/{dataset}-nverts.txt
#   datasets/{dataset}/{dataset}-simplices.txt
#   datasets/{dataset}/{dataset}-times.txt
#   processing_dataset/{dataset}/trg_open_train.pickle, trg_open_test.pickle
#
# 输出：
#   temporal_stats/{dataset}/edge_stats_train.npz
#   temporal_stats/{dataset}/edge_stats_test.npz
#   temporal_stats/{dataset}/node_stats_train.npz
#   temporal_stats/{dataset}/node_stats_test.npz
#   temporal_stats/{dataset}/tri_stats_train.npz
#   temporal_stats/{dataset}/tri_stats_test.npz

import os
import pickle
import argparse
import numpy as np
from collections import defaultdict
import itertools

def read_simplices_raw(dataset):
    base = f'datasets/{dataset}'
    with open(os.path.join(base, f'{dataset}-nverts.txt'), 'r') as f:
        nv_list = [int(x) for x in f]
    with open(os.path.join(base, f'{dataset}-simplices.txt'), 'r') as f:
        sp_list = [int(x) for x in f]
    with open(os.path.join(base, f'{dataset}-times.txt'), 'r') as f:
        tm_list = [int(x) for x in f]
    return nv_list, sp_list, tm_list

def split_by_time_window(nv_list, sp_list, tm_list, start_ratio, end_ratio):
    """按时间百分位 [start_ratio, end_ratio] 选出在该窗内的 simplices（节点集合+时间）"""
    tm_arr = np.array(tm_list)
    t_start = int(np.round(np.percentile(tm_arr, start_ratio)))
    t_end   = int(np.round(np.percentile(tm_arr, min(end_ratio,100))))

    simplices = []
    curr = 0
    for nv, t in zip(nv_list, tm_list):
        nodes = sp_list[curr:curr+nv]
        curr += nv
        if t < t_start or t > t_end:
            continue
        simplices.append((tuple(nodes), t))
    return simplices, (t_start, t_end)

def compute_edge_time_stats(simplices, num_segments=3):
    """
    输入：在某个结构窗内的 simplices = [(nodes, t), ...]
    输出：
      edge_stats: dict[(u,v)] -> dict of time stats
      node_stats: dict[node] -> dict of aggregated time stats
    """
    edge_times = defaultdict(list)
    # 1) 对每条边收集所有出现时间
    for nodes, t in simplices:
        if len(nodes) < 2:
            continue
        for u, v in itertools.combinations(nodes, 2):
            e = tuple(sorted((u,v)))
            edge_times[e].append(t)

    if not edge_times:
        return {}, {}

    # 全局时间范围（用于归一化 & segments）
    all_ts = [t for ts in edge_times.values() for t in ts]
    t_min, t_max = min(all_ts), max(all_ts)
    t_span = max(t_max - t_min, 1)

    # 预计算 segments 边界
    seg_bounds = [t_min + (t_span * k) / num_segments for k in range(1, num_segments)]
    # seg_bounds: 长度 num_segments-1，把 [t_min, t_max] 均分为 num_segments 段

    edge_stats = {}
    node_stats_tmp = defaultdict(lambda: defaultdict(float))  # 后面再整理

    for e, ts in edge_times.items():
        ts_sorted = sorted(ts)
        count = len(ts_sorted)
        first_t = ts_sorted[0]
        last_t  = ts_sorted[-1]

        # inter-event intervals
        if count >= 2:
            intervals = [ts_sorted[i+1] - ts_sorted[i] for i in range(count-1)]
            inter_mean = float(np.mean(intervals))
            inter_std  = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0
        else:
            inter_mean = 0.0
            inter_std  = 0.0

        # segments counts
        seg_counts = np.zeros(num_segments, dtype=np.int32)
        for t in ts_sorted:
            # 找所属 segment
            seg = 0
            while seg < num_segments-1 and t > seg_bounds[seg]:
                seg += 1
            seg_counts[seg] += 1

        # 归一化时间到 [0,1]
        first_norm = (first_t - t_min) / t_span
        last_norm  = (last_t  - t_min) / t_span

        edge_stats[e] = dict(
            count = count,
            first_time = first_t,
            last_time  = last_t,
            first_norm = first_norm,
            last_norm  = last_norm,
            span       = last_t - first_t,
            inter_mean = inter_mean,
            inter_std  = inter_std,
            seg_counts = seg_counts
        )

        # 聚合到节点上
        u, v = e
        for node in (u,v):
            ns = node_stats_tmp[node]
            ns['degree'] += 1
            ns['total_count'] += count
            ns['last_time'] = max(ns.get('last_time', t_min), last_t)
            if 'first_time' not in ns:
                ns['first_time'] = first_t
            else:
                ns['first_time'] = min(ns['first_time'], first_t)
            # segments
            if 'seg_counts' not in ns:
                ns['seg_counts'] = seg_counts.copy()
            else:
                ns['seg_counts'] += seg_counts

    # 整理 node_stats（归一化时间）
    node_stats = {}
    for node, ns in node_stats_tmp.items():
        first_t = ns.get('first_time', t_min)
        last_t  = ns.get('last_time',  t_min)
        node_stats[node] = dict(
            degree       = int(ns['degree']),
            total_count  = float(ns['total_count']),
            first_time   = first_t,
            last_time    = last_t,
            first_norm   = (first_t - t_min) / t_span,
            last_norm    = (last_t  - t_min) / t_span,
            seg_counts   = ns['seg_counts'].astype(np.int32)
        )

    return edge_stats, node_stats


def compute_tri_stats(open_tris, edge_stats, num_segments=3):
    """
    对 open triangles τ=(i,j,k)，基于 edge_stats 进行简单聚合：
    输出 tri_stats: list 与 open_tris 对齐，每个元素是 dict
    """
    tri_stats = []
    for tri in open_tris:
        i, j, k = tri
        edges = [
            tuple(sorted((i,j))),
            tuple(sorted((j,k))),
            tuple(sorted((i,k)))
        ]
        # 收集边级统计（缺失边则跳过）
        es = [edge_stats[e] for e in edges if e in edge_stats]
        if len(es) < 3:
            tri_stats.append(None)
            continue

        counts      = np.array([d['count']       for d in es], dtype=float)
        last_norms  = np.array([d['last_norm']   for d in es], dtype=float)
        first_norms = np.array([d['first_norm']  for d in es], dtype=float)
        spans       = np.array([d['span']        for d in es], dtype=float)
        # segments sum
        seg_sums    = np.sum([d['seg_counts'] for d in es], axis=0).astype(float)

        tri_stats.append(dict(
            count_sum   = float(counts.sum()),
            count_mean  = float(counts.mean()),
            count_max   = float(counts.max()),
            last_max    = float(last_norms.max()),
            last_min    = float(last_norms.min()),
            last_range  = float(last_norms.max() - last_norms.min()),
            first_max   = float(first_norms.max()),
            first_min   = float(first_norms.min()),
            first_range = float(first_norms.max() - first_norms.min()),
            span_max    = float(spans.max()),
            span_mean   = float(spans.mean()),
            seg_sums    = seg_sums
        ))
    return tri_stats


def save_edge_node_tri_stats(dataset, split, edge_stats, node_stats, tri_stats):
    save_dir = f'temporal_stats/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(save_dir, f'edge_stats_{split}.npz'),
        # 存成两个数组：edges 和 相应字段
        edges = np.array(list(edge_stats.keys()), dtype=np.int64),
        count = np.array([d['count'] for d in edge_stats.values()], dtype=np.int32),
        first_time = np.array([d['first_time'] for d in edge_stats.values()], dtype=np.int64),
        last_time  = np.array([d['last_time']  for d in edge_stats.values()], dtype=np.int64),
        first_norm = np.array([d['first_norm'] for d in edge_stats.values()], dtype=np.float32),
        last_norm  = np.array([d['last_norm']  for d in edge_stats.values()], dtype=np.float32),
        span       = np.array([d['span']       for d in edge_stats.values()], dtype=np.int64),
        inter_mean = np.array([d['inter_mean'] for d in edge_stats.values()], dtype=np.float32),
        inter_std  = np.array([d['inter_std']  for d in edge_stats.values()], dtype=np.float32),
        seg_counts = np.stack([d['seg_counts'] for d in edge_stats.values()], axis=0)
    )
    np.savez_compressed(
        os.path.join(save_dir, f'node_stats_{split}.npz'),
        nodes      = np.array(list(node_stats.keys()), dtype=np.int64),
        degree     = np.array([d['degree']      for d in node_stats.values()], dtype=np.int32),
        total_cnt  = np.array([d['total_count'] for d in node_stats.values()], dtype=np.float32),
        first_time = np.array([d['first_time']  for d in node_stats.values()], dtype=np.int64),
        last_time  = np.array([d['last_time']   for d in node_stats.values()], dtype=np.int64),
        first_norm = np.array([d['first_norm']  for d in node_stats.values()], dtype=np.float32),
        last_norm  = np.array([d['last_norm']   for d in node_stats.values()], dtype=np.float32),
        seg_counts = np.stack([d['seg_counts']  for d in node_stats.values()], axis=0)
    )
    # tri_stats 是跟 open_tris 对齐的 list[dict or None]
    # 为了方便，直接 pickle 保存
    with open(os.path.join(save_dir, f'tri_stats_{split}.pickle'), 'wb') as f:
        pickle.dump(tri_stats, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='email-Enron')
    parser.add_argument('--num_segments', type=int, default=3,
                        help='number of time segments within a window')
    args = parser.parse_args()

    ds = args.dataset
    print('===== Extract temporal stats for dataset:', ds, '=====')

    nv_list, sp_list, tm_list = read_simplices_raw(ds)

    # 结构窗划分：train: [0,60], test: [0,80]
    windows = {
        'train': (0, 60),
        'test':  (0, 80)
    }

    for split, (start_r, end_r) in windows.items():
        print(f'-- split={split}, window=[{start_r},{end_r}] --')
        simplices, (t_start, t_end) = split_by_time_window(nv_list, sp_list, tm_list,
                                                          start_ratio=start_r,
                                                          end_ratio=end_r)
        print(f'  simplices in window: {len(simplices)}, t_start={t_start}, t_end={t_end}')

        edge_stats, node_stats = compute_edge_time_stats(simplices, num_segments=args.num_segments)
        print(f'  edges: {len(edge_stats)}, nodes: {len(node_stats)}')

        # 加载 open triangles，用于 tri-level stats
        base = f'processing_dataset/{ds}'
        with open(os.path.join(base, f'trg_open_{split}.pickle'), 'rb') as f:
            open_tris = pickle.load(f)
        tri_stats = compute_tri_stats(open_tris, edge_stats, num_segments=args.num_segments)
        print(f'  tri_stats computed for {len(tri_stats)} triangles')

        save_edge_node_tri_stats(ds, split, edge_stats, node_stats, tri_stats)
        print('  saved to temporal_stats/{}\n'.format(ds))
