# TSMP: Temporal Simplicial Motif Predictor

This repository contains the reference implementation of **TSMP (Temporal Simplicial Motif Predictor)**, a framework for higher–order link prediction (simplicial closure) that augments structural encoders with triangle–level temporal summary features.

TSMP is evaluated on the benchmark datasets of Benson et al. (simplicial closure on temporal simplicial complexes). The code here reproduces the main quantitative results in the paper:

- SMPM (motif–based structural features only)
- TSMP–Motif (SMPM motifs + temporal summaries)
- TSMP–LocalStat (local edge/degree/common–neighbor statistics + temporal summaries)
- TSMP–TriEnv (triangle–environment statistics + temporal summaries)
- TSMP–Embed (Node2Vec–based embeddings + temporal summaries)

---

## 1. Data

### 1.1. Download

All datasets are from the **ScHoLP-Data** repository of Benson et al.:

> https://github.com/arbenson/ScHoLP-Data

Please clone or download that repository and place the datasets under the `datasets/` directory of this project. The recommended layout is:

```text
TSMP/
  datasets/
    email-Enron/
      email-Enron-nverts.txt
      email-Enron-simplices.txt
      email-Enron-times.txt
      ...
    email-Eu/
    contact-high-school/
    contact-primary-school/
    coauth-MAG-History/
    coauth-MAG-Geology/
    coauth-DBLP/
    NDC-classes/
    NDC-substances/
    threads-ask-ubuntu/
    tags-math-sx/
    tags-ask-ubuntu/
    DAWN/
```

TSMP scripts assume this directory structure when reading the raw simplices.

### 1.2. Preprocessing overview
The scripts in this repository follow the same temporal splitting and open–triangle generation protocol as in ScHoLP/SMPM:

- Temporal window splits at 60% / 80% quantiles of timestamps.
- Structural window skeletons are built from simplices in the structural window.
- Open triangles are enumerated on the skeleton; labels are defined by simplicial closure in the subsequent label window.

The preprocessing steps are embedded in the various training scripts (see below); there is no separate preprocessing entry point.

---

## 2. Source files

### 2.1. Files derived from SMPM

The following three scripts are taken (with minimal modifications for integration) from the **Simplicial-Motif-Predictor-Method** project:

- `read_simplices_data.py`
- `construct_motif_feature.py`
- `find_motifs.py`

Original source (SMPM):

> https://github.com/Rm-Y/Simplicial-Motif-Predictor-Method

These scripts are responsible for:

- reading the raw simplicial data from `datasets/`,
- constructing the skeleton graphs and temporal splits,
- computing SMPM-style simplicial motif features (MotifEncoder) for open triangles.

If you use these components, please also acknowledge/cite the SMPM work.

### 2.2. Temporal feature extraction

- `extract_temporal_stats.py`  

This script computes **triangle-level temporal summary features** for each open triangle, based solely on events in the structural window:

- counts of edge occurrences over time,
- normalized first/last times of each edge and their ranges across edges,
- lifetime (span) statistics,
- segment-wise counts over three equal segments (early/mid/late) of the structural window.

The outputs are stored under:

```text
temporal_stats/{dataset}/tri_stats_train.pickle
temporal_stats/{dataset}/tri_stats_test.pickle
```
These files are later loaded by the training scripts as temporal features.

## 3. Training scripts
All training scripts implement a multi-run protocol (e.g., 5 runs with different seeds) and report mean and standard deviation of AP/AUC. They also perform random under-sampling of negatives on the training set to handle class imbalance.
Common options (exact flags may vary slightly per script):

- --dataset: dataset name, e.g. email-Enron, contact-high-school, coauth-DBLP
- --runs: number of runs (default 5 for most scripts)
- --seed: base random seed
- --use_temporal: whether to append temporal features (TSMP variant) or not (struct-only baseline)

### 3.1. Motif-based baseline and TSMP–Motif
```bash
# 1) SMPM baseline (motif only)
python smpm_plus_temporal_lr_multi.py --dataset email-Enron --runs 5

# 2) TSMP–Motif (motif + temporal features)
python smpm_plus_temporal_lr_multi.py --dataset email-Enron --runs 5 --use_temporal
```

### 3.2. TSMP–LocalStat (edge + degree + common neighbors)

```bash
# LocalStat only
python structA_plus_temporal_lr_multi.py --dataset email-Enron --runs 5

# TSMP–LocalStat (LocalStat + temporal)
python structA_plus_temporal_lr_multi.py --dataset email-Enron --runs 5 --use_temporal
```

### 3.3. TSMP–TriEnv (triangle-environment features)
```bash
# TriEnv only
python structB_plus_temporal_lr_multi.py --dataset contact-high-school --runs 5

# TSMP–TriEnv (TriEnv + temporal)
python structB_plus_temporal_lr_multi.py --dataset contact-high-school --runs 5 --use_temporal
```
### 3.4. TSMP–Embed (Node2Vec-based encoder)

```bash
# Embed encoder only
python structC_embed_plus_temporal_lr_multi.py --dataset coauth-DBLP --runs 5

# TSMP–Embed (Embedding + temporal)
python structC_embed_plus_temporal_lr_multi.py --dataset coauth-DBLP --runs 5 --use_temporal
```

## 4. Citation
If you use this code or the TSMP method in your research, please cite both the TSMP paper (under review) and the original ScHoLP and SMPM works.
- Simplicial closure and higher-order link prediction. Austin R. Benson, Rediet Abebe, Michael T. Schaub, Ali Jadbabaie, and Jon Kleinberg. 2018.
- Simplicial motif predictor method for higher-order link prediction. Yang Rongmei, Bo Liu, Linyuan Lü. 2024.
