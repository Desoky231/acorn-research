# Filtered Vector Search: ACORN & SIEVE CPU Baselines

Empirical benchmarking, stress-testing, and failure analysis of graph-based filtered vector search baselines on the labeled SIFT1M dataset (1,000,000 vectors).

---

## 🚀 Key Experimental Findings

### 1. 1M SIFT1M Baseline Replication
* **ACORN Baseline Target:** Replicated SIGMOD '24 target accuracy (**89.85% Recall@10** at 3,156 QPS, 119.9s TTI).
* **Multi-Label Bitmasking:** Encoded multiple labels per vector using 32-bit bitmasks, boosting recall from 89.61% to 89.85%.

### 2. Low-Selectivity "Breaking Point" Stress Test
Empirical evaluation across 10 selectivity levels on 1M vectors with `gamma = 12` (theoretical threshold `s_min ≈ 8.33%`):
* **High Selectivity (20% to 5%):** 97% – 98% Recall@10 (Stable subgraph).
* **Medium Selectivity (1.0%):** 68.00% Recall@10 (Graph begins fragmenting).
* **Low Selectivity (≤ 0.20%):** Catastrophic collapse to **1.50%** and **0.15% Recall@10** due to isolated node islands.

```text
Selectivity (%)   Matching Items   Recall@10 (ef=256)   Throughput (QPS)   Status
-----------------------------------------------------------------------------------------
20.00%            200,000          98.30%               3,087 QPS          Stable
10.00%            100,000          97.25%               2,664 QPS          Stable
 5.00%             50,000          97.20%               3,086 QPS          Stable
 2.00%             20,000          87.15%               3,678 QPS          Starting Drop
 1.00%             10,000          68.00%               3,689 QPS          Fragmenting
 0.50%              5,000          40.60%               4,001 QPS          Degraded
 0.20%              2,000           1.50%              16,274 QPS          CRITICAL FAIL 💥
 0.10%              1,000           0.25%             115,792 QPS          TOTAL COLLAPSE 💥
 0.05%                500           0.10%             149,313 QPS          TOTAL COLLAPSE 💥
 0.01%                100           0.15%             200,705 QPS          TOTAL COLLAPSE 💥
-----------------------------------------------------------------------------------------
```

---

## 🛠️ Build & Run Instructions

### Prerequisites
* GCC / G++ (C++17)
* CMake >= 3.16
* OpenMP

### 1. Build Faiss & ACORN Benchmarks
```bash
cmake -B third_party/ACORN/build -S third_party/ACORN
cmake --build third_party/ACORN/build --target bench_sift1m_labels bench_selectivity_sweep -j 8
```

### 2. Run Benchmarks
```bash
# SIFT1M 1M Baseline Sweep:
./third_party/ACORN/build/demos/bench_sift1m_labels 1000000 12 32 32 1000

# Selectivity Breaking Point Sweep (10 levels from 20% down to 0.01%):
./third_party/ACORN/build/demos/bench_selectivity_sweep 1000000 200
```
