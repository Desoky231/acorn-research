#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <queue>
#include <algorithm>
#include <random>
#include <iomanip>
#include <omp.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexACORN.h>

// Helper to read SIFT1M binary .fbin files
float* read_fbin(const std::string& filepath, size_t& num_pts, size_t& dim, size_t max_pts = 0) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        exit(1);
    }
    
    int n, d;
    file.read(reinterpret_cast<char*>(&n), sizeof(int));
    file.read(reinterpret_cast<char*>(&d), sizeof(int));
    
    num_pts = (max_pts > 0 && max_pts < (size_t)n) ? max_pts : n;
    dim = d;
    
    float* data = new float[num_pts * dim];
    file.read(reinterpret_cast<char*>(data), num_pts * dim * sizeof(float));
    file.close();
    return data;
}

// Compute Euclidean distance squared between two float vectors
inline float compute_l2_sq(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Compute exact brute-force ground truth for a given boolean filter_map
std::vector<std::vector<faiss::idx_t>> compute_ground_truth(
    const float* xb, size_t nb, size_t dim,
    const float* xq, size_t nq,
    const std::vector<char>& filter_map,
    int k
) {
    std::vector<std::vector<faiss::idx_t>> gt(nq, std::vector<faiss::idx_t>(k, -1));
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t q = 0; q < nq; q++) {
        const float* query_vec = xq + q * dim;
        const char* q_filter = filter_map.data() + q * nb;
        
        // Max-heap: pair<distance, vector_id>
        std::priority_queue<std::pair<float, faiss::idx_t>> heap;
        
        for (size_t i = 0; i < nb; i++) {
            if (q_filter[i] == 0) continue;
            
            float dist = compute_l2_sq(query_vec, xb + i * dim, dim);
            
            if ((int)heap.size() < k) {
                heap.push({dist, (faiss::idx_t)i});
            } else if (dist < heap.top().first) {
                heap.pop();
                heap.push({dist, (faiss::idx_t)i});
            }
        }
        
        int idx = (int)heap.size() - 1;
        while (!heap.empty()) {
            if (idx >= 0 && idx < k) {
                gt[q][idx] = heap.top().second;
            }
            heap.pop();
            idx--;
        }
    }
    return gt;
}

// Evaluate Recall@k against exact ground truth
double compute_recall(
    const std::vector<std::vector<faiss::idx_t>>& ground_truth,
    const std::vector<faiss::idx_t>& search_results,
    size_t nq, int k
) {
    size_t total_hits = 0;
    size_t total_expected = 0;
    
    for (size_t q = 0; q < nq; q++) {
        const auto& true_nn = ground_truth[q];
        size_t q_expected = 0;
        for (int i = 0; i < k; i++) {
            if (true_nn[i] != -1) q_expected++;
        }
        if (q_expected == 0) continue;
        total_expected += q_expected;
        
        for (int i = 0; i < k; i++) {
            faiss::idx_t result_id = search_results[q * k + i];
            if (result_id == -1) continue;
            for (int j = 0; j < k; j++) {
                if (result_id == true_nn[j]) {
                    total_hits++;
                    break;
                }
            }
        }
    }
    return (total_expected > 0) ? (double)total_hits / total_expected : 0.0;
}

int main(int argc, char** argv) {
    std::cout << "============================================================" << std::endl;
    std::cout << " ACORN Low-Selectivity Stress Test (\"The Breaking Point\")" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    size_t nb_limit = (argc > 1) ? std::stoul(argv[1]) : 1000000;
    size_t nq_limit = (argc > 2) ? std::stoul(argv[2]) : 200;
    int gamma = 12;
    int M = 32;
    int M_beta = 32;
    int k = 10;
    
    std::cout << " Configuration:" << std::endl;
    std::cout << " - Vectors to index (N): " << nb_limit << std::endl;
    std::cout << " - Queries to test (NQ): " << nq_limit << std::endl;
    std::cout << " - Gamma expansion:      " << gamma << " (s_min ~ " << std::fixed << std::setprecision(2) << (100.0 / gamma) << "%)" << std::endl;
    std::cout << " - Graph degree (M):     " << M << std::endl;
    std::cout << " - M_beta (Compression): " << M_beta << std::endl;
    std::cout << " - Top-k:                " << k << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // 1. Load Dataset
    std::cout << "\n[1/3] Loading dataset from data_sift1m/sift1M..." << std::endl;
    size_t nb = 0, dim = 0, nq = 0, qdim = 0;
    float* xb = read_fbin("../../data_sift1m/sift1M/base.fbin", nb, dim, nb_limit);
    float* xq = read_fbin("../../data_sift1m/sift1M/query.fbin", nq, qdim, nq_limit);
    
    std::cout << " -> Loaded " << nb << " base vectors (dim=" << dim << ")" << std::endl;
    std::cout << " -> Loaded " << nq << " query vectors (dim=" << qdim << ")" << std::endl;
    
    // 2. Build ACORN Index
    std::cout << "\n[2/3] Building ACORN-gamma index..." << std::endl;
    auto t_build_start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> dummy_metadata(nb, 0);
    faiss::IndexACORNFlat acorn_index(dim, M, gamma, dummy_metadata, M_beta);
    acorn_index.acorn.efConstruction = 40;
    acorn_index.add(nb, xb);
    
    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(t_build_end - t_build_start).count();
    std::cout << " -> Index constructed in " << std::fixed << std::setprecision(2) << build_time << " s (TTI)" << std::endl;
    
    // 3. Selectivity Sweep Experiment
    std::vector<double> selectivities = {
        0.20,   // 20.0%  (200,000 matching items)
        0.10,   // 10.0%  (100,000 matching items)
        0.05,   //  5.0%  ( 50,000 matching items)
        0.02,   //  2.0%  ( 20,000 matching items)
        0.01,   //  1.0%  ( 10,000 matching items)
        0.005,  //  0.5%  (  5,000 matching items)
        0.002,  //  0.2%  (  2,000 matching items)
        0.001,  //  0.1%  (  1,000 matching items)
        0.0005, //  0.05% (    500 matching items)
        0.0001  //  0.01% (    100 matching items)
    };
    
    std::cout << "\n[3/3] Running Selectivity Sweeps (10 Levels from 20% down to 0.01%)..." << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;
    std::cout << " Selectivity | Matching Items | Recall@10 (ef=128) | Recall@10 (ef=256) | QPS (ef=256) | Status" << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;
    
    for (double sel : selectivities) {
        size_t expected_matches = (size_t)std::round(sel * nb);
        
        // Generate reproducible boolean filter map for target selectivity
        std::vector<char> filter_map(nq * nb, 0);
        #pragma omp parallel for
        for (size_t q = 0; q < nq; q++) {
            std::mt19937 rng(42 + q * 1000 + (int)(sel * 100000));
            std::bernoulli_distribution dist(sel);
            size_t count = 0;
            for (size_t i = 0; i < nb; i++) {
                if (dist(rng)) {
                    filter_map[q * nb + i] = 1;
                    count++;
                }
            }
            // Ensure at least k items match
            if (count < (size_t)k) {
                for (size_t i = 0; i < (size_t)k; i++) {
                    filter_map[q * nb + i] = 1;
                }
            }
        }
        
        // Ground truth for this selectivity
        auto ground_truth = compute_ground_truth(xb, nb, dim, xq, nq, filter_map, k);
        
        // Test efSearch = 128
        acorn_index.acorn.efSearch = 128;
        std::vector<faiss::idx_t> results_128(nq * k);
        std::vector<float> distances_128(nq * k);
        acorn_index.search(nq, xq, k, distances_128.data(), results_128.data(), filter_map.data());
        double recall_128 = compute_recall(ground_truth, results_128, nq, k) * 100.0;
        
        // Test efSearch = 256
        acorn_index.acorn.efSearch = 256;
        std::vector<faiss::idx_t> results_256(nq * k);
        std::vector<float> distances_256(nq * k);
        
        auto t_search_start = std::chrono::high_resolution_clock::now();
        acorn_index.search(nq, xq, k, distances_256.data(), results_256.data(), filter_map.data());
        auto t_search_end = std::chrono::high_resolution_clock::now();
        
        double search_time = std::chrono::duration<double>(t_search_end - t_search_start).count();
        double qps_256 = nq / search_time;
        double recall_256 = compute_recall(ground_truth, results_256, nq, k) * 100.0;
        
        std::string status = "Stable";
        if (recall_256 < 30.0) {
            status = "CRITICAL FAIL";
        } else if (recall_256 < 60.0) {
            status = "Degraded";
        } else if (recall_256 < 80.0) {
            status = "Fragmenting";
        }
        
        std::cout << " " << std::setw(10) << std::fixed << std::setprecision(2) << (sel * 100.0) << "% | "
                  << std::setw(14) << expected_matches << " | "
                  << std::setw(17) << std::fixed << std::setprecision(2) << recall_128 << "% | "
                  << std::setw(17) << std::fixed << std::setprecision(2) << recall_256 << "% | "
                  << std::setw(11) << std::fixed << std::setprecision(0) << qps_256 << " QPS | "
                  << status << std::endl;
    }
    
    std::cout << "-----------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "\n✨ Selectivity Stress Test Completed!" << std::endl;
    
    delete[] xb;
    delete[] xq;
    return 0;
}
