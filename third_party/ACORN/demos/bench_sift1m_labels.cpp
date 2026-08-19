#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cassert>
#include <unordered_set>
#include <set>
#include <omp.h>

#include <faiss/IndexACORN.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>

// Helper to load .fbin format (uint32_t npts, uint32_t dim, then float data)
float* read_fbin(const std::string& path, size_t& npts_out, size_t& dim_out, size_t max_pts = 0) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening binary file: " << path << std::endl;
        exit(1);
    }
    uint32_t npts = 0, dim = 0;
    file.read(reinterpret_cast<char*>(&npts), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    
    if (max_pts > 0 && max_pts < npts) {
        npts = max_pts;
    }
    
    npts_out = npts;
    dim_out = dim;
    
    float* data = new float[npts * dim];
    file.read(reinterpret_cast<char*>(data), npts * dim * sizeof(float));
    file.close();
    return data;
}

// Helper to load multi-tag labels from base.txt
std::vector<std::vector<int>> read_tags(const std::string& path, size_t max_lines = 0) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error opening tags file: " << path << std::endl;
        exit(1);
    }
    std::vector<std::vector<int>> all_tags;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> tags;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            try {
                if (!item.empty()) {
                    tags.push_back(std::stoi(item));
                }
            } catch (...) {}
        }
        all_tags.push_back(tags);
        if (max_lines > 0 && all_tags.size() >= max_lines) {
            break;
        }
    }
    file.close();
    return all_tags;
}

// Helper to load query filters from query.txt
std::vector<int> read_query_filters(const std::string& path, size_t max_lines = 0) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error opening query filters: " << path << std::endl;
        exit(1);
    }
    std::vector<int> filters;
    std::string line;
    while (std::getline(file, line)) {
        try {
            if (!line.empty()) {
                filters.push_back(std::stoi(line));
            }
        } catch (...) {}
        if (max_lines > 0 && filters.size() >= max_lines) {
            break;
        }
    }
    file.close();
    return filters;
}

// Exact Brute Force Ground Truth Calculator for Filtered Search
std::vector<std::vector<faiss::idx_t>> compute_ground_truth(
    const float* xb, size_t nb, size_t dim,
    const float* xq, size_t nq,
    const std::vector<std::vector<int>>& base_tags,
    const std::vector<int>& query_filters,
    int k
) {
    std::vector<std::vector<faiss::idx_t>> gt(nq);
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t q = 0; q < nq; q++) {
        int filter_tag = query_filters[q];
        std::vector<std::pair<float, faiss::idx_t>> candidates;
        
        for (size_t i = 0; i < nb; i++) {
            // Check predicate: does base vector i contain query filter tag?
            const auto& tags = base_tags[i];
            bool match = false;
            for (int t : tags) {
                if (t == filter_tag) {
                    match = true;
                    break;
                }
            }
            if (!match) continue;
            
            // Compute Euclidean distance squared
            float dist = 0.0f;
            for (size_t d = 0; d < dim; d++) {
                float diff = xb[i * dim + d] - xq[q * dim + d];
                dist += diff * diff;
            }
            candidates.push_back({dist, static_cast<faiss::idx_t>(i)});
        }
        
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + std::min((size_t)k, candidates.size()),
            candidates.end()
        );
        
        for (size_t r = 0; r < std::min((size_t)k, candidates.size()); r++) {
            gt[q].push_back(candidates[r].second);
        }
    }
    
    return gt;
}

int main(int argc, char* argv[]) {
    size_t nb_limit = (argc > 1) ? std::stoul(argv[1]) : 10000;
    int gamma = (argc > 2) ? std::stoi(argv[2]) : 12;
    int M = (argc > 3) ? std::stoi(argv[3]) : 32;
    int M_beta = (argc > 4) ? std::stoi(argv[4]) : 32;
    size_t nq_limit = (argc > 5) ? std::stoul(argv[5]) : 1000;
    int k = 10;
    
    std::cout << "============================================================" << std::endl;
    std::cout << " ACORN Filtered Vector Search Benchmark on SIFT1M" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << " Configuration:" << std::endl;
    std::cout << " - Vectors to index (N): " << nb_limit << std::endl;
    std::cout << " - Gamma expansion:     " << gamma << std::endl;
    std::cout << " - M (Graph degree):    " << M << std::endl;
    std::cout << " - M_beta (Compression):" << M_beta << std::endl;
    std::cout << " - Queries to run (NQ): " << nq_limit << std::endl;
    std::cout << " - Top-k:               " << k << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // 1. Load Data
    std::cout << "\n[1/4] Loading SIFT1M dataset from data_sift1m/sift1M..." << std::endl;
    size_t nb = 0, dim = 0, nq = 0, qdim = 0;
    
    float* xb = read_fbin("../../data_sift1m/sift1M/base.fbin", nb, dim, nb_limit);
    float* xq = read_fbin("../../data_sift1m/sift1M/query.fbin", nq, qdim, nq_limit);
    auto base_tags = read_tags("../../data_sift1m/sift1M/base.txt", nb_limit);
    auto query_filters = read_query_filters("../../data_sift1m/sift1M/query.txt", nq_limit);
    
    std::cout << " -> Loaded " << nb << " base vectors (dim=" << dim << ")" << std::endl;
    std::cout << " -> Loaded " << nq << " query vectors (dim=" << qdim << ")" << std::endl;
    std::cout << " -> Loaded " << base_tags.size() << " tag rows" << std::endl;
    
    // Encode all multi-label tags into a 32-bit bitmask for ACORN constructor metadata
    std::vector<int> metadata(nb, 0);
    for (size_t i = 0; i < nb; i++) {
        int mask = 0;
        for (int tag : base_tags[i]) {
            if (tag >= 0 && tag < 32) {
                mask |= (1 << tag);
            }
        }
        metadata[i] = mask;
    }
    
    // Build boolean filter map: filter_map[q * nb + i] = 1 if point i satisfies query q
    std::cout << " -> Building query filter map..." << std::endl;
    std::vector<char> filter_map(nq * nb, 0);
    for (size_t q = 0; q < nq; q++) {
        int target_tag = query_filters[q];
        for (size_t i = 0; i < nb; i++) {
            for (int t : base_tags[i]) {
                if (t == target_tag) {
                    filter_map[q * nb + i] = 1;
                    break;
                }
            }
        }
    }
    
    // 2. Compute Exact Ground Truth
    std::cout << "\n[2/4] Computing exact brute-force ground truth for " << nq << " queries..." << std::endl;
    auto t_gt_start = std::chrono::high_resolution_clock::now();
    auto ground_truth = compute_ground_truth(xb, nb, dim, xq, nq, base_tags, query_filters, k);
    auto t_gt_end = std::chrono::high_resolution_clock::now();
    double gt_time = std::chrono::duration<double>(t_gt_end - t_gt_start).count();
    std::cout << " -> Ground truth computed in " << gt_time << " s" << std::endl;
    
    // 3. Build ACORN Index
    std::cout << "\n[3/4] Building ACORN-gamma index (gamma=" << gamma << ", M=" << M << ", M_beta=" << M_beta << ")..." << std::endl;
    auto t_build_start = std::chrono::high_resolution_clock::now();
    
    faiss::IndexACORNFlat acorn_index(dim, M, gamma, metadata, M_beta);
    acorn_index.acorn.efConstruction = 40;
    acorn_index.add(nb, xb);
    
    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(t_build_end - t_build_start).count();
    std::cout << " -> ACORN-gamma constructed successfully in " << build_time << " s (TTI)" << std::endl;
    
    // 4. Benchmark Query Sweeps over efSearch
    std::cout << "\n[4/4] Running Filtered Search Query Sweeps over efSearch..." << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << " efSearch | Recall@10 (%) | QPS (Queries/sec) | Avg Latency (ms)" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    
    std::vector<int> ef_values = {16, 32, 64, 128, 256};
    
    for (int efs : ef_values) {
        acorn_index.acorn.efSearch = efs;
        
        std::vector<faiss::idx_t> result_ids(nq * k);
        std::vector<float> result_dists(nq * k);
        
        auto t_search_start = std::chrono::high_resolution_clock::now();
        
        acorn_index.search(nq, xq, k, result_dists.data(), result_ids.data(), filter_map.data());
        
        auto t_search_end = std::chrono::high_resolution_clock::now();
        double search_time = std::chrono::duration<double>(t_search_end - t_search_start).count();
        
        // Evaluate Recall@10
        size_t total_gt_found = 0;
        size_t total_gt_possible = 0;
        
        for (size_t q = 0; q < nq; q++) {
            const auto& gt_q = ground_truth[q];
            total_gt_possible += gt_q.size();
            
            std::unordered_set<faiss::idx_t> gt_set(gt_q.begin(), gt_q.end());
            for (int r = 0; r < k; r++) {
                faiss::idx_t found_id = result_ids[q * k + r];
                if (found_id >= 0 && gt_set.count(found_id)) {
                    total_gt_found++;
                }
            }
        }
        
        double recall = (total_gt_possible > 0) ? (static_cast<double>(total_gt_found) / total_gt_possible) : 0.0;
        double qps = nq / search_time;
        double avg_lat_ms = (search_time / nq) * 1000.0;
        
        printf(" %8d | %12.2f%% | %17.1f | %16.3f\n", efs, recall * 100.0, qps, avg_lat_ms);
    }
    
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "\n✨ Benchmark completed successfully!" << std::endl;
    
    delete[] xb;
    delete[] xq;
    return 0;
}
