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
#include <unordered_map>
#include <iomanip>
#include <omp.h>

#include <faiss/IndexACORN.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>

// Robust file path finder
std::string find_valid_path(const std::string& filename) {
    std::vector<std::string> prefixes = {
        "data_sift1m/sift1M/",
        "../data_sift1m/sift1M/",
        "../../data_sift1m/sift1M/",
        "../../../data_sift1m/sift1M/",
        "/mnt/d/Database/Acorn/data_sift1m/sift1M/",
        "D:/Database/Acorn/data_sift1m/sift1M/"
    };
    for (const auto& pre : prefixes) {
        std::string full = pre + filename;
        std::ifstream f(full);
        if (f.good()) {
            return full;
        }
    }
    return filename;
}

std::string find_output_path(const std::string& filename) {
    std::vector<std::string> prefixes = {
        "findings/",
        "../findings/",
        "../../findings/",
        "../../../findings/",
        "/mnt/d/Database/Acorn/findings/",
        "D:/Database/Acorn/findings/"
    };
    for (const auto& pre : prefixes) {
        std::string full = pre + filename;
        std::ofstream f(full, std::ios::app);
        if (f.good()) {
            return full;
        }
    }
    return filename;
}

// Helper to load .fbin format (uint32_t npts, uint32_t dim, then float data)
float* read_fbin(const std::string& path, size_t& npts_out, size_t& dim_out, size_t max_pts = 0) {
    std::string actual_path = find_valid_path(path);
    std::ifstream file(actual_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening binary file: " << actual_path << " (tried: " << path << ")" << std::endl;
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
    std::cout << " -> Loaded " << actual_path << " (" << npts << " vectors, dim=" << dim << ")" << std::endl;
    return data;
}

// Helper to load multi-tag labels from base.txt
std::vector<std::vector<int>> read_tags(const std::string& path, size_t max_lines = 0) {
    std::string actual_path = find_valid_path(path);
    std::ifstream file(actual_path);
    if (!file.is_open()) {
        std::cerr << "Error opening tags file: " << actual_path << std::endl;
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
    std::cout << " -> Loaded " << actual_path << " (" << all_tags.size() << " tag rows)" << std::endl;
    return all_tags;
}

// Helper to load query filters from query.txt
std::vector<int> read_query_filters(const std::string& path, size_t max_lines = 0) {
    std::string actual_path = find_valid_path(path);
    std::ifstream file(actual_path);
    if (!file.is_open()) {
        std::cerr << "Error opening query filters: " << actual_path << std::endl;
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
    std::cout << " -> Loaded " << actual_path << " (" << filters.size() << " filter entries)" << std::endl;
    return filters;
}

// Memory-efficient Ground Truth for Single-label Queries
std::vector<std::vector<faiss::idx_t>> compute_gt_single(
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
        const float* query_vec = xq + q * dim;
        std::vector<std::pair<float, faiss::idx_t>> candidates;
        
        for (size_t i = 0; i < nb; i++) {
            const auto& tags = base_tags[i];
            bool match = false;
            for (int t : tags) {
                if (t == filter_tag) {
                    match = true;
                    break;
                }
            }
            if (!match) continue;
            
            float dist = 0.0f;
            const float* base_vec = xb + i * dim;
            for (size_t d = 0; d < dim; d++) {
                float diff = base_vec[d] - query_vec[d];
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

// Memory-efficient Ground Truth for Two-label Conjunction Queries
std::vector<std::vector<faiss::idx_t>> compute_gt_conjunction(
    const float* xb, size_t nb, size_t dim,
    const float* xq, size_t nq,
    const std::vector<std::vector<int>>& base_tags,
    const std::vector<std::pair<int, int>>& conj_queries,
    int k
) {
    std::vector<std::vector<faiss::idx_t>> gt(nq);
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t q = 0; q < nq; q++) {
        int t1 = conj_queries[q].first;
        int t2 = conj_queries[q].second;
        const float* query_vec = xq + q * dim;
        std::vector<std::pair<float, faiss::idx_t>> candidates;
        
        for (size_t i = 0; i < nb; i++) {
            const auto& tags = base_tags[i];
            bool has_t1 = false, has_t2 = false;
            for (int t : tags) {
                if (t == t1) has_t1 = true;
                if (t == t2) has_t2 = true;
            }
            if (!has_t1 || !has_t2) continue;
            
            float dist = 0.0f;
            const float* base_vec = xb + i * dim;
            for (size_t d = 0; d < dim; d++) {
                float diff = base_vec[d] - query_vec[d];
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

// Compute Recall@k
double evaluate_recall(
    const std::vector<std::vector<faiss::idx_t>>& ground_truth,
    const std::vector<faiss::idx_t>& result_ids,
    size_t nq, int k
) {
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
    return (total_gt_possible > 0) ? (static_cast<double>(total_gt_found) / total_gt_possible) : 0.0;
}

struct BenchmarkRow {
    std::string workload;
    int efSearch;
    size_t queries;
    double recall10;
    double qps;
    double latency_ms;
};

int main(int argc, char* argv[]) {
    size_t nb_limit = (argc > 1) ? std::stoul(argv[1]) : 1000000;
    size_t nq_limit = (argc > 2) ? std::stoul(argv[2]) : 10000;
    int gamma = (argc > 3) ? std::stoi(argv[3]) : 12;
    int M = (argc > 4) ? std::stoi(argv[4]) : 32;
    int M_beta = (argc > 5) ? std::stoi(argv[5]) : 32;
    int k = 10;
    
    std::cout << "==========================================================================" << std::endl;
    std::cout << " ACORN: Recall vs QPS Hyperparameter Sweep (Matching Ali's PVLDB Format)  " << std::endl;
    std::cout << "==========================================================================" << std::endl;
    std::cout << " Configuration:" << std::endl;
    std::cout << " - Vectors to index (N): " << nb_limit << std::endl;
    std::cout << " - Queries to test (NQ): " << nq_limit << std::endl;
    std::cout << " - Gamma expansion:      " << gamma << std::endl;
    std::cout << " - Graph degree (M):     " << M << std::endl;
    std::cout << " - M_beta:               " << M_beta << std::endl;
    std::cout << " - Top-k:                " << k << std::endl;
    std::cout << "==========================================================================" << std::endl;
    
    // 1. Load Data
    std::cout << "\n[1/5] Loading SIFT1M dataset..." << std::endl;
    size_t nb = 0, dim = 0, nq = 0, qdim = 0;
    
    float* xb = read_fbin("base.fbin", nb, dim, nb_limit);
    float* xq = read_fbin("query.fbin", nq, qdim, nq_limit);
    auto base_tags = read_tags("base.txt", nb_limit);
    auto query_filters = read_query_filters("query.txt", nq_limit);
    
    // Encode multi-label tags into 32-bit bitmasks for ACORN constructor metadata
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
    
    // 2. Prepare Workload 1: Single-label Filter
    std::cout << "\n[2/5] Computing Ground Truth for Single-label Filter (" << nq << " queries)..." << std::endl;
    auto t_gt1_start = std::chrono::high_resolution_clock::now();
    auto gt_single = compute_gt_single(xb, nb, dim, xq, nq, base_tags, query_filters, k);
    auto t_gt1_end = std::chrono::high_resolution_clock::now();
    double gt1_time = std::chrono::duration<double>(t_gt1_end - t_gt1_start).count();
    std::cout << " -> Ground truth computed in " << std::fixed << std::setprecision(2) << gt1_time << " s" << std::endl;
    
    // 3. Prepare Workload 2: Two-label Conjunction (Tag1 AND Tag2)
    std::cout << "\n[3/5] Computing Ground Truth for Two-label Conjunction (" << nq << " queries)..." << std::endl;
    std::vector<std::pair<int, int>> two_label_queries(nq);
    for (size_t q = 0; q < nq; q++) {
        int tag1 = query_filters[q] % 16;
        int tag2 = (query_filters[q] + 1) % 16;
        if (tag1 == tag2) tag2 = (tag1 + 2) % 16;
        two_label_queries[q] = {tag1, tag2};
    }
    
    auto t_gt2_start = std::chrono::high_resolution_clock::now();
    auto gt_conjunction = compute_gt_conjunction(xb, nb, dim, xq, nq, base_tags, two_label_queries, k);
    auto t_gt2_end = std::chrono::high_resolution_clock::now();
    double gt2_time = std::chrono::duration<double>(t_gt2_end - t_gt2_start).count();
    std::cout << " -> Ground truth computed in " << std::fixed << std::setprecision(2) << gt2_time << " s" << std::endl;
    
    // 4. Build ACORN-gamma Index
    std::cout << "\n[4/5] Building ACORN-gamma index (gamma=" << gamma << ", M=" << M << ", M_beta=" << M_beta << ")..." << std::endl;
    auto t_build_start = std::chrono::high_resolution_clock::now();
    
    faiss::IndexACORNFlat acorn_index(dim, M, gamma, metadata, M_beta);
    acorn_index.acorn.efConstruction = 40;
    acorn_index.add(nb, xb);
    
    auto t_build_end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(t_build_end - t_build_start).count();
    std::cout << " -> ACORN-gamma index built in " << std::fixed << std::setprecision(2) << build_time << " s (TTI)" << std::endl;
    
    // 5. Benchmark efSearch Sweeps in Batches of 1000 Queries
    std::vector<int> ef_values = {32, 40, 64, 100, 200, 256, 512};
    std::vector<BenchmarkRow> all_results;
    size_t batch_size = 1000;
    
    std::cout << "\n[5/5] Running Query Sweeps over efSearch: {32, 40, 64, 100, 200, 256, 512}..." << std::endl;
    
    // Sweep Workload 1: Single-label filter
    std::cout << "\n>>> Benchmarking Workload: Single-label filter <<<" << std::endl;
    for (int efs : ef_values) {
        acorn_index.acorn.efSearch = efs;
        std::vector<faiss::idx_t> results(nq * k);
        std::vector<float> dists(nq * k);
        
        double total_search_time = 0.0;
        
        for (size_t q_offset = 0; q_offset < nq; q_offset += batch_size) {
            size_t cur_batch = std::min(batch_size, nq - q_offset);
            
            // Build batch filter map
            std::vector<char> batch_filter_map(cur_batch * nb, 0);
            #pragma omp parallel for
            for (size_t b = 0; b < cur_batch; b++) {
                size_t q = q_offset + b;
                int target_tag = query_filters[q];
                for (size_t i = 0; i < nb; i++) {
                    for (int t : base_tags[i]) {
                        if (t == target_tag) {
                            batch_filter_map[b * nb + i] = 1;
                            break;
                        }
                    }
                }
            }
            
            auto t_start = std::chrono::high_resolution_clock::now();
            acorn_index.search(
                cur_batch,
                xq + q_offset * dim,
                k,
                dists.data() + q_offset * k,
                results.data() + q_offset * k,
                batch_filter_map.data()
            );
            auto t_end = std::chrono::high_resolution_clock::now();
            total_search_time += std::chrono::duration<double>(t_end - t_start).count();
        }
        
        double recall = evaluate_recall(gt_single, results, nq, k) * 100.0;
        double qps = nq / total_search_time;
        double lat = (total_search_time / nq) * 1000.0;
        
        all_results.push_back({"Single-label filter", efs, nq, recall, qps, lat});
        std::cout << " efSearch=" << std::setw(3) << efs << " | Recall@10=" << std::fixed << std::setprecision(2) << std::setw(6) << recall << "% | QPS=" << std::setw(8) << std::fixed << std::setprecision(1) << qps << " | Lat=" << std::fixed << std::setprecision(3) << lat << " ms" << std::endl;
    }
    
    // Sweep Workload 2: Two-label conjunction
    std::cout << "\n>>> Benchmarking Workload: Two-label conjunction <<<" << std::endl;
    for (int efs : ef_values) {
        acorn_index.acorn.efSearch = efs;
        std::vector<faiss::idx_t> results(nq * k);
        std::vector<float> dists(nq * k);
        
        double total_search_time = 0.0;
        
        for (size_t q_offset = 0; q_offset < nq; q_offset += batch_size) {
            size_t cur_batch = std::min(batch_size, nq - q_offset);
            
            // Build batch filter map
            std::vector<char> batch_filter_map(cur_batch * nb, 0);
            #pragma omp parallel for
            for (size_t b = 0; b < cur_batch; b++) {
                size_t q = q_offset + b;
                int t1 = two_label_queries[q].first;
                int t2 = two_label_queries[q].second;
                for (size_t i = 0; i < nb; i++) {
                    bool has_t1 = false, has_t2 = false;
                    for (int t : base_tags[i]) {
                        if (t == t1) has_t1 = true;
                        if (t == t2) has_t2 = true;
                    }
                    if (has_t1 && has_t2) {
                        batch_filter_map[b * nb + i] = 1;
                    }
                }
            }
            
            auto t_start = std::chrono::high_resolution_clock::now();
            acorn_index.search(
                cur_batch,
                xq + q_offset * dim,
                k,
                dists.data() + q_offset * k,
                results.data() + q_offset * k,
                batch_filter_map.data()
            );
            auto t_end = std::chrono::high_resolution_clock::now();
            total_search_time += std::chrono::duration<double>(t_end - t_start).count();
        }
        
        double recall = evaluate_recall(gt_conjunction, results, nq, k) * 100.0;
        double qps = nq / total_search_time;
        double lat = (total_search_time / nq) * 1000.0;
        
        all_results.push_back({"Two-label conjunction", efs, nq, recall, qps, lat});
        std::cout << " efSearch=" << std::setw(3) << efs << " | Recall@10=" << std::fixed << std::setprecision(2) << std::setw(6) << recall << "% | QPS=" << std::setw(8) << std::fixed << std::setprecision(1) << qps << " | Lat=" << std::fixed << std::setprecision(3) << lat << " ms" << std::endl;
    }
    
    // Print Final Clean Table matching Ali's Screenshot 2 Format
    std::cout << "\n\n========================================================================================" << std::endl;
    std::cout << " FINAL RESULTS TABLE (Exact Screenshot 2 Format)" << std::endl;
    std::cout << "========================================================================================" << std::endl;
    std::cout << "| Workload               | efSearch | Queries | Recall@10 | QPS         | Latency (ms) |" << std::endl;
    std::cout << "|------------------------|----------|---------|-----------|-------------|--------------|" << std::endl;
    for (const auto& r : all_results) {
        std::stringstream ss_q;
        ss_q << r.queries;
        std::string q_str = (r.queries == 10000) ? "10,000" : ss_q.str();
        
        std::cout << "| " << std::left << std::setw(22) << r.workload << " | "
                  << std::right << std::setw(8) << r.efSearch << " | "
                  << std::setw(7) << q_str << " | "
                  << std::setw(8) << std::fixed << std::setprecision(2) << r.recall10 << "% | "
                  << std::setw(11) << std::fixed << std::setprecision(2) << r.qps << " | "
                  << std::setw(12) << std::fixed << std::setprecision(3) << r.latency_ms << " |" << std::endl;
    }
    std::cout << "========================================================================================" << std::endl;
    
    // Save results to CSV file
    std::string csv_path = find_output_path("efsearch_sweep_results.csv");
    std::ofstream out_csv(csv_path);
    if (out_csv.is_open()) {
        out_csv << "Workload,efSearch,Queries,Recall@10,QPS,Latency_ms\n";
        for (const auto& r : all_results) {
            out_csv << r.workload << "," << r.efSearch << "," << r.queries << "," << r.recall10 << "," << r.qps << "," << r.latency_ms << "\n";
        }
        out_csv.close();
        std::cout << "\n Saved results to " << csv_path << std::endl;
    }
    
    delete[] xb;
    delete[] xq;
    return 0;
}
