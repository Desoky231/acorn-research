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
#include <iomanip>
#include <random>
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

// Exact Brute Force Ground Truth Calculator using bool match vector
std::vector<std::vector<faiss::idx_t>> compute_ground_truth(
    const float* xb, size_t nb, size_t dim,
    const float* xq, size_t nq,
    const std::vector<char>& matches_both_tags,
    int k
) {
    std::vector<std::vector<faiss::idx_t>> gt(nq);
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t q = 0; q < nq; q++) {
        const float* query_vec = xq + q * dim;
        std::vector<std::pair<float, faiss::idx_t>> candidates;
        
        for (size_t i = 0; i < nb; i++) {
            if (matches_both_tags[i] == 0) continue;
            
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

struct CorrelationRegime {
    std::string name;
    double p_A;
    double p_B_given_A;
    double p_B_given_not_A;
    std::string description;
};

struct BenchmarkRow {
    std::string regime;
    int efSearch;
    size_t matching_items;
    double lift;
    double recall10;
    double qps;
    double latency_ms;
};

int main(int argc, char* argv[]) {
    size_t nb_limit = (argc > 1) ? std::stoul(argv[1]) : 1000000;
    size_t nq_limit = (argc > 2) ? std::stoul(argv[2]) : 1000;
    int gamma = (argc > 3) ? std::stoi(argv[3]) : 12;
    int M = (argc > 4) ? std::stoi(argv[4]) : 32;
    int M_beta = (argc > 5) ? std::stoi(argv[5]) : 32;
    int k = 10;
    
    std::cout << "==========================================================================" << std::endl;
    std::cout << " ACORN: 3-Dataset Tag Correlation Benchmark (Positive / Neutral / Negative) " << std::endl;
    std::cout << "==========================================================================" << std::endl;
    std::cout << " Configuration:" << std::endl;
    std::cout << " - Vectors to index (N): " << nb_limit << std::endl;
    std::cout << " - Conjunction queries:  " << nq_limit << " (Tag A AND Tag B)" << std::endl;
    std::cout << " - Gamma expansion:      " << gamma << std::endl;
    std::cout << " - Graph degree (M):     " << M << std::endl;
    std::cout << " - M_beta:               " << M_beta << std::endl;
    std::cout << " - Top-k:                " << k << std::endl;
    std::cout << "==========================================================================" << std::endl;
    
    // 1. Load Base & Query Vectors
    std::cout << "\n[1/4] Loading SIFT1M dataset..." << std::endl;
    size_t nb = 0, dim = 0, nq = 0, qdim = 0;
    float* xb = read_fbin("base.fbin", nb, dim, nb_limit);
    float* xq = read_fbin("query.fbin", nq, qdim, nq_limit);
    
    // 2. Define the 3 Correlation Regimes
    std::vector<CorrelationRegime> regimes = {
        {
            "Positive Correlation",
            0.10, // P(A) = 10%
            0.80, // P(B|A) = 80%
            0.02, // P(B|not A) = 2%
            "Strong Positive Co-occurrence (Lift ~ 8.0)"
        },
        {
            "Neutral / Independent",
            0.10, // P(A) = 10%
            0.10, // P(B|A) = 10% (Independent coin flip)
            0.10, // P(B|not A) = 10%
            "Independent Random Chance (Lift = 1.0)"
        },
        {
            "Negative Correlation",
            0.10, // P(A) = 10%
            0.005, // P(B|A) = 0.5% (Very rare co-occurrence)
            0.11,  // P(B|not A) = 11%
            "Inversely Correlated / Isolated Islands (Lift ~ 0.05)"
        }
    };
    
    std::vector<int> ef_values = {32, 40, 64, 100, 200, 256, 512};
    std::vector<BenchmarkRow> all_results;
    
    // 3. Execute Benchmark for Each Correlation Regime
    for (size_t r_idx = 0; r_idx < regimes.size(); r_idx++) {
        const auto& reg = regimes[r_idx];
        std::cout << "\n==========================================================================" << std::endl;
        std::cout << " [" << (r_idx + 1) << "/3] REGIME: " << reg.name << " (" << reg.description << ")" << std::endl;
        std::cout << "==========================================================================" << std::endl;
        
        // Generate Synthetic Tag Assignments
        std::mt19937 rng(42 + (int)r_idx * 100);
        std::uniform_real_distribution<double> udist(0.0, 1.0);
        
        std::vector<int> metadata(nb, 0);
        std::vector<char> matches_both(nb, 0);
        size_t count_A = 0, count_B = 0, count_AB = 0;
        
        for (size_t i = 0; i < nb; i++) {
            bool has_A = (udist(rng) < reg.p_A);
            double prob_B = has_A ? reg.p_B_given_A : reg.p_B_given_not_A;
            bool has_B = (udist(rng) < prob_B);
            
            if (has_A) count_A++;
            if (has_B) count_B++;
            if (has_A && has_B) {
                count_AB++;
                matches_both[i] = 1;
            }
            
            // Encode in 32-bit bitmask: bit 0 = Tag A, bit 1 = Tag B
            int mask = 0;
            if (has_A) mask |= (1 << 0);
            if (has_B) mask |= (1 << 1);
            metadata[i] = mask;
        }
        
        // Ensure at least 10 items match
        if (count_AB < (size_t)k) {
            for (size_t i = 0; i < (size_t)k; i++) {
                if (!matches_both[i]) {
                    matches_both[i] = 1;
                    metadata[i] |= (1 << 0) | (1 << 1);
                    count_AB++;
                }
            }
        }
        
        double p_A_actual = (double)count_A / nb;
        double p_B_actual = (double)count_B / nb;
        double p_AB_actual = (double)count_AB / nb;
        double lift = (p_A_actual * p_B_actual > 0) ? (p_AB_actual / (p_A_actual * p_B_actual)) : 0.0;
        
        std::cout << " -> Tag A items:       " << count_A << " (" << std::fixed << std::setprecision(2) << (p_A_actual * 100.0) << "%)" << std::endl;
        std::cout << " -> Tag B items:       " << count_B << " (" << std::fixed << std::setprecision(2) << (p_B_actual * 100.0) << "%)" << std::endl;
        std::cout << " -> Matches (Tag A&B): " << count_AB << " (" << std::fixed << std::setprecision(3) << (p_AB_actual * 100.0) << "%)" << std::endl;
        std::cout << " -> Actual Lift Ratio: " << std::fixed << std::setprecision(2) << lift << std::endl;
        
        // Compute Exact Ground Truth
        std::cout << "\n -> Computing exact ground truth for " << nq << " queries..." << std::endl;
        auto t_gt_start = std::chrono::high_resolution_clock::now();
        auto ground_truth = compute_ground_truth(xb, nb, dim, xq, nq, matches_both, k);
        auto t_gt_end = std::chrono::high_resolution_clock::now();
        double gt_time = std::chrono::duration<double>(t_gt_end - t_gt_start).count();
        std::cout << " -> Ground truth computed in " << std::fixed << std::setprecision(2) << gt_time << " s" << std::endl;
        
        // Build ACORN Index
        std::cout << " -> Constructing ACORN-gamma index (gamma=" << gamma << ", M=" << M << ", M_beta=" << M_beta << ")..." << std::endl;
        auto t_build_start = std::chrono::high_resolution_clock::now();
        
        faiss::IndexACORNFlat acorn_index(dim, M, gamma, metadata, M_beta);
        acorn_index.acorn.efConstruction = 40;
        acorn_index.add(nb, xb);
        
        auto t_build_end = std::chrono::high_resolution_clock::now();
        double build_time = std::chrono::duration<double>(t_build_end - t_build_start).count();
        std::cout << " -> ACORN-gamma constructed in " << std::fixed << std::setprecision(2) << build_time << " s (TTI)" << std::endl;
        
        // Prepare Filter Map for Queries (all queries require Tag A AND Tag B)
        std::vector<char> filter_map(nq * nb, 0);
        #pragma omp parallel for
        for (size_t q = 0; q < nq; q++) {
            for (size_t i = 0; i < nb; i++) {
                filter_map[q * nb + i] = matches_both[i];
            }
        }
        
        // Sweep efSearch
        std::cout << " -> Running efSearch sweeps..." << std::endl;
        for (int efs : ef_values) {
            acorn_index.acorn.efSearch = efs;
            std::vector<faiss::idx_t> results(nq * k);
            std::vector<float> dists(nq * k);
            
            auto t_search_start = std::chrono::high_resolution_clock::now();
            acorn_index.search(nq, xq, k, dists.data(), results.data(), filter_map.data());
            auto t_search_end = std::chrono::high_resolution_clock::now();
            
            double sec = std::chrono::duration<double>(t_search_end - t_search_start).count();
            double recall = evaluate_recall(ground_truth, results, nq, k) * 100.0;
            double qps = nq / sec;
            double lat = (sec / nq) * 1000.0;
            
            all_results.push_back({reg.name, efs, count_AB, lift, recall, qps, lat});
            std::cout << "    efSearch=" << std::setw(3) << efs << " | Recall@10=" << std::fixed << std::setprecision(2) << std::setw(6) << recall << "% | QPS=" << std::setw(8) << std::fixed << std::setprecision(1) << qps << " | Lat=" << std::fixed << std::setprecision(3) << lat << " ms" << std::endl;
        }
    }
    
    // Print Final Comparative Table
    std::cout << "\n\n====================================================================================================" << std::endl;
    std::cout << " FINAL COMPARATIVE CORRELATION RESULTS TABLE" << std::endl;
    std::cout << "====================================================================================================" << std::endl;
    std::cout << "| Correlation Regime      | efSearch | Matching Items | Lift Ratio | Recall@10 | QPS         | Lat (ms) |" << std::endl;
    std::cout << "|-------------------------|----------|----------------|------------|-----------|-------------|----------|" << std::endl;
    for (const auto& r : all_results) {
        std::cout << "| " << std::left << std::setw(23) << r.regime << " | "
                  << std::right << std::setw(8) << r.efSearch << " | "
                  << std::setw(14) << r.matching_items << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.lift << " | "
                  << std::setw(8) << std::fixed << std::setprecision(2) << r.recall10 << "% | "
                  << std::setw(11) << std::fixed << std::setprecision(1) << r.qps << " | "
                  << std::setw(8) << std::fixed << std::setprecision(3) << r.latency_ms << " |" << std::endl;
    }
    std::cout << "====================================================================================================" << std::endl;
    
    // Save to CSV
    std::string csv_path = find_output_path("correlation_sweep_results.csv");
    std::ofstream out_csv(csv_path);
    if (out_csv.is_open()) {
        out_csv << "Regime,efSearch,MatchingItems,LiftRatio,Recall10,QPS,Latency_ms\n";
        for (const auto& r : all_results) {
            out_csv << r.regime << "," << r.efSearch << "," << r.matching_items << "," << r.lift << "," << r.recall10 << "," << r.qps << "," << r.latency_ms << "\n";
        }
        out_csv.close();
        std::cout << "\n Saved complete comparative results to: " << csv_path << std::endl;
    }
    
    delete[] xb;
    delete[] xq;
    return 0;
}
