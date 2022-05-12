total_number_of_features = 47
# balancing_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
balancing_factors = [0.2]
n = [10, 15, 20, 25, 30, 35, 40, 45]
# methods = ["msd", "mpt", "mmr", "gas", "topk"]
methods = ["msd", "mpt", "mmr", "topk"]
# methods = ["mmr"]
measure = "ndcg"  # ndcg or map
