total_number_of_features = 47
# balancing_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# balancing_factors = [0.1, 0.5, 0.9]
balancing_factors = [0.1]
# n = [10, 15, 20, 25, 30, 35, 40, 45]
# n = [10, 15, 20]
n = [10]
# methods = ["msd", "mpt", "mmr", "gas", "topk"]
# methods = ["msd", "mpt"]
methods = ["msd"]
measure = "ndcg"  # ndcg or map
training_file = '../../../pre_processing/resources/svmcache/lambdamart2020_feature_selection_one_indexed_train_90.densesvm'
test_file = '../../../pre_processing/resources/svmcache/lambdamart2020_feature_selection_one_indexed_train_-10.densesvm'
