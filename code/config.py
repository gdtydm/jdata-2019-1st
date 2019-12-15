
import os

feature_path = "../input/feature/"
base_file_path = "../input/"

model_path = "../input/result/model/"
feature_improtant_path = "../input/result/feature_important/"
result_path = "../input/result/"
cache_path = "../input/cache/"

for path in [model_path, feature_path, model_path, feature_improtant_path, result_path, cache_path]:
    if not os.path.exists(path):
        os.makedirs(path)
