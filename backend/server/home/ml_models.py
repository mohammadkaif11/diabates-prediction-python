import joblib
import os

# Path to the current file
current_file_path = os.path.abspath(__file__)
# Move up three levels from the current file path to reach the project root
BASE_DIR = os.path.abspath(os.path.join(current_file_path, '..', '..', '..', '..'))

# Construct the paths to the model files
random_forest_model_path = os.path.join(BASE_DIR, 'ml-model', 'random_forest.joblib')
extra_trees_model_path = os.path.join(BASE_DIR, 'ml-model', 'extra_trees.joblib')

# Loading the models
random_forest_model = joblib.load(random_forest_model_path)
extra_trees_model = joblib.load(extra_trees_model_path)
