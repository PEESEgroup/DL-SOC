import numpy as np
import pandas as pd
import shap
import time
import os
import sys
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_script_dir, '../dl_models/ssl_model'))
from ssl_model import build_model


# Load DL model
w_path = '../dl_models/ssl_model/ckpt/ckpt'
model = ssl_model.build_model()
model.load_weights(w_path)

# Sample background data and implement the explainer
np.random.seed(42)
rand_ind = np.random.choice(X_train.shape[0], 100)
X_train = np.load('../data/patch_data/X_train')
explainer = shap.DeepExplainer(model, [X_train[rand_ind], depth_train[rand_ind]])

# Run the explainer
start = time.time()
shap_values = explainer.shap_values([X_test, depth_test], check_additivity=False)
print("Time elapsed: ", time.time()-start)

# Process the results
shap_sum = np.sum(shap_values[0][0], axis=(1, 2))
shap_avg = np.mean(np.abs(shap_sum), axis=0)
df = pd.DataFrame(average_impact, columns=["Average Contribution"])
# df.to_csv('./covariate_shap.csv')