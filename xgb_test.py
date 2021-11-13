import numpy as np
import xgboost as xgb

from constants import MODEL_PARAMS_FOLDER
from template_test import get_param_grid, find_best_params, main_base


def xgb_demo():
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split
    cancer = load_breast_cancer()
    x = cancer.data
    y = cancer.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,
                              use_label_encoder=False)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    return model


def get_xgb_param_grid():
    return {}


def find_best_xgb_params(train_x: np.ndarray, train_y: np.ndarray,
                         dataset_name: str, partition: int, folder_name: str,
                         pair_method: str, n_jobs: int):
    return {}


def main(find_params=True, n_jobs=-1):
    main_base(find_params=find_params,
              out_params_name=MODEL_PARAMS_FOLDER + '/xgb_params',
              find_params_fn=find_best_xgb_params,
              out_results_name='xgb_results',
              clasif_fn=xgb.XGBClassifier,
              use_std_masks=False,
              n_cmim=0,
              n_jobs=n_jobs)

