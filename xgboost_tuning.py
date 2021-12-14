from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

from utils import generate_dmatrix

DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'eval_metric': 'logloss',
}
SEED = 42


def phase_1(data: dict, lr_list, njobs: int):
    """Adjust learning rate and number of trees."""
    # Determine initial value for number of trees
    train_x = data['train_x']
    train_y = data['train_y']
    test_x = data['test_x']
    test_y = data['test_y']
    test_x, val_x, test_y, val_y = train_test_split(
        test_x, test_y, test_size=1 / 3, stratify=test_y, random_state=SEED
    )
    results = {}
    for lr in lr_list:
        # Get base number of trees from xgb native CV
        cur_results = {}
        model = XGBClassifier(
            **DEFAULT_PARAMS,
            learning_rate=lr,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            nthread=njobs,
            scale_pos_weight=1,
            seed=SEED
        )
        dmatrix = generate_dmatrix(train_x, train_y)
        params = model.get_params()
        nbr = params['n_estimators']
        cv_results = xgb.cv(
            params,
            dmatrix,
            num_boost_round=nbr,
            metrics='logloss',
            early_stopping_rounds=50,
            nfold=5,
            verbose_eval=True
        )
        cur_results['cv_results'] = cv_results
        n_est = cv_results.shape[0]
        # Evaluate using these parameters
        model.set_params(
            n_estimators=n_est,
        )
        model.fit(
            train_x, train_y,
            eval_set=[(train_x, train_y), (val_x, val_y)],
            eval_metric=['error', 'logloss']
        )
        pred = model.predict(test_x)
        report = classification_report(test_y, pred, output_dict=True)
        cur_results['report'] = report
        results[lr] = cur_results

    return results
