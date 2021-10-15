import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             average_precision_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def get_bootstrap(y, n_bootstrap):
    rng = np.random.RandomState(42)
    y0_inds = np.where(y == 0)[0]
    y1_inds = np.where(y == 1)[0]
    for _ in range(n_bootstrap):
        bs_inds0 = rng.choice(y0_inds, len(y0_inds), replace=True)
        bs_inds1 = rng.choice(y1_inds, len(y1_inds), replace=True)
        y_bs = np.r_[np.zeros_like(bs_inds0),
                     np.ones_like(bs_inds1)].astype(int)
        yield bs_inds0, bs_inds1, y_bs


def eval_bs(clf, X_test, bs_inds0, bs_inds1, y_bs, full_scoring=False):
    to_test = X_test[np.r_[bs_inds0, bs_inds1]]
    y_pred_proba = clf.predict_proba(to_test)[:, 1]
    auc = roc_auc_score(y_bs, y_pred_proba)

    precision = -1
    recall = -1
    ap = -1

    if full_scoring:
        y_pred = clf.predict(to_test)
        precision = precision_score(y_bs, y_pred)
        recall = recall_score(y_bs, y_pred)
        ap = average_precision_score(y_bs, y_pred_proba)

    return auc, precision, recall, ap


def get_model(model_type):
    clf = None
    if model_type == 'gssvm':
        # GSSVM with all features
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cost_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        svc_params = dict(kernel='linear', probability=True,
                          random_state=42, class_weight='balanced')
        gc_fit_params = {'C': cost_range}
        svm = SVC(**svc_params)
        gssvm = GridSearchCV(svm, gc_fit_params, cv=skf, scoring='roc_auc')
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('gssvm', gssvm)
        ])
    elif model_type == 'et-reduced':
        # Extra-trees (DOC-forest)
        et_params = dict(n_jobs=-1, n_estimators=2000,
                         max_features=1, max_depth=4,
                         random_state=42, class_weight='balanced',
                         criterion='entropy')
        clf_model = ExtraTreesClassifier(**et_params)
        clf = Pipeline([
            ('scaler', RobustScaler()),
            ('et-reduced', clf_model)])
    elif model_type == 'dummy':
        clf = DummyClassifier(strategy='stratified')
    elif model_type == 'dummy-negative':
        clf = DummyClassifier(strategy='constant', constant=0)
    else:
        raise ValueError(f'Uknown model type {model_type}')

    return clf


def sanitise_cross_validate(result):
    df = pd.DataFrame(result)
    df = df.rename(
        columns={'test_roc_auc': 'AUC',
                 'test_precision': 'Precision',
                 'test_recall': 'Recall',
                 'test_average_precision': 'AP'})
    df.index.name = 'Fold'
    return df


def sanitise_eval_bs(result):
    aucs, precisions, recalls, aps = zip(*result)
    df = pd.DataFrame({
        'AUC': list(aucs),
        'Precision': list(precisions),
        'Recall': list(recalls),
        'AP': list(aps),
        'BS': np.arange(1, len(result) + 1)})

    return df
