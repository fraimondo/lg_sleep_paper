import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score

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


def eval_bs(clf, X_test, bs_inds0, bs_inds1, y_bs):
    y_pred = clf.predict_proba(X_test[np.r_[bs_inds0, bs_inds1]])[:, 1]
    res = roc_auc_score(y_bs, y_pred)
    return res


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
        clf = DummyClassifier()
    elif model_type == 'dummy-negative':
        clf = DummyClassifier(strategy='constant', constant=0)
    else:
        raise ValueError(f'Uknown model type {model_type}')

    return clf
