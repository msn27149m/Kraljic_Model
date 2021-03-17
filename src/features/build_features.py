from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelBinarizer, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, precision_recall_curve, make_scorer, accuracy_score, precision_score, recall_score, classification_report, f1_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV




class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            return X[self.columns]
        except:
            print("Could not find selected columns {:s} in available columns {:s}".format(
                self.columns, X.columns))
            raise


class StringIndexer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dictionaries = dict()
        self.columns = list()

    def fit(self, X, y=None):
        self.columns = X.columns.values
        for col in self.columns:
            categories = np.unique(X[col])
            self.dictionaries[col] = dict(
                zip(categories, range(len(categories))))
        return self

    def transform(self, X):
        column_array = []
        for col in self.columns:
            dictionary = self.dictionaries[col]
            na_value = len(dictionary) + 1
            transformed_column = X[col].apply(
                lambda x: dictionary.get(x, na_value))
            column_array.append(transformed_column.values.reshape(-1, 1))
        return np.hstack(column_array)


class FeatureRatioCal(BaseEstimator, TransformerMixin):
    def __init__(self, columns):  # no *args or **kargs
        self.columns = columns

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        ratio = X[self.columns[0]] / (X[self.columns[1]] + 1e-6)
        ratio_name = '_'.join([self.columns[0], self.columns[1], 'ratio'])
        X.loc[:, ratio_name] = ratio
        return X


class FeatureDiffCal(BaseEstimator, TransformerMixin):
    def __init__(self, columns):  # no *args or **kargs
        self.columns = columns

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        ratio = X[self.columns[0]] - X[self.columns[1]]
        ratio_name = '_'.join([self.columns[0], self.columns[1], 'sub'])
        X.loc[:, ratio_name] = ratio
        return X

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown



if __name__ == '__main__':
    df = pd.read_excel('../../data/processed/jj_all_df_mgc_class155.xlsx', sheet_name='Sheet1', header=0)
    raw_feats_cols = df.columns.tolist()
    raw_feats_drop = [
        '材料组中文名称',
        '材料组CODE',
        'material_name',
        'material_group_code',
        'inbound_LAH',
        'inbound_JIT',
        'inbound_VLB',
        'inbound_JIS',
        'p',
        'amount_class',
        'empolder_price_score',
        'single_sourcing_score',
        'part_num_score',
        'acceptance_price_score',
        'bmg_percent_score',
        'leadtime_score',
        'production_time_score',
        'zero_mile_quantity_score',
        'bka_red_time_score',
        'bka_complete_percent_score',
        'supplier_cooperation_days_score',
        'logistics_distance_score',
        'logistics_duration_score',
        'supplier_direct_num_score',
        'supplier_logistics_score_score',
        'supplier_var_score',
        'supplier_num_score',
        'bidding_num_score',
        'quality_ok_num_score',
        'tech_ok_num_score',
        'risk_score',
        'risk_grade',
        'material_group_amount_score',
        'material_group_amount_percent_score',
        'price_reduction_total_score',
        'price_reduction_percent_score',
        'pca_score',
        'pca_percent_score',
        'tia_value_score',
        'optimize_amount_score',
        'over_target_price_score',
        'money_score',
        'single_sourcing_score',
        '材料组中文名称',
        '科室_y'
        ]
    raw_feats_x_num = [
        'material_group_amount',
        'bka_red_time',
        'production_time',
        'zero_mile_quantity',
        'tia_value',
        'production_risk_times',
        'pca',
        'pca_percent',
        'supplier_var',
        'part_num',
        'material_group_amount_percent',
        'empolder_price',
        'acceptance_price',
        'bmg_percent', 'single_sourcing',
        'bka_complete_percent',
        'supplier_cooperation_days',
        'supplier_direct_num',
        'supplier_num',
        'tech_ok_num',
        'quality_ok_num',
        'optimize_amount',
        'bidding_num',
        'over_target_price',
        'logistics_distance',
        'logistics_duration',
        'inbound_score',
        'supplier_logistics_score',
        'price_reduction_total',
        'price_reduction_percent',
        'mbdl_num',
        'frm_score',
        'frm_svw_to_percent'
    ]
    raw_feats_x_cat = ['科室_x']

    raw_feats_y_cat = ['risk类型']

    # Split data into train and test data
    seed = 123456

    df.drop(columns=raw_feats_drop, axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['risk类型', 'money类型']),
                                                        df['risk类型'],
                                                        test_size=0.3,
                                                        stratify=df['risk类型'], random_state=42)

    # feat_cat_transform = pipe_feat_cat_proc.fit_transform(df)
    # Categorical feature processing definition
    poly_features = PolynomialFeatures(degree=2, include_bias=True)

    pipe_feat_cat_x_process = Pipeline(steps=[
        ('cat_feature_selection', ColumnSelector(raw_feats_x_cat)),
        ('cat_feature_procss', StringIndexer())]
    )

    # Numerical Feature Processing Definition
    pipe_feat_num_x_process = Pipeline(steps=[
        ('cat_feature_selection', ColumnSelector(raw_feats_x_num)),
        ('cat_feature_fillna', KNNImputer(n_neighbors=3)),
        ('Ploy_feat', poly_features),
        ('cat_feature_scaler', MinMaxScaler())]
    )

    pipe_feat_cat_y_process = Pipeline(steps=[
        ('cat_feature_selection', ColumnSelector(raw_feats_y_cat)),
        ('cat_feature_procss', StringIndexer())]
    )

    #feat_num_transform = pipe_feat_num_proc.fit_transform(df)

    # Fit feature union to training data
    pipe_feats_union = FeatureUnion(transformer_list=[('cat_x', pipe_feat_cat_x_process),
                                                      ('num_x', pipe_feat_num_x_process)])

    #X_train_transform = pipe_feats_union.fit_transform(X_train)
    #y_train_transform = pipe_feat_cat_y_process.fit_transform(y_train)
    lb = LabelBinarizer()
    y_train_transform = lb.fit_transform(y_train)


    #Model
    log_clf = LogisticRegression(penalty='l2', C=0.1, max_iter=10000, random_state=42)
    svm_clf = SVC(gamma="auto", kernel='rbf', degree=3, C=0.01,  coef0=0.0001, probability=True, random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=30, max_features=None, random_state=42)
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3, criterion='entropy', min_samples_split=0.8, random_state=42),
        n_estimators=150, algorithm="SAMME.R", learning_rate=0.1, random_state=42)

    svm_tuned_parameters = [{'model__kernel': ['rbf'], 'model__gamma': [1e-3, 1e-4], 'model__C': [1, 10, 100, 1000],
                             'model__degree':[1,2,3,4,5]},
                            {'model__kernel': ['linear'], 'model__C': [1, 10, 100, 1000]}]


    over = SMOTE(sampling_strategy=0.1)

    pca = PCA(n_components=20, random_state=42)

    pipe_model = Pipeline(steps=[('pipe_feats_union', pipe_feats_union),
                                 ('PCA', pca),
                                 ('model', svm_clf)
                                ])

    #t = pipe_model.fit(X_train, y_train_transform)
    skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_index, test_index in skfolds.split(X_train, y_train_transform):
        X_train_folds = X_train.iloc[train_index]
        y_train_folds = y_train_transform[train_index]
        X_test_folds = X_train.iloc[test_index]
        y_test_folds = y_train_transform[test_index]

        scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
        grid_model = GridSearchCV(pipe_model, param_grid=svm_tuned_parameters, scoring=scoring, refit='AUC', return_train_score=True)
        grid_model.fit(X_train_folds, y_train_folds.ravel())
        y_train_pred = grid_model.predict(X_train_folds)
        print("Precision score of training set is {0}".format(precision_score(y_train_folds, y_train_pred)))
        print("Recall score of training set is {0}".format(recall_score(y_train_folds, y_train_pred)))



        y_test_pred = grid_model.predict(X_test_folds)

        print("Precision score of test set is {0}".format(precision_score(y_test_folds, y_test_pred)))
        print("Recall score of test set is {0}".format(recall_score(y_test_folds, y_test_pred)))
        print(classification_report(y_test_folds, y_test_pred))


        def to_labels(pos_probs, threshold):
            return (pos_probs >= threshold).astype('int')

        y_test_probs = grid_model.predict_proba(X_test_folds)[:,0]
        thresholds = np.arange(0, 1, 0.001)
        scores = [f1_score(y_test_folds, to_labels(y_test_probs, t), zero_division=1, average='weighted') for t in thresholds]
        ix = np.argmax(scores)
        print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

        threshold_best = thresholds[ix]
        y_test_thresh_pred = np.where(y_test_probs>threshold_best, 1, 0)
        print("Precision score of changed threshold for test set is {0}".format(precision_score(y_test_folds, y_test_thresh_pred)))
        print("Recall score of  changed threshold for test set is {0}".format(recall_score(y_test_folds, y_test_thresh_pred)))
        print('Best Parameters : '.format(grid_model.best_params_))
        print("Done")
