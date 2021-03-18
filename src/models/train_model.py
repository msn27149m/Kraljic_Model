import os
import sys
import inspect

#current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parent_dir = os.path.dirname(current_dir)
#sys.path.insert(0, parent_dir)

from src.features import build_features as feat_process
import pandas as pd
import joblib
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve




if __name__ == '__main__':


	feat_cat_transform = Pipeline(
		steps=[('Feat_cat_select', feat_process.ColumnSelector(columns=feat_cat)), ('Feat_cat_transform', feat_process.StringIndexer())])

	feat_num_transform = Pipeline(
		steps=[('Feat_num_select', feat_process.ColumnSelector(columns=feat_num)), ('Feat_ratio', feat_process.FeatureRatioCal(['material_price', 'total_price'])), ('Feat_diff', feat_process.FeatureDiffCal(['total_price', 'material_price']))])

	# Define Models
	forest_clf = RandomForestClassifier(n_estimators=100)
	extra_trees_clf = ExtraTreesClassifier(n_estimators=100)
	svm_clf = LinearSVC(random_state=42)
	mlp_clf = MLPClassifier(random_state=42)

	# Define Models' Potential Parameters
	forest_clf_param_grid = {'n_estimators': [3, 10, 30], 'max_features': [2, 4], 'max_depth': [2, 4, 6, 8, 10], 'bootstrap': [False]}

	forest_grid_search = GridSearchCV(forest_clf, forest_clf_param_grid, cv=5, scoring='accuracy',return_train_score=True)

	# Define Model training Pipeline
	model_pipeline = Pipeline([
		('union', FeatureUnion(n_jobs=-1, transformer_list=[
			# ('cat_feat_trans', feat_cat_transform),
			('num_feat_trans', feat_num_transform)
		])),
		('model_training', forest_grid_search)
	])

	#Decsion Tree Model Training
	X = model_pipeline['union'].fit_transform(df)
	y = feat_cat_transform.fit_transform(df)
	model_pipeline.fit(df, y.ravel())
	model = model_pipeline['model_training']

	# MLP Model Training
	MLP_param_grid = {
		'hidden_layer_sizes': [(7, 7), (128,), (128, 7)],
		'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
		'epsilon': [1e-3, 1e-7, 1e-8, 1e-9, 1e-8]
	}
	MLP_Grid = GridSearchCV(
		MLPClassifier(learning_rate='adaptive', learning_rate_init=1., early_stopping=True, shuffle=True),
		param_grid=MLP_param_grid, n_jobs=-1)

	MLP_model_pipeline = Pipeline([
		('union', FeatureUnion(n_jobs=-1, transformer_list=[
			# ('cat_feat_trans', feat_cat_transform),
			('num_feat_trans', feat_num_transform)
		])),
		('model_training', forest_grid_search)
	])
	MLP_model_pipeline.fit(df, y.ravel())

	# Model Evaluation
	X_train, X_test, y_train, y_test = train_test_split(X, y)
	print("The best parameters for random forest model is {}".format(model.best_params_))
	fpr, tpr, thresholds = roc_curve(model.predict(X_test), y_test)
	plot_roc_curve(fpr, tpr)
	precisions, recalls, thresholds = precision_recall_curve(forest_grid_search.predict(X_test), y_test)
	log_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	model_name = 'forest_grid_search' + '_' + log_time + '.pkl'
	joblib.dump(model_pipeline, os.path.join('models', model_name))
	print("Done")

