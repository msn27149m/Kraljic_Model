
import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
import joblib


if __name__ =='__main__':
	model = joblib.load('../../models/forest_grid_search_20200731093419.pkl')
	df_test1 = pd.DataFrame.from_dict(
		data={'Part_No': [12111], 'material_price': [70], 'total_price': [100], 'amount': [75000]})
	df_test2 = pd.DataFrame.from_dict(
		data={'Part_No': [12111], 'material_price': [23], 'total_price': [33], 'amount': [750000]})
	model.predict_proba(df_test1)
	print('Done')