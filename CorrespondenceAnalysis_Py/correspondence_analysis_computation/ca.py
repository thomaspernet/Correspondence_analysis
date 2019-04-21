import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class compute_ca:
	def __init__(self, df):
		self.df = df

	def correspondance_analysis(df):
		"""
		Compute dimensions 1 and 2 for
		- rows
		- columns

		The normalization is only applied to rows
		return
		pc_rows: dim 1 & 2 for rows
		pc_column: dim 1 & 2 for columns
		variance explained
		eigenvalues
		"""

		### Step 1: Compute observed proportion
		P = np.array(df / df.values.sum())

		### Step 2: computes masses
		column_masses = P.sum(axis=0)
		row_masses = P.sum(axis=1)
		E = np.outer(row_masses, column_masses)
		### Step 3: Compute residuals
		R = P - E

		### Step 4: Compute indexed Residuals
		I = R / E

		### Step 5: Reconstituting endexed residual
		Z = I * np.sqrt(E)

		#### SVD
		u, d, v  = np.linalg.svd(Z, full_matrices=False)
		v = v.T

		#### Eigen value
		eigenvalues = np.power(d, 2)

		#### Compute variance explained
		variance_explained = eigenvalues / eigenvalues.sum()

		### Step 7: Standard coordinate
		#### Only rows normaization for now

		##### Rows
		size = len(row_masses)
		row_masses_t = row_masses.reshape((size,1))
		std_rows_coordinate = np.divide(u, np.sqrt(row_masses_t))

		##### Columns
		size = len(column_masses)
		column_masses_t = column_masses.reshape((size,1))
		std_columns_coordinate = np.divide(v, np.sqrt(column_masses_t))
		principal_coordinate_rows = std_rows_coordinate * d
		principal_coordinate_columns = std_columns_coordinate * d

		##### To DF
		df_coordinate_r = pd.DataFrame(principal_coordinate_rows[:, :-1],
							columns = ['dim_1', 'dim_2'],
							index = df.index)

		df_coordinate_c = pd.DataFrame(std_columns_coordinate[:, :-1],
							columns = ['dim_1', 'dim_2'],
							index = list(df))

		dic_results = {
			'pc_rows': df_coordinate_r,
			'pc_columns': df_coordinate_c,
			'variance_explained': variance_explained,
			'eigenvalues': eigenvalues
			}

	return dic_results
