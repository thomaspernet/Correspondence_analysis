import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class compute_ca:
	def __init__(self, df):
		self.df = df

	def correspondance_analysis(self):
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
		P = np.array(self.df / self.df.values.sum())

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
		df_coordinate_r = pd.DataFrame(principal_coordinate_rows[:, :2],
							columns = ['dim_1', 'dim_2'],
							index = self.df.index)

		df_coordinate_c = pd.DataFrame(std_columns_coordinate[:, :2],
							columns = ['dim_1', 'dim_2'],
							index = list(self.df))
		df_observed_proportion = pd.DataFrame(
							P,
							columns  = list(self.df),
							index = self.df.index)

		dic_results = {
			'observed_proportion': df_observed_proportion,
			'pc_rows': df_coordinate_r,
			'pc_columns': df_coordinate_c,
			'variance_explained': variance_explained,
			'eigenvalues': eigenvalues
			}

		return dic_results

def normalize(x, y, scaler = 0.8, option = 'column'):
		"""
		objective:  Get the coordinates on the cirle line
		Euclidean distance is square root of sum of x square
		and y square
		We know that the euclidean distance should be equal to
		the radius (ie 1) to be exactly on the circle
		therefore, we need to find alpha to shrink our vector

		For the column, rescale is alpha * distance
		for the row, need to rescale to fit the circle
		take the minimum alpha (ie largest distance) and apply a
		scaler. If scaler is 1, the largest row's distance will
		be on the circle line
		"""
		distance = np.sqrt(np.power(x,2) + np.power(y,2))
		alpha = 1 / distance
		if option == 'column':
			x_shrink = x * alpha
			y_shrink = y * alpha
			dic_df = {
				'x': x,
				'y': y,
				'x_shrink': x_shrink,
				'y_shrink': y_shrink,
				'distance': distance,
				'alpha': alpha
			}
		else:
			min_alpha = min(alpha)
			x_shrink = x * min_alpha * scaler
			y_shrink = y * min_alpha * scaler
			dic_df = {
				'x': x,
				'y': y,
				'x_shrink': x_shrink,
				'y_shrink': y_shrink,
				'distance': distance,
				'alpha': alpha
				}
		df_shrink = pd.DataFrame(dic_df)
		return df_shrink

def row_principal_coordinates(df_x, df_y, variance_explained):
	"""
	plot principal coordinates
		"""
	dic_df = {
			'rows': [
				df_x.index,
				df_x['dim_1'],
				df_x['dim_2']],
			'columns': [
				df_y.index,
				df_y['dim_1'],
				df_y['dim_2']]
				}

	fig, ax = plt.subplots(figsize=(10, 10))

	ax.scatter(df_x['dim_1'],
			 df_x['dim_2'], label='Rows')
	ax.scatter(df_y['dim_1'],
			 df_y['dim_2'], label='Columns')

	for x, name in dic_df.items():
		for i, label in enumerate(name[0]):
			ax.text(name[1][i]+0.12,
				name[2][i]+0.12,
				label,
				color='black',
				ha='center',
				va='center',
				fontsize=10)
	ax.set_xlim(-2, 2)
	ax.axhline(y=0, color='k', linewidth=1, linestyle = "--")
	ax.axvline(x=0, color='k', linewidth=1, linestyle = "--")
	ax.set_xlim(-2, 2)
	ax.legend(loc='upper center', bbox_to_anchor=(1.1, 0.8), shadow=True, ncol=1)
	plt.title('Correspondence analysis: Row Principal Coordinates')

	# Add the axis labels
	plt.xlabel('Dimension 1 (%.2f%%)' % (variance_explained[0]*100))
	plt.ylabel('Dimension 2 (%.2f%%)' % (variance_explained[1]*100))
	plt.show()

def row_focus_coordinates(df_x, df_y, variance_explained, export_data = False):
	"""
	plot the columns on the circle and rescale
	the rows
	"""
	row_norm = normalize(df_x['dim_1'],
				   df_x['dim_2'],
				   scaler = 0.8,
				   option  = 'row')
	col_norm = normalize(df_y['dim_1'],
				   df_y['dim_2'])

	dic_df_rescale = {
		'rows_rescale': [row_norm.index, row_norm['x_shrink'],
			row_norm['y_shrink']],
		'columns_rescale': [col_norm.index, col_norm['x_shrink'],
			col_norm['y_shrink']]
			}

	fig, ax = plt.subplots(figsize=(8, 8))
	an = np.linspace(0, 2 * np.pi, 1000)

	ax.scatter(row_norm['x_shrink'],
	   row_norm['y_shrink'])
	ax.scatter(col_norm['x_shrink'],
	   col_norm['y_shrink'])

	for i, name in dic_df_rescale.items():
		for i, label in enumerate(name[0]):
			ax.text(name[1][i]+0.12,
		 name[2][i]+0.12,
		 label,
		 color='black',
		 ha='center',
		 va='center',
		 fontsize=10)
	ax.axhline(y=0, color='k', linewidth=1, linestyle = "--")
	ax.axvline(x=0, color='k', linewidth=1, linestyle = "--")
	plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
	plt.axis('equal')
	ax.set_title('Variable factor map')
	# Add the axis labels
	plt.xlabel('Dimension 1 (%.2f%%)' % (variance_explained[0]*100))
	plt.ylabel('Dimension 2 (%.2f%%)' % (variance_explained[1]*100))
	plt.show()
	dic_data = {
		'x_circle': np.cos(an),
		'y_circle': np.sin(an),
		'row_norm':row_norm,
		'col_norm':col_norm,
	}
	if export_data == True:
		return dic_data
