import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle

features_df = pd.read_csv('features.csv')
stores_df = pd.read_csv('stores.csv')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
#print(test_df.head())

#print(features_df.isnull().sum())
#print(stores_df.isnull().sum())
#print(train_df.isnull().sum())
#print(test_df.isnull().sum())

#print(features_df['Store'].isin(stores_df['Store']).value_counts())

features_stores_df = pd.merge(features_df, stores_df, on='Store', how='left')
final_train_df = pd.merge(train_df, features_stores_df, on=['Store', 'Date', 'IsHoliday'], how='left')
final_test_df = pd.merge(test_df, features_stores_df, on=['Store', 'Date', 'IsHoliday'], how='left')
#print(final_train_df.columns.tolist())
#print(len(final_train_df))
#print(final_train_df.isnull().sum())

#print(final_train_df.dtypes.value_counts())
#print(final_train_df['Temperature'].isnull().sum())
#print(final_train_df['Temperature'].isna().sum())


# Replacing null markdowns with 0s.
final_train_df['MarkDown1'] = final_train_df['MarkDown1'].fillna(0)
final_train_df['MarkDown2'] = final_train_df['MarkDown2'].fillna(0)
final_train_df['MarkDown3'] = final_train_df['MarkDown3'].fillna(0)
final_train_df['MarkDown4'] = final_train_df['MarkDown4'].fillna(0)
final_train_df['MarkDown5'] = final_train_df['MarkDown5'].fillna(0)
# Applying same on test dataframe.
final_test_df['MarkDown1'] = final_test_df['MarkDown1'].fillna(0)
final_test_df['MarkDown2'] = final_test_df['MarkDown2'].fillna(0)
final_test_df['MarkDown3'] = final_test_df['MarkDown3'].fillna(0)
final_test_df['MarkDown4'] = final_test_df['MarkDown4'].fillna(0)
final_test_df['MarkDown5'] = final_test_df['MarkDown5'].fillna(0)

# Replace null values in CPI and Unemployment columns with their average values.
final_train_df['CPI'] = final_train_df['CPI'].fillna(final_train_df['CPI'].mean())
final_train_df['Unemployment'] = final_train_df['Unemployment'].fillna(final_train_df['Unemployment'].mean())
# Applying same on test dataframe.
final_test_df['CPI'] = final_test_df['CPI'].fillna(final_test_df['CPI'].mean())
final_test_df['Unemployment'] = final_test_df['Unemployment'].fillna(final_test_df['Unemployment'].mean())

#print(final_train_df.isnull().sum())
# Replace IsHoliday and Type columns into numeric.
final_train_df['IsHoliday'] = final_train_df['IsHoliday'].replace({True: 1, False: 0})
final_train_df['Type'] = final_train_df['Type'].replace({'A': 1, 'B': 2, 'C': 3})
# Applying same on test dataframe.
final_test_df['IsHoliday'] = final_test_df['IsHoliday'].replace({True: 1, False: 0})
final_test_df['Type'] = final_test_df['Type'].replace({'A': 1, 'B': 2, 'C': 3})
#print(final_train_df.dtypes)

final_train_df['Date'] = pd.to_datetime(final_train_df['Date'])
final_train_df['Date'] = final_train_df['Date'].values.astype(float)
final_test_df['Date'] = pd.to_datetime(final_test_df['Date'])
final_test_df['Date'] = final_test_df['Date'].values.astype(float)

# Plot charts
"""
store_types = stores_df['Type'].value_counts()
#print(store_types)

labels = 'A store','B store','C store'
sizes = [(22/(45))*100,(17/(45))*100,(6/(45))*100]
plt.pie(sizes, labels=labels, startangle=90)
plt.show()

# boxplot for sizes of types of stores
store_type_size = pd.concat([final_train_df['Type'], final_train_df['Size']], axis=1)
sns.boxplot(x='Type', y='Size', data=store_type_size, showfliers=False)
plt.show()

# boxplot for weekly sales for store type.
store_type_sales = pd.concat([final_train_df['Type'], final_train_df['Weekly_Sales']], axis=1)
sns.boxplot(x='Type', y='Weekly_Sales', data=store_type_sales, showfliers=False)
plt.show()

#boxplot for sales on holidays and non-holidays.
holiday_sales = pd.concat([final_train_df['IsHoliday'], final_train_df['Weekly_Sales']], axis=1)
sns.boxplot(x='IsHoliday', y='Weekly_Sales', data=holiday_sales, showfliers=False)
plt.show()

# Plotting correlation between all important features
corr = final_train_df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True)
plt.show()
"""
Y = final_train_df['Weekly_Sales'].values
X = final_train_df.drop(columns=['Weekly_Sales']).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=10)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
test_data = scaler.fit_transform(final_test_df)

'''
# Applying KNN Regressor
knn = KNeighborsRegressor(n_neighbors=20)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
#print(Y_train)
#print(Y_pred)
#plt.scatter(Y_test, Y_pred)
#plt.show()
print("KNN MSE: ", metrics.mean_squared_error(Y_test, Y_pred))
print("KNN Accuracy: ", knn.score(X_test, Y_test))
'''

dt = DecisionTreeRegressor(min_samples_leaf=5, random_state=10)
dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)
print("DT MSE: ", metrics.mean_squared_error(Y_test, Y_pred))
print("DT Accuracy: ", dt.score(X_test, Y_test))
# Save the trained model to the disk.
filename = 'dt_trained_model.sav'
pickle.dump(dt, open(filename, 'wb'))

# Load the saved model from disk.
loaded_model = pickle.load(open(filename, 'rb'))
test_pred = loaded_model.predict(test_data)
print("Test pred:", test_pred)

'''

#max_features hyperparameter is set to defaut value of 1, making it same as bagged regressor.
rf = RandomForestRegressor(n_estimators=400)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
print("RF MSE: ", metrics.mean_squared_error(Y_test, Y_pred))
print("RF Accuracy: ", rf.score(X_test, Y_test))


svr = SVR()
svr.fit(X_train, Y_train)
Y_pred = svr.predict((X_test))
print("SVR MSE: ", metrics.mean_squared_error(Y_test, Y_pred))
print("SVR Accuracy: ", svr.score(X_test, Y_test))


nn = MLPRegressor(max_iter= 500, early_stopping=True, random_state=10)
nn.fit(X_train, Y_train)
Y_pred = nn.predict((X_test))
print("NN MSE: ", metrics.mean_squared_error(Y_test, Y_pred))
print("NN Accuracy: %.3f", nn.score(X_test, Y_test))


## Applying grid search to get best hyperparameter values.

hidden_layer_size_range = ((50,), (10, 10), (20, 20, 20), (30,30,30,30))
max_iter_range = range(300, 500, 100)
alpha_values = [0.0001, 0.005, 0.05]
learning_rate_init_values= [0.001, 0.002, 0.004, 0.006, 0.008, 0.01]
solver_options = ['adam', 'sgd']
param_grid = dict(hidden_layer_sizes=hidden_layer_size_range, alpha=alpha_values, learning_rate_init=learning_rate_init_values, solver=solver_options, max_iter=max_iter_range)
nn = MLPRegressor(early_stopping=True, random_state=10)
kfold = KFold(n_splits=10, random_state=1, shuffle=True)
grid = GridSearchCV(nn, param_grid, cv=kfold, scoring='accuracy')
grid.fit(X_train, Y_train)
print('Best params:', grid.best_params_)
print('Best score', grid.best_score_)
best_model = grid.best_estimator_
Y_pred = best_model.predict(X_test)

print("NN MSE: ", metrics.mean_squared_error(Y_test, Y_pred))
print("NN Accuracy: %.3f", best_model.score(X_test, Y_test))
'''



