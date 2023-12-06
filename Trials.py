# %% Importing the necessary library:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import xgboost as xg
from matplotlib.cbook import boxplot_stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# %% Loading the dataset:

data = pd.read_csv("C:/Users/umesh/Downloads/biopsy.csv")

# %% Basic Information of DataSet:

print(data.info())
stats = data.describe()
print(data.isna().sum())

# %% Total Number of ? per column:

# Can be done with all column at once:
print(data.isin(['?']).sum())

# %% Dropping the values which have more ? if values:

print(data.shape)
data.drop(data.columns[data.isin(['?']).sum() > ((25/100)*(data.count()))], inplace=True, axis=1)
print(data.isin(['?']).sum())
print(data.shape)

# %% Replacing the ? value with the needed Value:
# Replacing the ? to Nan value:

data.replace('?', np.nan, inplace=True)
impute_data = data.isna().sum()

# %% Selecting Columns with only Nan:

impute_data = impute_data[impute_data != 0]
print(impute_data)

# %% Removing all null values and replacing it with mean:

if len(impute_data) > 0:
    data[impute_data.index] = SimpleImputer(strategy="median").fit_transform(data[impute_data.index])
print(data.info())

# %% Converting all data types as Float:

data = data.astype(float)
print(data.info())
stats = data.describe()
print(data.columns)

# %% Finding out the classification classes:

category = data.columns[data.isin([0, 1]).all()]
print(category)
data_category = data[category]
print(data_category.nunique())

# %% Getting the column names:

data_columns = np.array(data.columns)
print(data_columns)

# %% Splitting the data into Dependent and Independent:

X = data.iloc[:, :-1]
Y = np.array(data.loc[:, 'Biopsy']).reshape((-1,1))

# %% Feature Scaling with MinMax:

min_max = MinMaxScaler()
min_max.fit(np.array(X))
X.loc[:, :] = min_max.transform(np.array(X.loc[:, :]))
min_max.fit(np.array(Y))
Y = min_max.transform(np.array(Y))

# %% Splitting Train and Test data:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# %% # %% Using Linear Model to fit the data:

lr = LogisticRegression()
lr.fit(X_train, Y_train.ravel())
Y_predict = lr.predict(X_test)
lr1 = accuracy_score(Y_test, Y_predict)
cm = confusion_matrix(Y_test, Y_predict)
sn.heatmap(cm, annot=True, fmt="d")
plt.show()

# %% KNeighbors Classifier:

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train.ravel())
Y_predict = classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_predict)
sn.heatmap(cm, annot=True, fmt="d")
plt.show()
kc1 = accuracy_score(Y_test, Y_predict)

# %% Random Forest Regressor Classification:

lr = RandomForestRegressor(n_jobs=-1, verbose=2, n_estimators=200)
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)
cm = confusion_matrix(Y_test, Y_predict)
sn.heatmap(cm, annot=True, fmt="d")
plt.show()
rf1 = accuracy_score(Y_test, Y_predict)

# %% XGB Classifier:

xgb = xg.XGBClassifier(n_estimators=1000, verbosity=3, n_jobs=-1)
xgb.fit(X_train, Y_train)
Y_predict = xgb.predict(X_test)
cm = confusion_matrix(Y_test, Y_predict)
sn.heatmap(cm, annot=True, fmt="d")
plt.show()
xgb1 = accuracy_score(Y_test, Y_predict)

# %% Tabulation of the results:

models = pd.DataFrame({'Model': ['Logistic Regression', 'KNeighbour', 'Random Forest', 'XGB'], 'Score': [lr1, kc1, rf1, xgb1]})
sorted_model = models.sort_values(by='Score', ascending=False)
print(sorted_model)

# %%
