
# Data Preprocessing 

# Importing the common libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Importing the training dataset
dataset = pd.read_csv('Flow Training Data.csv') # name or path of trainng data

# Cleaning the data
dataset = dataset[dataset.Duration != 0]

# Splitting variables 

# Use for prediciting VPN vs non-VPN
def splitVPNnonVPN():
    global X, y
    X = dataset.iloc[:, 8:84].values
    y = dataset.iloc[:, 84].values
    return X, y

# Use for prediciting Application 
def splitApplication():
    X = dataset.iloc[:, 8:84].values
    y = dataset.iloc[:, 85].values
    return X, y

X, y = splitVPNnonVPN()  # select function above chose to predict VPN or Application

# Encoding the Independent Variables

from sklearn.preprocessing import LabelEncoder
labelencoderY = LabelEncoder()
y = labelencoderY.fit_transform(y)
Y_axis = np.unique(labelencoderY.inverse_transform(y))

# Splitting the dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Import the new testset
testset = pd.read_csv(r'C:\Users\q_blo\Documents\Masters Project BTC\Data\merged\Paul_tcpdump.pcap_Flow.csv')  # name or path to file to be predicted
X_new = testset.iloc[:, 7:83].values
y_new = testset.iloc[:, 83].values

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
X_new = sc_X.transform(X_new)

# Making metrics
def Make_metrics(name, y_pred):
    global cm 
    accuracy= accuracy_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print(name,"\nAccuracy:", accuracy, "\nRecall:", recall, "\nPrecision:", precision, "\nf1:", f1)


# Random Forest with RandomSearch
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

# Refine the performance of the model by adding details here - adding more will increase training time
grid = {'n_estimators': [ 800]}
            
VPN_grid = {'n_estimators': [ 800]}

App_grid = {'n_estimators': [ 800],
            'criterion': ['entropy'],
            'max_features': ['auto'],
            'max_depth': [60],
            'min_samples_split': [10],
            'min_samples_leaf': [4],
            'bootstrap': [False]
                }

"""
full_grid = {'n_estimators': [100, 300, 500, 800, 1000],
               'criterion': ['gini', 'entropy'],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 5)],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}
"""

# Cross Validation and Training
rf_grid= RandomizedSearchCV(estimator = rf, param_distributions = grid, n_iter=30, cv = 5, verbose=2, n_jobs = -1)
Best_model = rf_grid.fit(X_train, y_train)

# Confirming the model's perfomance
y_pred_RF = Best_model.predict(X_test)
cm_RF = confusion_matrix(y_test, y_pred_RF)
Make_metrics("RF",y_pred_RF)
print(classification_report(y_test, y_pred_RF))

# Predciting results for new data
y_pred_RF_new = Best_model.predict(X_new)

# Replace the labels
index = []
for i in range(len(Y_axis)):
    index.append(i)
key_index = np.array(index)
key_index = np.column_stack((key_index, Y_axis))
key_index = pd.DataFrame(key_index)
pred = pd.DataFrame(y_pred_RF_new)
results = []
for r in y_pred_RF_new:
    for i in key_index[0]:
        if r == i:
            results.append(key_index.iloc[i,1])

# Add the predictions to the testset
testset['predictions'] = results
