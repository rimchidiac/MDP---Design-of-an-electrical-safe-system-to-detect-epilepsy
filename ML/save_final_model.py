import scipy.io as sio
import pandas
import pickle
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import recall_score, confusion_matrix, f1_score,accuracy_score,precision_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from time import perf_counter

d=sio.loadmat("MDP_final")
data=d['NewData']
X=data[:,:4]
y=data[:,4]

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0,stratify=y)

scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

clf = OneVsRestClassifier(SVC( C=1.0, kernel='rbf', degree=3, gamma=1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)).fit(X_train_scaled, y_train)

pkl_filename='final.pkl'
with open(pkl_filename,'wb') as file:
    pickle.dump(clf,file)
