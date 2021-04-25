import scipy.io as sio
import numpy as np 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix ,classification_report, f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from imblearn.under_sampling import TomekLinks
from time import perf_counter
from details import details

d=sio.loadmat("MDP_binary")
data=d['NewData']
X_1=data[:,:4]
y_1=data[:,4]

TL = TomekLinks()
X, y = TL.fit_resample(X_1, y_1)

t1_start = perf_counter() 
X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=0,stratify=y)

knn=KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=None)
scores = cross_val_score(knn, X, y, cv=5)
knn.fit(X_train, y_train)
t1_stop = perf_counter()
print("Training Time(sec):",t1_stop-t1_start)

p_start=perf_counter()
y_pred= knn.predict(X_test)
p_stop=perf_counter()
print("Prediction Speed(sec):",p_stop-p_start)
cm=confusion_matrix(y_test,y_pred)

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))
print("accuracy:",knn.score(X_test,y_test))
details(cm)
print(scores)

train_accuracy=accuracy_score(training_target, trained_model.predict(training_features))
