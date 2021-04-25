import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix ,classification_report, f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from details import details
from time import perf_counter

d=sio.loadmat("MDP_balanced")
data=d['NewData']
X=data[:,:4]
y=data[:,4]

t1_start = perf_counter()
X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=0)

scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

knn=BaggingClassifier(base_estimator=KNeighborsClassifier())
knn.fit(X_train_scaled,y_train)
t1_stop = perf_counter()
print("Training Time(sec):",t1_stop-t1_start)

p_start=perf_counter()
y_pred=knn.predict(X_test_scaled)
p_stop=perf_counter()
print("Prediction Speed(sec):",p_stop-p_start)
cm=confusion_matrix(y_test,y_pred)

target_names = ['class 0', 'class 1','class2']
print(classification_report(y_test, y_pred, target_names=target_names))

print("accuracy:",knn.score(X_test_scaled,y_test))
details(cm)
