import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import RUSBoostClassifier
from time import perf_counter
from details import details

d=sio.loadmat("MDP_binary")
data=d['NewData']
X=data[:,:4]
y=data[:,4]

print(data.shape)

t1_start = perf_counter()
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0,stratify=y)

scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

rusboost = RUSBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=200, algorithm='SAMME.R',random_state=0)
rusboost.fit(X_train_scaled,y_train)
t1_stop = perf_counter()
print("Training Time(sec):",t1_stop-t1_start)

p_start=perf_counter()
y_pred=rusboost.predict(X_test_scaled)
p_stop=perf_counter()
print("Prediction Speed(sec):",p_stop-p_start)
cm=confusion_matrix(y_test,y_pred)

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))
print("accuracy:",rusboost.score(X_test_scaled,y_test))

details(cm)
