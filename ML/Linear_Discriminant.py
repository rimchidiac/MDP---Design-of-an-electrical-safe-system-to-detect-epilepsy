import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix ,classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from time import perf_counter
from details import details

d=sio.loadmat("MDP_binary")
data=d['NewData']
X=data[:,:4]
y=data[:,4]

t1_start = perf_counter()
X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=0,stratify=y)

scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

clf= LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(X_train_scaled,y_train)
t1_stop = perf_counter()
print("Training Time(sec):",t1_stop-t1_start)

p_start=perf_counter()
y_pred=clf.predict(X_test_scaled)
p_stop=perf_counter()
print("Prediction Speed(sec):",p_stop-p_start)
cm=confusion_matrix(y_test,y_pred)

target_names=['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

print("accuracy:",clf.score(X_test_scaled,y_test))
details(cm)
