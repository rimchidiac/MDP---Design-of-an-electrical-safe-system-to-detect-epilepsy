import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from time import perf_counter
from details import details


d=sio.loadmat("MDP_binary")
data=d['NewData']
X=data[:,:4]
y=data[:,4]

print(X.shape)
t1_start = perf_counter() 
X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=0)

scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=None)
knn.fit(X_train_scaled, y_train)
t1_stop = perf_counter()
print("Training Time(sec):",t1_stop-t1_start)

p_start=perf_counter()
y_pred= knn.predict(X_test_scaled)
p_stop=perf_counter()
print("Prediction Speed(sec):",p_stop-p_start)
cm=confusion_matrix(y_test,y_pred)

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

print("accuracy:",knn.score(X_test_scaled,y_test))
details(cm)
train_accuracy=accuracy_score(y_train, knn.predict(X_train_scaled))
test_accuracy=accuracy_score(y_test,y_pred)

print (train_accuracy)
print(test_accuracy)
