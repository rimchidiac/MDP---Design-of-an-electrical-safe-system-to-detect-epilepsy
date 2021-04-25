import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from details import details

d=sio.loadmat("MDP_binary")
data=d['NewData']
X=data[:,:4]
y=data[:,4]

X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=0,stratify=y)

scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

knn=KNeighborsClassifier(n_neighbors=100, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='euclidean', metric_params=None, n_jobs=None)
knn.fit(X_train_scaled, y_train)

y_pred=knn.predict(X_test_scaled)
cm= confusion_matrix(y_test,y_pred)
target_names=['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

print("accuracy:",knn.score(X_test_scaled,y_test))
details(cm)
