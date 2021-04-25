import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from details import details

d=sio.loadmat("MDP_binary")
data=d['NewData']
X=data[:,:4]
y=data[:,4]

print(data.shape)

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0,stratify=y)

scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

clf = OneVsRestClassifier(SVC( C=1.0, kernel='rbf', degree=3, gamma=8, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)).fit(X_train_scaled, y_train)

y_pred=clf.predict(X_test_scaled)
cm=confusion_matrix(y_test,y_pred)

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

details(cm)
