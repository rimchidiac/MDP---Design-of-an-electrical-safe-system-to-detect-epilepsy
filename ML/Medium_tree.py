import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
from time import perf_counter
from details import details

d=sio.loadmat("MDP_binary")
data=d['NewData']
X=data[:,:4]
y=data[:,4]

t1_start = perf_counter()
X_train, X_test, y_train, y_test= train_test_split(X,y, random_state=0)

clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,min_samples_split=20, min_samples_leaf=1, max_features=None, random_state=None, max_leaf_nodes=None)
clf.fit(X_train, y_train)
t1_stop = perf_counter()
print("Training Time(sec):",t1_stop-t1_start)

p_start=perf_counter()


y_pred = clf.predict(X_test)
p_stop=perf_counter()
print("Prediction Speed(sec):",p_stop-p_start)

cm=confusion_matrix(y_test,y_pred)

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
details(cm)
