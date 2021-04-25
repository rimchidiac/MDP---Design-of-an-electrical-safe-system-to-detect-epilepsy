import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.combine import SMOTEENN
from sklearn.multiclass import OneVsRestClassifier
from details import details

d=sio.loadmat("MDP_binary")
data=d['NewData']
X=data[:,:4]
y=data[:,4]

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0,stratify=y)

clf = OneVsRestClassifier(SVC( C=1.0, kernel='poly', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)).fit(X_train, y_train)

y_pred=clf.predict(X_test)
cm=confusion_matrix(y_test,y_pred)

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

details(cm)
