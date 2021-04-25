import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
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

clf = OneVsRestClassifier(SVC( C=1.0, kernel='rbf', degree=3, gamma=1, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=True, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)).fit(X_train_scaled, y_train)
t1_stop = perf_counter()
print("Training Time(sec):",t1_stop-t1_start)

p_start=perf_counter()
y_pred=clf.predict(X_test_scaled)
p_stop=perf_counter()
print("Prediction Speed(sec):",p_stop-p_start)
#cm=confusion_matrix(y_test,y_pred)

target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

#details(cm)

n=sio.loadmat("chb12_27")
personne=d['NewData']
Xp=data[:,:4]

i=0
y_pred=clf.predict(Xp)
print(y_pred)
count=0
while (i<len(y_pred) and count<2):
    y=y_pred[i]
    if (y==1):
        print("hi")
        count+=1
    if(y==0):
        count=0
    i+=1

if count==2:
    print("the patient is having a seizure")
