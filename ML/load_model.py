import scipy.io as sio
import pandas
import pickle
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import recall_score, confusion_matrix, f1_score,accuracy_score,precision_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from time import perf_counter

d=sio.loadmat("MDP_final")
data=d['NewData']
X=data[:,:4]
y=data[:,4]
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0,stratify=y)
scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)
with open ('final.pkl','rb') as file:
    clf=pickle.load(file)

t=sio.loadmat("chb12_27")
test=t['NewData']
Xp=test[:,:4]
yp=test[:,4]

Xp_scaled=scaler.transform(Xp)

y=clf.predict(Xp_scaled)
count=0
i=0
while (i<len(y) and count<2):
    if y[i]==0:
        count=0
    if y[i]==1:
        count+=1
    i+=1

if count==2:
    print("ALERT: seizure")



