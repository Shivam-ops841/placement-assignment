import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



dataset=pd.read_csv('heart.csv')
dataset=dataset.drop_duplicates()
cat_val=[]
cont_val=[]
#Separation of categorical and numerical columns
for column in dataset.columns:
    if dataset[column].nunique()<=10:
        cat_val.append(column)
    else:
        cont_val.append(column)
print(cat_val)
print(cont_val)
cat_val.remove('sex')
cat_val.remove('target')
dataset=pd.get_dummies(dataset,columns=cat_val,drop_first=True)
print(dataset.head())

st=StandardScaler()
dataset[cont_val]=st.fit_transform(dataset[cont_val])
print(dataset.head())


X=dataset.drop('target',axis=1)
y=dataset['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)



#Logistic Regression
log=LogisticRegression()
log.fit(X_train,y_train)
y_pred1=log.predict(X_test)
cm=confusion_matrix(y_test,y_pred1)
print('Accuracy score of Logistic Regression:')
print(accuracy_score(y_test,y_pred1))
print(cm)
print()
print()
print()


##SVM
svm=svm.SVC()
svm.fit(X_train,y_train)
y_pred2=svm.predict(X_test)
cm=confusion_matrix(y_test,y_pred2)
print("Accuracy of support vector classifer")

print(accuracy_score(y_test,y_pred2))
print()
print()
print()


#KNN CLASSIFIER

#When value of n is 5
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred3=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred3)
print(accuracy_score(y_test,y_pred3))

#Selecting value of k
score=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    t=knn.predict(X_test)
    score.append(accuracy_score(y_test,t))
print(score)
print(cm)

#Using KNN after selecting value of K
knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train,y_train)
y_pred3=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred1)
print(accuracy_score(y_test,y_pred3))


dataset=pd.read_csv('heart.csv')
X=dataset.drop('target',axis=1)
y=dataset['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(cm)
print()
print()



#Decision Tree Classifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred4=dt.predict(X_test)
print(accuracy_score(y_test,y_pred4))
cm=confusion_matrix(y_test,y_pred4)
print(cm)
print()


#Random Forest Classifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred5=rf.predict(X_test)
print(accuracy_score(y_test,y_pred5))
cm=confusion_matrix(y_test,y_pred5)
print(cm)
print()
print()



#Gradient Boosting Classifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_pred6=gbc.predict(X_test)
print(accuracy_score(y_test,y_pred6))
cm=confusion_matrix(y_test,y_pred6)
print(cm)


#Prediction
new_data={
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
    'slope':2,
    'ca':2,
    'thal':3,
}
p=rf.predict(X_test)
for i in X_test:
    if(i==0):
        print("Person is Healthy")
    else:
         print("Person is unhealthy")