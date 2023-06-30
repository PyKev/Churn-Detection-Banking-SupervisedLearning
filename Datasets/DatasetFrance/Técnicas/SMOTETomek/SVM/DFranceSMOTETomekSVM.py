import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetFrance.DatasetFrance import X_testFrance, y_testFrance, X_smote_tomekFrance, y_smote_tomekFrance, X_validFrance, y_validFrance

X_testFrance = pd.concat([X_testFrance, X_validFrance])
y_testFrance = pd.concat([y_testFrance, y_validFrance])

parametros = {'C': [1000], # Los valores probados en este parámetro fueron: 0.1, 1, 10, 100, 1000. Y seleccionamos el que se muestra ahora.
              'gamma': [1], # Los valores probados en este parámetro fueron: 1, 0.1, 0.01, 0.001, 0.0001. Y seleccionamos el que se muestra ahora.
              'kernel': ['rbf']}

SVM_model = GridSearchCV(SVC(), parametros, refit = True, verbose = 3, cv=2)

SVM_model.fit(X_smote_tomekFrance, y_smote_tomekFrance)
print(SVM_model.best_params_, ":",SVM_model.scoring)
SVM_model.score(X_testFrance, y_testFrance)
y_pred = SVM_model.predict(X_testFrance)

comparacion = pd.DataFrame({'Churn real': y_testFrance, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', SVM_model.best_score_)
precision = precision_score(y_testFrance, y_pred)
exactitud = accuracy_score(y_testFrance, y_pred)
Sensibilidad = recall_score(y_testFrance, y_pred)
valorF1 = f1_score(y_testFrance, y_pred)
CurvaROC = roc_auc_score(y_testFrance, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)