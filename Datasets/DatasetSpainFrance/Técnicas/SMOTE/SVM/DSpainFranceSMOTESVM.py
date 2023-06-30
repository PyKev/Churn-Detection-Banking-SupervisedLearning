import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetSpainFrance.DatasetSpainFrance import X_testSpainFrance, y_testSpainFrance, X_smoteSpainFrance, y_smoteSpainFrance, X_validSpainFrance, y_validSpainFrance

X_testSpainFrance = pd.concat([X_testSpainFrance, X_validSpainFrance])
y_testSpainFrance = pd.concat([y_testSpainFrance, y_validSpainFrance])

parametros = {'C': [1000], # Los valores probados en este parámetro fueron: 0.1, 1, 10, 100, 1000. Y seleccionamos el que se muestra ahora.
              'gamma': [1], # Los valores probados en este parámetro fueron: 1, 0.1, 0.01, 0.001, 0.0001. Y seleccionamos el que se muestra ahora.
              'kernel': ['rbf']}

SVM_model = GridSearchCV(SVC(), parametros, refit = True, verbose = 3, cv=2)

SVM_model.fit(X_smoteSpainFrance, y_smoteSpainFrance)
print(SVM_model.best_params_, ":",SVM_model.scoring)
SVM_model.score(X_testSpainFrance, y_testSpainFrance)
y_pred = SVM_model.predict(X_testSpainFrance)

comparacion = pd.DataFrame({'Churn real': y_testSpainFrance, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', SVM_model.best_score_)
precision = precision_score(y_testSpainFrance, y_pred)
exactitud = accuracy_score(y_testSpainFrance, y_pred)
Sensibilidad = recall_score(y_testSpainFrance, y_pred)
valorF1 = f1_score(y_testSpainFrance, y_pred)
CurvaROC = roc_auc_score(y_testSpainFrance, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)