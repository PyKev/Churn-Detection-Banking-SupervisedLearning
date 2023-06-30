import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetGermany.DatasetGermany import X_testGermany, y_testGermany, X_smote_ennGermany, y_smote_ennGermany, X_validGermany, y_validGermany

X_testGermany = pd.concat([X_testGermany, X_validGermany])
y_testGermany = pd.concat([y_testGermany, y_validGermany])

parametros = {'C': [1000], # Los valores probados en este parámetro fueron: 0.1, 1, 10, 100, 1000. Y seleccionamos el que se muestra ahora.
              'gamma': [1], # Los valores probados en este parámetro fueron: 0.0001, 0.001, 0.01, 0.1, 1. Y seleccionamos el que se muestra ahora.
              'kernel': ['rbf']}

SVM_model = GridSearchCV(SVC(), parametros, refit = True, verbose = 3, cv=2)

SVM_model.fit(X_smote_ennGermany, y_smote_ennGermany)
print(SVM_model.best_params_, ":",SVM_model.scoring)
SVM_model.score(X_testGermany, y_testGermany)
y_pred = SVM_model.predict(X_testGermany)

comparacion = pd.DataFrame({'Churn real': y_testGermany, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', SVM_model.best_score_)
precision = precision_score(y_testGermany, y_pred)
exactitud = accuracy_score(y_testGermany, y_pred)
Sensibilidad = recall_score(y_testGermany, y_pred)
valorF1 = f1_score(y_testGermany, y_pred)
CurvaROC = roc_auc_score(y_testGermany, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)