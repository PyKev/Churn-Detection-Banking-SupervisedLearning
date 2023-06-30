import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetGermanySpain.DatasetGermanySpain import X_testGermanySpain, y_testGermanySpain, X_rusGermanySpain, y_rusGermanySpain, X_validGermanySpain, y_validGermanySpain

X_testGermanySpain = pd.concat([X_testGermanySpain, X_validGermanySpain])
y_testGermanySpain = pd.concat([y_testGermanySpain, y_validGermanySpain])

parametros = {'var_smoothing': [0.0005336699231206307]} # Los valores probados en este parámetro
# fueron: np.logspace(0,-9, num=100) que devuelve números espaciados uniformemente en una escala
# logarítmica desde 0 hasta -9 y genera 100 muestras Y seleccionamos el que se muestra ahora.

NB_model = GridSearchCV(estimator=GaussianNB(), param_grid=parametros, refit = True, verbose = 3, cv=3)

NB_model.fit(X_rusGermanySpain, y_rusGermanySpain)
print(NB_model.best_params_, ":",NB_model.scoring)
NB_model.score(X_testGermanySpain, y_testGermanySpain)
y_pred = NB_model.predict(X_testGermanySpain)

comparacion = pd.DataFrame({'Churn real': y_testGermanySpain, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', NB_model.best_score_)
precision = precision_score(y_testGermanySpain, y_pred)
exactitud = accuracy_score(y_testGermanySpain, y_pred)
Sensibilidad = recall_score(y_testGermanySpain, y_pred)
valorF1 = f1_score(y_testGermanySpain, y_pred)
CurvaROC = roc_auc_score(y_testGermanySpain, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)