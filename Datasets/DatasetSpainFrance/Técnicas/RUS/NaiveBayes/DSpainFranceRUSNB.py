import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetSpainFrance.DatasetSpainFrance import X_testSpainFrance, y_testSpainFrance, X_rusSpainFrance, y_rusSpainFrance, X_validSpainFrance, y_validSpainFrance

X_testSpainFrance = pd.concat([X_testSpainFrance, X_validSpainFrance])
y_testSpainFrance = pd.concat([y_testSpainFrance, y_validSpainFrance])

parametros = {'var_smoothing': [0.0005336699231206307]} # Los valores probados en este parámetro
# fueron: np.logspace(0,-9, num=100) que devuelve números espaciados uniformemente en una escala
# logarítmica desde 0 hasta -9 y genera 100 muestras Y seleccionamos el que se muestra ahora.

NB_model = GridSearchCV(estimator=GaussianNB(), param_grid=parametros, refit = True, verbose = 3, cv=3)

NB_model.fit(X_rusSpainFrance, y_rusSpainFrance)
print(NB_model.best_params_, ":",NB_model.scoring)
NB_model.score(X_testSpainFrance, y_testSpainFrance)
y_pred = NB_model.predict(X_testSpainFrance)

comparacion = pd.DataFrame({'Churn real': y_testSpainFrance, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', NB_model.best_score_)
precision = precision_score(y_testSpainFrance, y_pred)
exactitud = accuracy_score(y_testSpainFrance, y_pred)
Sensibilidad = recall_score(y_testSpainFrance, y_pred)
valorF1 = f1_score(y_testSpainFrance, y_pred)
CurvaROC = roc_auc_score(y_testSpainFrance, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)