import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetSpain.DatasetSpain import X_testSpain, y_testSpain, X_smoteSpain, y_smoteSpain, X_validSpain, y_validSpain

X_testSpain = pd.concat([X_testSpain, X_validSpain])
y_testSpain = pd.concat([y_testSpain, y_validSpain])

parametros = {'var_smoothing': [0.0001]} # Los valores probados en este parámetro
# fueron: np.logspace(0,-9, num=100) que devuelve números espaciados uniformemente en una escala
# logarítmica desde 0 hasta -9 y genera 100 muestras Y seleccionamos el que se muestra ahora.

NB_model = GridSearchCV(estimator=GaussianNB(), param_grid=parametros, refit = True, verbose = 3, cv=3)

NB_model.fit(X_smoteSpain, y_smoteSpain)
print(NB_model.best_params_, ":",NB_model.scoring)
NB_model.score(X_testSpain, y_testSpain)
y_pred = NB_model.predict(X_testSpain)

comparacion = pd.DataFrame({'Churn real': y_testSpain, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', NB_model.best_score_)
precision = precision_score(y_testSpain, y_pred)
exactitud = accuracy_score(y_testSpain, y_pred)
Sensibilidad = recall_score(y_testSpain, y_pred)
valorF1 = f1_score(y_testSpain, y_pred)
CurvaROC = roc_auc_score(y_testSpain, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)