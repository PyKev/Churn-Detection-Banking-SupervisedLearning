import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV
from Datasets.DatasetFrance.DatasetFrance import X_testFrance, y_testFrance, X_smote_ennFrance, y_smote_ennFrance, X_validFrance, y_validFrance
xgb = xgboost.XGBClassifier()

parametros = {'nthreads' : [1],
              'objective': ['binary:logistic'],
              'learning_rate': [0.3165], #utilizamos 0.3165 dado que da el mejor valor
              'n_estimators': [35]} #utilizamos 35 dado que da el mejor valor

fit_parametros = {'early_stopping_rounds': 10,
              'eval_metric': 'logloss',
              'eval_set': [(X_testFrance, y_testFrance)]}

clasificador = GridSearchCV(xgb,parametros, cv=3 , scoring = 'accuracy')

clasificador.fit(X_smote_ennFrance, y_smote_ennFrance, **fit_parametros)
print('mejor estimador:',clasificador.best_estimator_)
print('mejor puntaje:', clasificador.best_score_)

best_xgb = clasificador.best_estimator_
y_pred = best_xgb.predict(X_validFrance)
comparacion = pd.DataFrame({'Churn real': y_validFrance, 'Churn predicho': y_pred})
#print(comparacion.head(30))

precision = precision_score(y_validFrance, y_pred)
exactitud = accuracy_score(y_validFrance, y_pred)
Sensibilidad = recall_score(y_validFrance, y_pred)
valorF1 = f1_score(y_validFrance, y_pred)
CurvaROC = roc_auc_score(y_validFrance, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)
