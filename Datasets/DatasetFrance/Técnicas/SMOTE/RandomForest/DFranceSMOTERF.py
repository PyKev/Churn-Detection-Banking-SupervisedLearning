import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetFrance.DatasetFrance import X_testFrance, y_testFrance, X_smoteFrance, y_smoteFrance, X_validFrance, y_validFrance

X_testFrance = pd.concat([X_testFrance, X_validFrance])
y_testFrance = pd.concat([y_testFrance, y_validFrance])

parametros = {'n_estimators': [96], # Los valores probados en este parámetro fueron: 19,32,57,62,96. Y seleccionamos el que se muestra ahora.
              'min_samples_leaf': [8], # Los valores probados en este parámetro fueron: 8,18,22,37,44. Y seleccionamos el que se muestra ahora.
             }

BA_model = GridSearchCV(estimator  = RandomForestClassifier(random_state = 123),
        param_grid = parametros,
        scoring    = 'accuracy',
        cv         = 3,
        return_train_score = True)

BA_model.fit(X_smoteFrance, y_smoteFrance)
print(BA_model.best_params_, ":",BA_model.scoring)
BA_model.score(X_testFrance, y_testFrance)
y_pred = BA_model.predict(X_testFrance)

comparacion = pd.DataFrame({'Churn real': y_testFrance, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', BA_model.best_score_)
precision = precision_score(y_testFrance, y_pred)
exactitud = accuracy_score(y_testFrance, y_pred)
Sensibilidad = recall_score(y_testFrance, y_pred)
valorF1 = f1_score(y_testFrance, y_pred)
CurvaROC = roc_auc_score(y_testFrance, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)