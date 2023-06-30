import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetGeneral.DatasetGeneral import X_testGeneral, y_testGeneral, X_smote_ennGeneral, y_smote_ennGeneral, X_validGeneral, y_validGeneral

X_testGeneral = pd.concat([X_testGeneral, X_validGeneral])
y_testGeneral = pd.concat([y_testGeneral, y_validGeneral])

parametros = {'n_estimators': [62], # Los valores probados en este parámetro fueron: 19,32,57,62,96. Y seleccionamos el que se muestra ahora.
              'min_samples_leaf': [5], # Los valores probados en este parámetro fueron: 5,17,24,35,50. Y seleccionamos el que se muestra ahora.
             }

BA_model = GridSearchCV(estimator  = RandomForestClassifier(random_state = 123),
        param_grid = parametros,
        scoring    = 'accuracy',
        cv         = 3,
        return_train_score = True)

BA_model.fit(X_smote_ennGeneral, y_smote_ennGeneral)
print(BA_model.best_params_, ":",BA_model.scoring)
BA_model.score(X_testGeneral, y_testGeneral)
y_pred = BA_model.predict(X_testGeneral)

comparacion = pd.DataFrame({'Churn real': y_testGeneral, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', BA_model.best_score_)
precision = precision_score(y_testGeneral, y_pred)
exactitud = accuracy_score(y_testGeneral, y_pred)
Sensibilidad = recall_score(y_testGeneral, y_pred)
valorF1 = f1_score(y_testGeneral, y_pred)
CurvaROC = roc_auc_score(y_testGeneral, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)