import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetSpain.DatasetSpain import X_testSpain, y_testSpain, X_rusSpain, y_rusSpain, X_validSpain, y_validSpain

X_testSpain = pd.concat([X_testSpain, X_validSpain])
y_testSpain = pd.concat([y_testSpain, y_validSpain])

parametros = {'n_estimators': [57], # Los valores probados en este parámetro fueron: 19,32,57,62,96. Y seleccionamos el que se muestra ahora.
              'min_samples_leaf': [8], # Los valores probados en este parámetro fueron: 8,18,22,37,44. Y seleccionamos el que se muestra ahora.
             }

BA_model = GridSearchCV(estimator  = RandomForestClassifier(random_state = 123),
        param_grid = parametros,
        scoring    = 'accuracy',
        cv         = 3,
        return_train_score = True)

BA_model.fit(X_rusSpain, y_rusSpain)
print(BA_model.best_params_, ":",BA_model.scoring)
BA_model.score(X_testSpain, y_testSpain)
y_pred = BA_model.predict(X_testSpain)

comparacion = pd.DataFrame({'Churn real': y_testSpain, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', BA_model.best_score_)
precision = precision_score(y_testSpain, y_pred)
exactitud = accuracy_score(y_testSpain, y_pred)
Sensibilidad = recall_score(y_testSpain, y_pred)
valorF1 = f1_score(y_testSpain, y_pred)
CurvaROC = roc_auc_score(y_testSpain, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)