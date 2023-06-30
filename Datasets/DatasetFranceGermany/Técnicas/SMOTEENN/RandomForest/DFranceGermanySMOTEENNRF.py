import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetFranceGermany.DatasetFranceGermany import X_testFranceGermany, y_testFranceGermany, X_smote_ennFranceGermany, y_smote_ennFranceGermany, X_validFranceGermany, y_validFranceGermany

X_testFranceGermany = pd.concat([X_testFranceGermany, X_validFranceGermany])
y_testFranceGermany = pd.concat([y_testFranceGermany, y_validFranceGermany])

parametros = {'n_estimators': [30], # Los valores probados en este parámetro fueron: 10,30,50,80,95. Y seleccionamos el que se muestra ahora.
              'min_samples_leaf': [2], # Los valores probados en este parámetro fueron: 2,6,10,14,20. Y seleccionamos el que se muestra ahora.
             }

BA_model = GridSearchCV(estimator  = RandomForestClassifier(random_state = 123),
        param_grid = parametros,
        scoring    = 'accuracy',
        cv         = 3,
        return_train_score = True)

BA_model.fit(X_smote_ennFranceGermany, y_smote_ennFranceGermany)
print(BA_model.best_params_, ":",BA_model.scoring)
BA_model.score(X_testFranceGermany, y_testFranceGermany)
y_pred = BA_model.predict(X_testFranceGermany)

comparacion = pd.DataFrame({'Churn real': y_testFranceGermany, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', BA_model.best_score_)
precision = precision_score(y_testFranceGermany, y_pred)
exactitud = accuracy_score(y_testFranceGermany, y_pred)
Sensibilidad = recall_score(y_testFranceGermany, y_pred)
valorF1 = f1_score(y_testFranceGermany, y_pred)
CurvaROC = roc_auc_score(y_testFranceGermany, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)
