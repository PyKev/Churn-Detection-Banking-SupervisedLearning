import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetGermany.DatasetGermany import X_testGermany, y_testGermany, X_smote_tomekGermany, y_smote_tomekGermany, X_validGermany, y_validGermany

X_testGermany = pd.concat([X_testGermany, X_validGermany])
y_testGermany = pd.concat([y_testGermany, y_validGermany])

parametros = {'n_estimators': [22], # Los valores probados en este parámetro fueron: 22,44,66,88. Y seleccionamos el que se muestra ahora.
              'min_samples_leaf': [41], # Los valores probados en este parámetro fueron: 3,15,27,39,41. Y seleccionamos el que se muestra ahora.
             }

BA_model = GridSearchCV(estimator  = RandomForestClassifier(random_state = 123),
        param_grid = parametros,
        scoring    = 'accuracy',
        cv         = 3,
        return_train_score = True)

BA_model.fit(X_smote_tomekGermany, y_smote_tomekGermany)
print(BA_model.best_params_, ":",BA_model.scoring)
BA_model.score(X_testGermany, y_testGermany)
y_pred = BA_model.predict(X_testGermany)

comparacion = pd.DataFrame({'Churn real': y_testGermany, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', BA_model.best_score_)
precision = precision_score(y_testGermany, y_pred)
exactitud = accuracy_score(y_testGermany, y_pred)
Sensibilidad = recall_score(y_testGermany, y_pred)
valorF1 = f1_score(y_testGermany, y_pred)
CurvaROC = roc_auc_score(y_testGermany, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)