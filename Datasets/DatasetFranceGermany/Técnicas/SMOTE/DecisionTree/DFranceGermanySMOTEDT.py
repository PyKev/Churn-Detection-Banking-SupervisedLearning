import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetFranceGermany.DatasetFranceGermany import X_testFranceGermany, y_testFranceGermany, X_smoteFranceGermany, y_smoteFranceGermany, X_validFranceGermany, y_validFranceGermany

X_testFranceGermany = pd.concat([X_testFranceGermany, X_validFranceGermany])
y_testFranceGermany = pd.concat([y_testFranceGermany, y_validFranceGermany])

parametros = {'max_depth': [10], # Los valores probados en este parámetro fueron: 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20. Y seleccionamos el que se muestra ahora.
              'min_samples_leaf': [29], # Los valores probados en este parámetro fueron: 3,15,29,34,45. Y seleccionamos el que se muestra ahora.
             }

DT_model = GridSearchCV(estimator  = DecisionTreeClassifier(random_state = 123),
        param_grid = parametros,
        scoring    = 'accuracy',
        cv         = 3,
        return_train_score = True)

DT_model.fit(X_smoteFranceGermany, y_smoteFranceGermany)
print(DT_model.best_params_, ":",DT_model.scoring)
DT_model.score(X_testFranceGermany, y_testFranceGermany)
y_pred = DT_model.predict(X_testFranceGermany)

comparacion = pd.DataFrame({'Churn real': y_testFranceGermany, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', DT_model.best_score_)
precision = precision_score(y_testFranceGermany, y_pred)
exactitud = accuracy_score(y_testFranceGermany, y_pred)
Sensibilidad = recall_score(y_testFranceGermany, y_pred)
valorF1 = f1_score(y_testFranceGermany, y_pred)
CurvaROC = roc_auc_score(y_testFranceGermany, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)