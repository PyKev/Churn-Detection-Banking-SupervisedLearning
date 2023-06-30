import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetGermany.DatasetGermany import X_testGermany, y_testGermany, X_smoteGermany, y_smoteGermany, X_validGermany, y_validGermany

X_testGermany = pd.concat([X_testGermany, X_validGermany])
y_testGermany = pd.concat([y_testGermany, y_validGermany])

parametros = {'max_depth': [6], # Los valores probados en este parámetro fueron: 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20. Y seleccionamos el que se muestra ahora.
              'min_samples_leaf': [3], # Los valores probados en este parámetro fueron: 3,15,29,34,45. Y seleccionamos el que se muestra ahora.
             }

DT_model = GridSearchCV(estimator  = DecisionTreeClassifier(random_state = 123),
        param_grid = parametros,
        scoring    = 'accuracy',
        cv         = 3,
        return_train_score = True)

DT_model.fit(X_smoteGermany, y_smoteGermany)
print(DT_model.best_params_, ":",DT_model.scoring)
DT_model.score(X_testGermany, y_testGermany)
y_pred = DT_model.predict(X_testGermany)

comparacion = pd.DataFrame({'Churn real': y_testGermany, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', DT_model.best_score_)
precision = precision_score(y_testGermany, y_pred)
exactitud = accuracy_score(y_testGermany, y_pred)
Sensibilidad = recall_score(y_testGermany, y_pred)
valorF1 = f1_score(y_testGermany, y_pred)
CurvaROC = roc_auc_score(y_testGermany, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)