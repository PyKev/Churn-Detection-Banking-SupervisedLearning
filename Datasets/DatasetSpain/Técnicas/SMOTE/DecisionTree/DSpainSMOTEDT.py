import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetSpain.DatasetSpain import X_testSpain, y_testSpain, X_smoteSpain, y_smoteSpain, X_validSpain, y_validSpain

X_testSpain = pd.concat([X_testSpain, X_validSpain])
y_testSpain = pd.concat([y_testSpain, y_validSpain])

parametros = {'max_depth': [14], # Los valores probados en este parámetro fueron: 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20. Y seleccionamos el que se muestra ahora.
              'min_samples_leaf': [2], # Los valores probados en este parámetro fueron: 2,6,10,14,20. Y seleccionamos el que se muestra ahora.
             }

DT_model = GridSearchCV(estimator  = DecisionTreeClassifier(random_state = 123),
        param_grid = parametros,
        scoring    = 'accuracy',
        cv         = 3,
        return_train_score = True)

DT_model.fit(X_smoteSpain, y_smoteSpain)
print(DT_model.best_params_, ":",DT_model.scoring)
DT_model.score(X_testSpain, y_testSpain)
y_pred = DT_model.predict(X_testSpain)

comparacion = pd.DataFrame({'Churn real': y_testSpain, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', DT_model.best_score_)
precision = precision_score(y_testSpain, y_pred)
exactitud = accuracy_score(y_testSpain, y_pred)
Sensibilidad = recall_score(y_testSpain, y_pred)
valorF1 = f1_score(y_testSpain, y_pred)
CurvaROC = roc_auc_score(y_testSpain, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)