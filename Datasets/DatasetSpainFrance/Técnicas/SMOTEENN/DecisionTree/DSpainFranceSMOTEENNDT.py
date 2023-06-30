import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetSpainFrance.DatasetSpainFrance import X_testSpainFrance, y_testSpainFrance, X_smote_ennSpainFrance, y_smote_ennSpainFrance, X_validSpainFrance, y_validSpainFrance

X_testSpainFrance = pd.concat([X_testSpainFrance, X_validSpainFrance])
y_testSpainFrance = pd.concat([y_testSpainFrance, y_validSpainFrance])

parametros = {'max_depth': [11], # Los valores probados en este parámetro fueron: 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20. Y seleccionamos el que se muestra ahora.
              'min_samples_leaf': [5], # Los valores probados en este parámetro fueron: 5,17,24,35,50. Y seleccionamos el que se muestra ahora.
             }

DT_model = GridSearchCV(estimator  = DecisionTreeClassifier(random_state = 123),
        param_grid = parametros,
        scoring    = 'accuracy',
        cv         = 3,
        return_train_score = True)

DT_model.fit(X_smote_ennSpainFrance, y_smote_ennSpainFrance)
print(DT_model.best_params_, ":",DT_model.scoring)
DT_model.score(X_testSpainFrance, y_testSpainFrance)
y_pred = DT_model.predict(X_testSpainFrance)

comparacion = pd.DataFrame({'Churn real': y_testSpainFrance, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', DT_model.best_score_)
precision = precision_score(y_testSpainFrance, y_pred)
exactitud = accuracy_score(y_testSpainFrance, y_pred)
Sensibilidad = recall_score(y_testSpainFrance, y_pred)
valorF1 = f1_score(y_testSpainFrance, y_pred)
CurvaROC = roc_auc_score(y_testSpainFrance, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)