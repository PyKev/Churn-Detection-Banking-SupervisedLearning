import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetGermanySpain.DatasetGermanySpain import X_testGermanySpain, y_testGermanySpain, X_smoteGermanySpain, y_smoteGermanySpain, X_validGermanySpain, y_validGermanySpain

X_testGermanySpain = pd.concat([X_testGermanySpain, X_validGermanySpain])
y_testGermanySpain = pd.concat([y_testGermanySpain, y_validGermanySpain])

parametros = {'max_depth': [10], # Los valores probados en este parámetro fueron: 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20. Y seleccionamos el que se muestra ahora.
              'min_samples_leaf': [15], # Los valores probados en este parámetro fueron: 3,15,27,39,41. Y seleccionamos el que se muestra ahora.
             }

DT_model = GridSearchCV(estimator  = DecisionTreeClassifier(random_state = 123),
        param_grid = parametros,
        scoring    = 'accuracy',
        cv         = 3,
        return_train_score = True)

DT_model.fit(X_smoteGermanySpain, y_smoteGermanySpain)
print(DT_model.best_params_, ":",DT_model.scoring)
DT_model.score(X_testGermanySpain, y_testGermanySpain)
y_pred = DT_model.predict(X_testGermanySpain)

comparacion = pd.DataFrame({'Churn real': y_testGermanySpain, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', DT_model.best_score_)
precision = precision_score(y_testGermanySpain, y_pred)
exactitud = accuracy_score(y_testGermanySpain, y_pred)
Sensibilidad = recall_score(y_testGermanySpain, y_pred)
valorF1 = f1_score(y_testGermanySpain, y_pred)
CurvaROC = roc_auc_score(y_testGermanySpain, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)