import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetGermanySpain.DatasetGermanySpain import X_testGermanySpain, y_testGermanySpain, X_smote_tomekGermanySpain, y_smote_tomekGermanySpain, X_validGermanySpain, y_validGermanySpain

X_testGermanySpain = pd.concat([X_testGermanySpain, X_validGermanySpain])
y_testGermanySpain = pd.concat([y_testGermanySpain, y_validGermanySpain])

parametros = {'n_estimators': [89], # Los valores probados en este parámetro fueron: 18,27,45,67,89. Y seleccionamos el que se muestra ahora.
              'min_samples_leaf': [2], # Los valores probados en este parámetro fueron: 2,6,10,14,20. Y seleccionamos el que se muestra ahora.
             }

BA_model = GridSearchCV(estimator  = RandomForestClassifier(random_state = 123),
        param_grid = parametros,
        scoring    = 'accuracy',
        cv         = 3,
        return_train_score = True)

BA_model.fit(X_smote_tomekGermanySpain, y_smote_tomekGermanySpain)
print(BA_model.best_params_, ":",BA_model.scoring)
BA_model.score(X_testGermanySpain, y_testGermanySpain)
y_pred = BA_model.predict(X_testGermanySpain)

comparacion = pd.DataFrame({'Churn real': y_testGermanySpain, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', BA_model.best_score_)
precision = precision_score(y_testGermanySpain, y_pred)
exactitud = accuracy_score(y_testGermanySpain, y_pred)
Sensibilidad = recall_score(y_testGermanySpain, y_pred)
valorF1 = f1_score(y_testGermanySpain, y_pred)
CurvaROC = roc_auc_score(y_testGermanySpain, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)
