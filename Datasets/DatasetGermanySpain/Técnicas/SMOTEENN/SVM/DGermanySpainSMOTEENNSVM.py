import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetGermanySpain.DatasetGermanySpain import X_testGermanySpain, y_testGermanySpain, X_smote_ennGermanySpain, y_smote_ennGermanySpain, X_validGermanySpain, y_validGermanySpain

X_testGermanySpain = pd.concat([X_testGermanySpain, X_validGermanySpain])
y_testGermanySpain = pd.concat([y_testGermanySpain, y_validGermanySpain])

parametros = {'C': [100], # Los valores probados en este parámetro fueron: 0.1, 1, 10, 100, 1000. Y seleccionamos el que se muestra ahora.
              'gamma': [1], # Los valores probados en este parámetro fueron: 1, 0.1, 0.01, 0.001, 0.0001. Y seleccionamos el que se muestra ahora.
              'kernel': ['rbf']}

SVM_model = GridSearchCV(SVC(), parametros, refit = True, verbose = 3, cv=2)

SVM_model.fit(X_smote_ennGermanySpain, y_smote_ennGermanySpain)
print(SVM_model.best_params_, ":",SVM_model.scoring)
SVM_model.score(X_testGermanySpain, y_testGermanySpain)
y_pred = SVM_model.predict(X_testGermanySpain)

comparacion = pd.DataFrame({'Churn real': y_testGermanySpain, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', SVM_model.best_score_)
precision = precision_score(y_testGermanySpain, y_pred)
exactitud = accuracy_score(y_testGermanySpain, y_pred)
Sensibilidad = recall_score(y_testGermanySpain, y_pred)
valorF1 = f1_score(y_testGermanySpain, y_pred)
CurvaROC = roc_auc_score(y_testGermanySpain, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)