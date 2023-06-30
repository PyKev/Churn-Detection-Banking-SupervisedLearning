import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetFrance.DatasetFrance import X_testFrance, y_testFrance, X_smoteFrance, y_smoteFrance, X_validFrance, y_validFrance

X_testFrance = pd.concat([X_testFrance, X_validFrance])
y_testFrance = pd.concat([y_testFrance, y_validFrance])

knn = KNeighborsClassifier()
parametros = {'n_neighbors': [2], # Los valores probados en este parámetro fueron: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30. Y seleccionamos el que se muestra ahora.
            }

grid = GridSearchCV(knn, parametros, cv=3, scoring='accuracy', return_train_score=False, verbose=1)

KNN_model = grid.fit(X_smoteFrance, y_smoteFrance)

print(KNN_model.best_params_, ":",KNN_model.scoring)
KNN_model.score(X_testFrance, y_testFrance)
y_pred = KNN_model.predict(X_testFrance)

comparacion = pd.DataFrame({'Churn real': y_testFrance, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', KNN_model.best_score_)
precision = precision_score(y_testFrance, y_pred)
exactitud = accuracy_score(y_testFrance, y_pred)
Sensibilidad = recall_score(y_testFrance, y_pred)
valorF1 = f1_score(y_testFrance, y_pred)
CurvaROC = roc_auc_score(y_testFrance, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)