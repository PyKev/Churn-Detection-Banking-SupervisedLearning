import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from Datasets.DatasetFranceGermany.DatasetFranceGermany import X_testFranceGermany, y_testFranceGermany, X_smote_ennFranceGermany, y_smote_ennFranceGermany, X_validFranceGermany, y_validFranceGermany

X_testFranceGermany = pd.concat([X_testFranceGermany, X_validFranceGermany])
y_testFranceGermany = pd.concat([y_testFranceGermany, y_validFranceGermany])

knn = KNeighborsClassifier()
parametros = {'n_neighbors': [1], # Los valores probados en este parámetro fueron: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30. Y seleccionamos el que se muestra ahora.
            }

grid = GridSearchCV(knn, parametros, cv=3, scoring='accuracy', return_train_score=False, verbose=1)

KNN_model = grid.fit(X_smote_ennFranceGermany, y_smote_ennFranceGermany)

print(KNN_model.best_params_, ":",KNN_model.scoring)
KNN_model.score(X_testFranceGermany, y_testFranceGermany)
y_pred = KNN_model.predict(X_testFranceGermany)

comparacion = pd.DataFrame({'Churn real': y_testFranceGermany, 'Churn predicho': y_pred})
#print(comparacion.head(30))

print('mejor estimador:', KNN_model.best_score_)
precision = precision_score(y_testFranceGermany, y_pred)
exactitud = accuracy_score(y_testFranceGermany, y_pred)
Sensibilidad = recall_score(y_testFranceGermany, y_pred)
valorF1 = f1_score(y_testFranceGermany, y_pred)
CurvaROC = roc_auc_score(y_testFranceGermany, y_pred)
print("Precisión (Precision): ",precision,"\n","Exactitud (Accuracy): ",exactitud,"\n",
      "Sensibilidad (Recall): ",Sensibilidad,"\n","Valor F1 (f1 score): ",valorF1,"\n","Curva ROC Score: ",CurvaROC)