# Aqui va a ir el codigo de tu Proyecto Final4
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')
import sys, os

Datasets = [["DatasetFrance", "DFrance"],
            ["DatasetFranceGermany", "DFranceGermany"],
            ["DatasetGeneral", "DGeneral"],
            ["DatasetGermany", "DGermany"],
            ["DatasetGermanySpain", "DGermanySpain"],
            ["DatasetSpain", "DSpain"],
            ["DatasetSpainFrance", "DSpainFrance"]]

Algoritmos = [["DecisionTree", "DT"],
              ["KNN", "KNN"],
              ["NaiveBayes", "NB"],
              ["RandomForest", "RF"],
              ["RegresionLogistica", "LR"],
              ["SVM", "SVM"],
              ["XGBoost", "XGBoost"]]

Metodos = ["RUS", "SMOTE", "SMOTEENN", "SMOTETomek"]

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

for Dataset in Datasets:
    for Metodo in Metodos:
        for Algoritmo in Algoritmos:
            with suppress_stdout():
                COD = "from Datasets."+Dataset[0]+".Técnicas."+Metodo+"."+Algoritmo[0]+"."+Dataset[1]+Metodo+Algoritmo[1]+" import precision, exactitud, Sensibilidad, valorF1, CurvaROC"
                exec(COD)
            print("\t \n Métricas del conjunto ",Dataset[0], " - ",Metodo," - ", Algoritmo[0],"\n Precisión (Precision): ", precision, "\n", "Exactitud (Accuracy): ", exactitud, "\n", "Sensibilidad (Recall): ", Sensibilidad, "\n", "Valor F1 (f1 score): ", valorF1, "\n", "Curva ROC Score: ",CurvaROC)
