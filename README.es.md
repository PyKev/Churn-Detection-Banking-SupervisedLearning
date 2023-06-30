# Detección de deserción de clientes en el sector bancario utilizando aprendizaje supervisado

## Objetivo General del Proyecto 😃
Evaluar el desempeño del método de sobremuestreo SMOTE, el método de submuestreo RUS y los métodos de sobremuestreo seguido de submuestreo SMOTE+ENN y SMOTE+TOMEK, utilizando los algoritmos de clasificación XGBoost, LogisticRegression, SVM, Random Forest, Naive Bayes, Decision Tree y KNN en 7 datasets desbalanceados obtenidos de los países de España, Francia y Alemania en términos precisión, sensibilidad, exactitud, valor-F1 y área bajo la curva ROC.

## Objetivos Específicos del Proyecto 🧐
- OE1: Analizar las fortalezas y limitaciones de los algoritmos de clasificación como RandomForest, SVM, XGboost, LogisticRegression, Naive Bayes, Decision Tree y KNN; y métodos de desbalanceo como SMOTE, RUS, SMOTE+ENN y SMOTE+TOMEK utilizados en estudios anteriores.
- OE2: Diseñar un metodo de comparacion para identificar los mejores modelos con respecto a los algoritmos Random Forest, XGBoost, LogisticRegression, SVM, Naive Bayes, Decision Tree y KNN explorando los parámetros de los métodos de balanceo SMOTE, RUS, SMOTE+ENN y SMOTE+TOMEK y explorar la fusión de los datasets España, Francia y Alemania en variantes.
- OE3: Evaluar los métodos de remuestreo SMOTE, RUS, SMOTE+ENN, SMOTE+TOMEK para determinar la técnica más adecuada en base a las métricas Precision (precisión), Recall (sensibilidad), Accuracy (exactitud), F-measure (medida F1) y AUC(Área bajo la curva)
- OE4: Evaluar los diversos modelos obtenidos a partir de los algoritmos de clasificación Random Forest, XGBoost, LogisticRegression, SVM, Naive Bayes, Decision Tree y KNN para identificar el mejor modelo en base a las métricas Precision (precisión), Recall (sensibilidad), Accuracy (exactitud), F-measure (medida F1) y AUC(Área bajo la curva)
- OE5: Comparar el desempeño de los clasificadores Random Forest, XGBoost, LogisticRegression, SVM, Naive Bayes, Decision Tree y KNN versus los métodos de remuestreo SMOTE, RUS, SMOTE+ENN y SMOTE+TOMEK  sobre 7 variantes de datasets a partir de los 3 escenarios: España, Alemania y Francia en términos de Precision (precisión), Recall (sensibilidad), Accuracy (exactitud), F-measure (medida F1) y AUC(Área bajo la curva)

## Método propuesto 💯
Proponemos una comparativa, de los algoritmos con mejor desempeño presentado en la literatura, Extreme Gradient Boosting (XGBoost), Random Forest, SVM, Regresión Logística , Naive Bayes, Decision Tree y KNN Combinándolos con el método de submuestreo RUS, el método de sobremuestreo SMOTE, y los métodos de sobremuestreo seguidos de submmuestreo SMOTE-ENN y SMOTE-TOMEK para evaluar su comportamiento frente a 7 escenarios diferentes, con conjuntos de datos de Alemania, España, Francia y Alemania-España, España-Francia y Alemania-Francia. Ya que es probable que cada algoritmo presente un desempeño mejor en cada escenario, en base al método con el que se combine para lidiar con el problema de los datos desbalanceados en términos de precisión, sensibilidad, exactitud y valor-F1.

## Lista de Actividades realizadas para el proyecto ⭐ 💻
- Extracción del dataset de Kaggle
- Entendimiento del dataset (componentes y registros)
- Preparación de los datos (filtros, modificaciones y eliminaciones)
- Escalamiento de los datos (Reemplazar por rangos entre 0 y 1)
- Separación del dataset general en 7 dataset específicos (dataset de Francia, España, Alemania, Francia+Alemania, Francia+España, Alemania+España y Francia+Alemania+España)
- Gráfico pie chart para visualizar el desbalanceo por cada dataset
- Aplicación de los métodos RUS, SMOTE, SMOTE+ENN y SMOTE+TOMEK en cada uno de los datasets
- Aplicación del algoritmo XGBoost
- Aplicación de los algoritmos de predicción XGBoost, Random Forest, SVM, LogisticRegression, Naive Bayes, Decision Tree y KNN en cada uno de los escenarios resultado de las técnicas de muestreo en cada dataset.
- Evaluar en base a las métricas cada resultado obtenido en cada conjunto de algoritmo+técnica de remuestreo+dataset

## Lista de métricas ⚡
- Precisión
- Exhaustividad
- Exactitud
- Valor F1
- Área bajo la curva ROC (AUC)

---
## Pasos para ejecutar el aplicativo 📈
1. Clonar el proyecto 
2. Instalar las librerias ubicadas en el archivo paquetes.sh
3. Escoger python como lenguaje de interpretación
4. Seleccionar un dataset (guiarse por el nombre del archivo .py) y ejecutarlo para obtener los resultados de las técnicas RUS, SMOTE, SMOTE+ENN y SMOTE+TOMEK de ese dataset
5. Para ejecutar, en PyCharm en las opciones elegir Run > Run (Alt+Máyus+F10) y seleccionar el "Run Configuracion" por defecto  
6. Los archivos que empiezan por Dataset son derivados del proyecto principal PrototipoGrupo2.
7. En prototipogrupo2 se encuentra la preparación de datos y la separación de datasets de acuerdo a los países y se grafica los piechart que muestran la data desbalanceada.
8. En los archivos que empiezan con Dataset, en cada uno de estos se aplica los métodos de remuestreo, y en cada uno de los métodos aplicado en cada escenario se muestra un gráfico de barras de la versión antes de aplicar el método y luego de aplicado el método.

## Contribuidores🤝
- Kevin Chávez
- Kevin Humareda
- Pedro Shiguihara
