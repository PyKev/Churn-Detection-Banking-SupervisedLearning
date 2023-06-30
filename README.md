# Customer Churn Detection in the Banking Sector Using Supervised Learning

## General Project Objective 😃
Evaluate the performance of the SMOTE oversampling method, RUS undersampling method, and the combination of oversampling and undersampling methods (SMOTE+ENN and SMOTE+TOMEK) using the XGBoost, LogisticRegression, SVM, Random Forest, Naive Bayes, Decision Tree, and KNN classification algorithms on 7 imbalanced datasets obtained from Spain, France, and Germany in terms of precision, recall, accuracy, F1-score, and area under the ROC curve.

## Specific Project Objectives 🧐
- SO1: Analyze the strengths and limitations of classification algorithms such as RandomForest, SVM, XGBoost, LogisticRegression, Naive Bayes, Decision Tree, and KNN, and imbalance handling methods such as SMOTE, RUS, SMOTE+ENN, and SMOTE+TOMEK used in previous studies.
- SO2: Design a comparison method to identify the best models in terms of the Random Forest, XGBoost, LogisticRegression, SVM, Naive Bayes, Decision Tree, and KNN algorithms by exploring the parameters of the SMOTE, RUS, SMOTE+ENN, and SMOTE+TOMEK balancing methods and exploring the merging of the Spain, France, and Germany datasets in various combinations.
- SO3: Evaluate the SMOTE, RUS, SMOTE+ENN, and SMOTE+TOMEK resampling methods to determine the most suitable technique based on the precision, recall, accuracy, F1-score, and AUC metrics.
- SO4: Evaluate the different models obtained from the Random Forest, XGBoost, LogisticRegression, SVM, Naive Bayes, Decision Tree, and KNN classification algorithms to identify the best model based on the precision, recall, accuracy, F1-score, and AUC metrics.
- SO5: Compare the performance of the Random Forest, XGBoost, LogisticRegression, SVM, Naive Bayes, Decision Tree, and KNN classifiers versus the SMOTE, RUS, SMOTE+ENN, and SMOTE+TOMEK resampling methods on 7 different dataset variants from the Spain, Germany, and France scenarios in terms of precision, recall, accuracy, F1-score, and AUC.

## Proposed Method 💯
We propose a comparative study of the best-performing algorithms presented in the literature, including Extreme Gradient Boosting (XGBoost), Random Forest, SVM, Logistic Regression, Naive Bayes, Decision Tree, and KNN, combined with the RUS undersampling method, SMOTE oversampling method, and the combination of oversampling and undersampling methods (SMOTE-ENN and SMOTE-TOMEK). We evaluate their performance on 7 different scenarios, using datasets from Germany, Spain, France, and combinations of Germany-Spain, Spain-France, and Germany-France. Since each algorithm is likely to perform better in different scenarios based on the resampling method used to deal with the imbalanced data problem, we evaluate their performance in terms of precision, recall, accuracy, and F1-score.

## List of Activities carried out for the project ⭐ 💻
- Dataset extraction from Kaggle.
- Understanding the dataset (components and records).
- Data preparation (filters, modifications, and deletions).
- Data scaling (replace with ranges between 0 and 1).
- Separation of the overall dataset into 7 specific datasets (datasets for France, Spain, Germany, France+Germany, France+Spain, Germany+Spain, and France+Germany+Spain).
- Pie chart visualization to show the imbalance in each dataset.
- Application of the RUS, SMOTE, SMOTE+ENN, and SMOTE+TOMEK methods to each dataset.
- Application of the XGBoost algorithm.
- Application of the XGBoost, Random Forest, SVM, LogisticRegression, Naive Bayes, Decision Tree, and KNN prediction algorithms to each dataset resulting from the resampling techniques.
- Evaluation of the obtained results for each algorithm+resampling technique+dataset combination based on the metrics.

## List of Metrics ⚡
- Precision
- Recall
- Accuracy
- F1-score
- Area under the ROC curve (AUC)

## Steps to Execute the Application 📈
1. Clone the project.
2. Install the libraries located in the "paquetes.sh" file.
3. Choose Python as the interpretation language.
4. Select a dataset (refer to the .py file name) and execute it to obtain the results of the RUS, SMOTE, SMOTE+ENN, and SMOTE+TOMEK techniques for that dataset.
5. To execute in PyCharm, go to Run > Run (Alt+Shift+F10) and select the default "Run Configuration".
6. The files starting with "Dataset" are derived from the main project "PrototipoGrupo2".
7. In "prototipogrupo2", data preparation and dataset separation are performed according to the countries, and pie charts showing the imbalanced data are generated.
8. In the files starting with "Dataset", the resampling methods are applied, and for each method applied in each scenario, a bar chart is displayed showing the version before and after applying the method.

## Contributors🤝
- Kevin Chávez
- Kevin Humareda
- Pedro Shiguihara
