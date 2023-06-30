import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import seaborn as sns

# Leyendo Dataset
df = pd.read_csv('churn.csv')
# Imprimiendo dataset
# print(df)
# Información dataset
# df.info()
# print(df.dtypes)
# Cambio de columna "Exited" a "Churn"
df.rename(columns={'Exited': 'Churn'}, inplace=True)
# Gráfico desbalance de clases Churn
churn = df['Churn'].value_counts()
mylabels = ["No Desertores", "Desertores"]
plt.pie(churn, labels=mylabels, autopct='%1.1f%%', startangle=100, shadow=True)
plt.title('Porcentaje de clientes desertores Total')
#plt.show()

# Viendo datos únicos columna Geography
#print(df.Geography.unique())
# Viendo datos únicos columna Gender
#print(df.Gender.unique())

# Preparación de datos
# Cambio a valor numérico de las columnas Gender y Geography
##### Female = 0, Male = 1
df['Gender'] = [0 if x == 'Female' else 1 for x in df['Gender']]
##### France = 0, Germany = 1, Spain = 2
df['Geography'] = [0 if x == 'France' else 1 if x == 'Germany' else 2 for x in df['Geography']]

# Eliminación de las columnas innecesarias
to_drop = ['CustomerId', 'Surname']
df.drop(to_drop, axis=1, inplace=True)

# Ahora escalamos los datos, para que esten en el rango de 0 a 1 como porcentaje
##### 1. Escalamos un dataset
to_scale = [col for col in df.columns if df[col].max() > 1 and col != 'Geography']
mms = MinMaxScaler()
scaled = mms.fit_transform(df[to_scale])
scaled = pd.DataFrame(scaled, columns=to_scale)

##### 2. Reemplazamos el dataset en escala
for col in scaled:
    df[col] = scaled[col]

# Finalizamos visualizando los cambios
#print(df.head())

# Dividiendo dataset según el país
dfFrance = df[df["Geography"] == 0]
dfGermany = df[df["Geography"] == 1]
dfSpain = df[df["Geography"] == 2]
dfGeneral = df
dfFranceGermany = pd.concat([dfFrance, dfGermany])
dfGermanySpain = pd.concat([dfGermany, dfSpain])
dfSpainFrance = pd.concat([dfSpain, dfFrance])

# Nuevos dataset
#print(dfFrance)
#print(dfGermany)
#print(dfSpain)
#print(dfGeneral)
#print(dfFranceGermany)
#print(dfGermanySpain)
#print(dfSpainFrance)

# Gráfico desbalanceo por dataset Churn France
churn2 = dfFrance['Churn'].value_counts()
mylabels = ["No Desertores", "Desertores"]
#plt.pie(churn2, labels=mylabels, autopct='%1.1f%%', startangle=100, shadow=True)
#plt.title('Porcentaje de clientes desertores Francia')
#plt.show()

# Gráfico desbalanceo por dataset Churn Germany
churn3 = dfGermany['Churn'].value_counts()
mylabels = ["No Desertores", "Desertores"]
#plt.pie(churn3, labels=mylabels, autopct='%1.1f%%', startangle=100, shadow=True)
#plt.title('Porcentaje de clientes desertores Alemania')
#plt.show()


# Gráfico desbalanceo por dataset Churn Spain
churn4 = dfSpain['Churn'].value_counts()
mylabels = ["No Desertores", "Desertores"]
#plt.pie(churn4, labels=mylabels, autopct='%1.1f%%', startangle=100, shadow=True)
#plt.title('Porcentaje de clientes desertores España')
#plt.show()

# Gráfico desbalanceo por dataset Churn FranceGermany
churn5 = dfFranceGermany['Churn'].value_counts()
mylabels = ["No Desertores", "Desertores"]
#plt.pie(churn5, labels=mylabels, autopct='%1.1f%%', startangle=100, shadow=True)
#plt.title('Porcentaje de clientes desertores Francia y Alemania')
#plt.show()

# Gráfico desbalanceo por dataset Churn GermanySpain
churn6 = dfGermanySpain['Churn'].value_counts()
mylabels = ["No Desertores", "Desertores"]
#plt.pie(churn6, labels=mylabels, autopct='%1.1f%%', startangle=100, shadow=True)
#plt.title('Porcentaje de clientes desertores Alemania y España')
#plt.show()

# Gráfico desbalanceo por dataset Churn SpainFrance
churn7 = dfSpainFrance['Churn'].value_counts()
mylabels = ["No Desertores", "Desertores"]
#plt.pie(churn7, labels=mylabels, autopct='%1.1f%%', startangle=100, shadow=True)
#plt.title('Porcentaje de clientes desertores España y Francia')
#plt.show()
