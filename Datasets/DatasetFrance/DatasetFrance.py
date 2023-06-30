from PrototipoGrupo2 import dfFrance
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Dataframes con columna Churn y sin Churn
X_France = dfFrance.drop('Churn', axis=1)
y_France = dfFrance['Churn']

#BALANCEO DE CLASES

X_trainFrance, X_testFrance, y_trainFrance, y_testFrance = train_test_split(X_France, y_France, test_size= 0.3, random_state= 42)
#print(X_train.shape, X_test.shape)

X_validFrance, X_testFrance, y_validFrance, y_testFrance = train_test_split(X_testFrance, y_testFrance, test_size=0.5, random_state = 42)
print(X_trainFrance.shape, X_testFrance.shape, X_validFrance.shape)

#Aplicación de la tecnica de submuestreo RUS
rus = RandomUnderSampler(random_state=0)
X_rusFrance, y_rusFrance = rus.fit_resample(X_trainFrance, y_trainFrance)

print(f'''Cambio de X antes de RUS - France: {X_trainFrance.shape}
Cambio de X después de RUS - France: {X_rusFrance.shape}''')

print(' Balance positivo y negativo de la clase con RUS - France (%):')
print(y_rusFrance.value_counts(normalize = True)*100)

sns.countplot(x = y_rusFrance)
#plt.title(" Datos Balanceados RUS - France")
#plt.show()
########## print(X_rusFrance)

#Aplicación de la tecnica de sobremuestreo SMOTE
sm = SMOTE(random_state=42)
X_smoteFrance, y_smoteFrance = sm.fit_resample(X_trainFrance, y_trainFrance)

print(f'''Cambio de X antes de SMOTE - France: {X_trainFrance.shape}
Cambio de X después de SMOTE - France: {X_smoteFrance.shape}''')

print(' Balance positivo y negativo de la clase con SMOTE - France (%):')
print(y_smoteFrance.value_counts(normalize = True)*100)

sns.countplot(x = y_smoteFrance)
#plt.title(" Datos Balanceados SMOTE - France")
#plt.show()

########## print(X_smoteFrance)

#Aplicación de la tecnica de remuestreo SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_smote_ennFrance, y_smote_ennFrance = smote_enn.fit_resample(X_trainFrance, y_trainFrance)
print(f'''Cambio de X antes de SMOTEENN - France: {X_trainFrance.shape}
Cambio de X después de SMOTEENN - France: {X_smote_ennFrance.shape}''')

print(' Balance positivo y negativo de la clase con SMOTEENN - France (%):')
print(y_smote_ennFrance.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_ennFrance)
#plt.title(" Datos Balanceados SMOTEENN - France")
#plt.show()

########## print(X_smote_ennFrance)

#Aplicación de la tecnica de remuestreo SMOTETOMEK
smote_tomek = SMOTETomek(random_state=0)
X_smote_tomekFrance, y_smote_tomekFrance = smote_tomek.fit_resample(X_trainFrance, y_trainFrance)
print(f'''Cambio de X antes de SMOTETOMEK - France: {X_trainFrance.shape}
Cambio de X después de SMOTETOMEK - France: {X_smote_tomekFrance.shape}''')

print(' Balance positivo y negativo de la clase con SMOTETOMEK - France (%):')
print(y_smote_tomekFrance.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_tomekFrance)
#plt.title(" Datos Balanceados SMOTETOMEK - France")
#plt.show()

plt.close('all')