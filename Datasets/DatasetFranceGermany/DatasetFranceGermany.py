from PrototipoGrupo2 import dfFranceGermany
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Dataframes con columna Churn y sin Churn
X_FranceGermany = dfFranceGermany.drop('Churn', axis=1)
y_FranceGermany = dfFranceGermany['Churn']

#BALANCEO DE CLASES

X_trainFranceGermany, X_testFranceGermany, y_trainFranceGermany, y_testFranceGermany = train_test_split(X_FranceGermany, y_FranceGermany, test_size= 0.3, random_state= 42)
#print(X_train.shape, X_test.shape)

X_validFranceGermany, X_testFranceGermany, y_validFranceGermany, y_testFranceGermany = train_test_split(X_testFranceGermany, y_testFranceGermany, test_size=0.5, random_state = 42)
print(X_trainFranceGermany.shape, X_testFranceGermany.shape, X_validFranceGermany.shape)

#Aplicación de la tecnica de submuestreo RUS
rus = RandomUnderSampler(random_state=0)
X_rusFranceGermany, y_rusFranceGermany = rus.fit_resample(X_trainFranceGermany, y_trainFranceGermany)

print(f'''Cambio de X antes de RUS - FranceGermany: {X_trainFranceGermany.shape}
Cambio de X después de RUS - FranceGermany: {X_rusFranceGermany.shape}''')

print(' Balance positivo y negativo de la clase con RUS - FranceGermany (%):')
print(y_rusFranceGermany.value_counts(normalize = True)*100)

sns.countplot(x = y_rusFranceGermany)
#plt.title(" Datos Balanceados RUS - FranceGermany")
#plt.show()
########## print(X_rusFranceGermany)

#Aplicación de la tecnica de sobremuestreo SMOTE
sm = SMOTE(random_state=42)
X_smoteFranceGermany, y_smoteFranceGermany = sm.fit_resample(X_trainFranceGermany, y_trainFranceGermany)

print(f'''Cambio de X antes de SMOTE - FranceGermany: {X_trainFranceGermany.shape}
Cambio de X después de SMOTE - FranceGermany: {X_smoteFranceGermany.shape}''')

print(' Balance positivo y negativo de la clase con SMOTE - FranceGermany (%):')
print(y_smoteFranceGermany.value_counts(normalize = True)*100)

sns.countplot(x = y_smoteFranceGermany)
#plt.title(" Datos Balanceados SMOTE - FranceGermany")
#plt.show()

########## print(X_smoteFranceGermany)

#Aplicación de la tecnica de remuestreo SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_smote_ennFranceGermany, y_smote_ennFranceGermany = smote_enn.fit_resample(X_trainFranceGermany, y_trainFranceGermany)
print(f'''Cambio de X antes de SMOTEENN - FranceGermany: {X_trainFranceGermany.shape}
Cambio de X después de SMOTEENN - FranceGermany: {X_smote_ennFranceGermany.shape}''')

print(' Balance positivo y negativo de la clase con SMOTEENN - FranceGermany (%):')
print(y_smote_ennFranceGermany.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_ennFranceGermany)
#plt.title(" Datos Balanceados SMOTEENN - FranceGermany")
#plt.show()

########## print(X_smote_ennFranceGermany)

#Aplicación de la tecnica de remuestreo SMOTETOMEK
smote_tomek = SMOTETomek(random_state=0)
X_smote_tomekFranceGermany, y_smote_tomekFranceGermany = smote_tomek.fit_resample(X_trainFranceGermany, y_trainFranceGermany)
print(f'''Cambio de X antes de SMOTETOMEK - FranceGermany: {X_trainFranceGermany.shape}
Cambio de X después de SMOTETOMEK - FranceGermany: {X_smote_tomekFranceGermany.shape}''')

print(' Balance positivo y negativo de la clase con SMOTETOMEK - FranceGermany (%):')
print(y_smote_tomekFranceGermany.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_tomekFranceGermany)
#plt.title(" Datos Balanceados SMOTETOMEK - FranceGermany")
#plt.show()

########## print(X_smote_tomekFranceGermany)
plt.close('all')