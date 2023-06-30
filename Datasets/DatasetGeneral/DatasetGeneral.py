from PrototipoGrupo2 import dfGeneral
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Dataframes con columna Churn y sin Churn
X_General = dfGeneral.drop('Churn', axis=1)
y_General = dfGeneral['Churn']

#BALANCEO DE CLASES

X_trainGeneral, X_testGeneral, y_trainGeneral, y_testGeneral = train_test_split(X_General, y_General, test_size= 0.3, random_state= 42)
#print(X_train.shape, X_test.shape)

X_validGeneral, X_testGeneral, y_validGeneral, y_testGeneral = train_test_split(X_testGeneral, y_testGeneral, test_size=0.5, random_state = 42)
print(X_trainGeneral.shape, X_testGeneral.shape, X_validGeneral.shape)

#Aplicación de la tecnica de submuestreo RUS
rus = RandomUnderSampler(random_state=0)
X_rusGeneral, y_rusGeneral = rus.fit_resample(X_trainGeneral, y_trainGeneral)

print(f'''Cambio de X antes de RUS - General: {X_trainGeneral.shape}
Cambio de X después de RUS - General: {X_rusGeneral.shape}''')

print(' Balance positivo y negativo de la clase con RUS - General (%):')
print(y_rusGeneral.value_counts(normalize = True)*100)

sns.countplot(x = y_rusGeneral)
#plt.title(" Datos Balanceados RUS - General")
#plt.show()
########## print(X_rusGeneral)

#Aplicación de la tecnica de sobremuestreo SMOTE
sm = SMOTE(random_state=42)
X_smoteGeneral, y_smoteGeneral = sm.fit_resample(X_trainGeneral, y_trainGeneral)

print(f'''Cambio de X antes de SMOTE - General: {X_trainGeneral.shape}
Cambio de X después de SMOTE - General: {X_smoteGeneral.shape}''')

print(' Balance positivo y negativo de la clase con SMOTE - General (%):')
print(y_smoteGeneral.value_counts(normalize = True)*100)

sns.countplot(x = y_smoteGeneral)
#plt.title(" Datos Balanceados SMOTE - General")
#plt.show()

########## print(X_smoteGeneral)

#Aplicación de la tecnica de remuestreo SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_smote_ennGeneral, y_smote_ennGeneral = smote_enn.fit_resample(X_trainGeneral, y_trainGeneral)
print(f'''Cambio de X antes de SMOTEENN - General: {X_trainGeneral.shape}
Cambio de X después de SMOTEENN - General: {X_smote_ennGeneral.shape}''')

print(' Balance positivo y negativo de la clase con SMOTEENN - General (%):')
print(y_smote_ennGeneral.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_ennGeneral)
#plt.title(" Datos Balanceados SMOTEENN - General")
#plt.show()

########## print(X_smote_ennGeneral)

#Aplicación de la tecnica de remuestreo SMOTETOMEK
smote_tomek = SMOTETomek(random_state=0)
X_smote_tomekGeneral, y_smote_tomekGeneral = smote_tomek.fit_resample(X_trainGeneral, y_trainGeneral)
print(f'''Cambio de X antes de SMOTETOMEK - General: {X_trainGeneral.shape}
Cambio de X después de SMOTETOMEK - General: {X_smote_tomekGeneral.shape}''')

print(' Balance positivo y negativo de la clase con SMOTETOMEK - General (%):')
print(y_smote_tomekGeneral.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_tomekGeneral)
#plt.title(" Datos Balanceados SMOTETOMEK - General")
#plt.show()

########## print(X_smote_tomekGeneral)
plt.close('all')