from PrototipoGrupo2 import dfSpainFrance
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Dataframes con columna Churn y sin Churn
X_SpainFrance = dfSpainFrance.drop('Churn', axis=1)
y_SpainFrance = dfSpainFrance['Churn']

#BALANCEO DE CLASES

X_trainSpainFrance, X_testSpainFrance, y_trainSpainFrance, y_testSpainFrance = train_test_split(X_SpainFrance, y_SpainFrance, test_size= 0.3, random_state= 42)
#print(X_train.shape, X_test.shape)

X_validSpainFrance, X_testSpainFrance, y_validSpainFrance, y_testSpainFrance = train_test_split(X_testSpainFrance, y_testSpainFrance, test_size=0.5, random_state = 42)
print(X_trainSpainFrance.shape, X_testSpainFrance.shape, X_validSpainFrance.shape)

#Aplicación de la tecnica de submuestreo RUS
rus = RandomUnderSampler(random_state=0)
X_rusSpainFrance, y_rusSpainFrance = rus.fit_resample(X_trainSpainFrance, y_trainSpainFrance)

print(f'''Cambio de X antes de RUS - SpainFrance: {X_trainSpainFrance.shape}
Cambio de X después de RUS - SpainFrance: {X_rusSpainFrance.shape}''')

print(' Balance positivo y negativo de la clase con RUS - SpainFrance (%):')
print(y_rusSpainFrance.value_counts(normalize = True)*100)

sns.countplot(x = y_rusSpainFrance)
#plt.title(" Datos Balanceados RUS - SpainFrance")
#plt.show()
########## print(X_rusSpainFrance)

#Aplicación de la tecnica de sobremuestreo SMOTE
sm = SMOTE(random_state=42)
X_smoteSpainFrance, y_smoteSpainFrance = sm.fit_resample(X_trainSpainFrance, y_trainSpainFrance)

print(f'''Cambio de X antes de SMOTE - SpainFrance: {X_trainSpainFrance.shape}
Cambio de X después de SMOTE - SpainFrance: {X_smoteSpainFrance.shape}''')

print(' Balance positivo y negativo de la clase con SMOTE - SpainFrance (%):')
print(y_smoteSpainFrance.value_counts(normalize = True)*100)

sns.countplot(x = y_smoteSpainFrance)
#plt.title(" Datos Balanceados SMOTE - SpainFrance")
#plt.show()

########## print(X_smoteSpainFrance)

#Aplicación de la tecnica de remuestreo SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_smote_ennSpainFrance, y_smote_ennSpainFrance = smote_enn.fit_resample(X_trainSpainFrance, y_trainSpainFrance)
print(f'''Cambio de X antes de SMOTEENN - SpainFrance: {X_trainSpainFrance.shape}
Cambio de X después de SMOTEENN - SpainFrance: {X_smote_ennSpainFrance.shape}''')

print(' Balance positivo y negativo de la clase con SMOTEENN - SpainFrance (%):')
print(y_smote_ennSpainFrance.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_ennSpainFrance)
#plt.title(" Datos Balanceados SMOTEENN - SpainFrance")
#plt.show()

########## print(X_smote_ennSpainFrance)

#Aplicación de la tecnica de remuestreo SMOTETOMEK
smote_tomek = SMOTETomek(random_state=0)
X_smote_tomekSpainFrance, y_smote_tomekSpainFrance = smote_tomek.fit_resample(X_trainSpainFrance, y_trainSpainFrance)
print(f'''Cambio de X antes de SMOTETOMEK - SpainFrance: {X_trainSpainFrance.shape}
Cambio de X después de SMOTETOMEK - SpainFrance: {X_smote_tomekSpainFrance.shape}''')

print(' Balance positivo y negativo de la clase con SMOTETOMEK - SpainFrance (%):')
print(y_smote_tomekSpainFrance.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_tomekSpainFrance)
#plt.title(" Datos Balanceados SMOTETOMEK - SpainFrance")
#plt.show()

plt.close('all')
