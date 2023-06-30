from PrototipoGrupo2 import dfSpain
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Dataframes con columna Churn y sin Churn
X_Spain = dfSpain.drop('Churn', axis=1)
y_Spain = dfSpain['Churn']

#BALANCEO DE CLASES

X_trainSpain, X_testSpain, y_trainSpain, y_testSpain = train_test_split(X_Spain, y_Spain, test_size= 0.3, random_state= 42)
#print(X_train.shape, X_test.shape)

X_validSpain, X_testSpain, y_validSpain, y_testSpain = train_test_split(X_testSpain, y_testSpain, test_size=0.5, random_state = 42)
print(X_trainSpain.shape, X_testSpain.shape, X_validSpain.shape)

#Aplicación de la tecnica de submuestreo RUS
rus = RandomUnderSampler(random_state=0)
X_rusSpain, y_rusSpain = rus.fit_resample(X_trainSpain, y_trainSpain)

print(f'''Cambio de X antes de RUS - Spain: {X_trainSpain.shape}
Cambio de X después de RUS - Spain: {X_rusSpain.shape}''')

print(' Balance positivo y negativo de la clase con RUS - Spain (%):')
print(y_rusSpain.value_counts(normalize = True)*100)

sns.countplot(x = y_rusSpain)
#plt.title(" Datos Balanceados RUS - Spain")
#plt.show()
########## print(X_rusSpain)

#Aplicación de la tecnica de sobremuestreo SMOTE
sm = SMOTE(random_state=42)
X_smoteSpain, y_smoteSpain = sm.fit_resample(X_trainSpain, y_trainSpain)

print(f'''Cambio de X antes de SMOTE - Spain: {X_trainSpain.shape}
Cambio de X después de SMOTE - Spain: {X_smoteSpain.shape}''')

print(' Balance positivo y negativo de la clase con SMOTE - Spain (%):')
print(y_smoteSpain.value_counts(normalize = True)*100)

sns.countplot(x = y_smoteSpain)
#plt.title(" Datos Balanceados SMOTE - Spain")
#plt.show()

########## print(X_smoteSpain)

#Aplicación de la tecnica de remuestreo SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_smote_ennSpain, y_smote_ennSpain = smote_enn.fit_resample(X_trainSpain, y_trainSpain)
print(f'''Cambio de X antes de SMOTEENN - Spain: {X_trainSpain.shape}
Cambio de X después de SMOTEENN - Spain: {X_smote_ennSpain.shape}''')

print(' Balance positivo y negativo de la clase con SMOTEENN - Spain (%):')
print(y_smote_ennSpain.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_ennSpain)
#plt.title(" Datos Balanceados SMOTEENN - Spain")
#plt.show()

########## print(X_smote_ennSpain)

#Aplicación de la tecnica de remuestreo SMOTETOMEK
smote_tomek = SMOTETomek(random_state=0)
X_smote_tomekSpain, y_smote_tomekSpain = smote_tomek.fit_resample(X_trainSpain, y_trainSpain)
print(f'''Cambio de X antes de SMOTETOMEK - Spain: {X_trainSpain.shape}
Cambio de X después de SMOTETOMEK - Spain: {X_smote_tomekSpain.shape}''')

print(' Balance positivo y negativo de la clase con SMOTETOMEK - Spain (%):')
print(y_smote_tomekSpain.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_tomekSpain)
#plt.title(" Datos Balanceados SMOTETOMEK - Spain")
#plt.show()

plt.close('all')