from PrototipoGrupo2 import dfGermany
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Dataframes con columna Churn y sin Churn
X_Germany = dfGermany.drop('Churn', axis=1)
y_Germany = dfGermany['Churn']

#BALANCEO DE CLASES

X_trainGermany, X_testGermany, y_trainGermany, y_testGermany = train_test_split(X_Germany, y_Germany, test_size= 0.3, random_state= 42)
#print(X_train.shape, X_test.shape)

X_validGermany, X_testGermany, y_validGermany, y_testGermany = train_test_split(X_testGermany, y_testGermany, test_size=0.5, random_state = 42)
print(X_trainGermany.shape, X_testGermany.shape, X_validGermany.shape)

#Aplicación de la tecnica de submuestreo RUS
rus = RandomUnderSampler(random_state=0)
X_rusGermany, y_rusGermany = rus.fit_resample(X_trainGermany, y_trainGermany)

print(f'''Cambio de X antes de RUS - Germany: {X_trainGermany.shape}
Cambio de X después de RUS - Germany: {X_rusGermany.shape}''')

print(' Balance positivo y negativo de la clase con RUS - Germany (%):')
print(y_rusGermany.value_counts(normalize = True)*100)

sns.countplot(x = y_rusGermany)
#plt.title(" Datos Balanceados RUS - Germany")
#plt.show()
########## print(X_rusGermany)

#Aplicación de la tecnica de sobremuestreo SMOTE
sm = SMOTE(random_state=42)
X_smoteGermany, y_smoteGermany = sm.fit_resample(X_Germany, y_Germany)

print(f'''Cambio de X antes de SMOTE - Germany: {X_Germany.shape}
Cambio de X después de SMOTE - Germany: {X_smoteGermany.shape}''')

print(' Balance positivo y negativo de la clase con SMOTE - Germany (%):')
print(y_smoteGermany.value_counts(normalize = True)*100)

sns.countplot(x = y_smoteGermany)
#plt.title(" Datos Balanceados SMOTE - Germany")
#plt.show()

########## print(X_smoteGermany)

#Aplicación de la tecnica de remuestreo SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_smote_ennGermany, y_smote_ennGermany = smote_enn.fit_resample(X_Germany, y_Germany)
print(f'''Cambio de X antes de SMOTEENN - Germany: {X_Germany.shape}
Cambio de X después de SMOTEENN - Germany: {X_smote_ennGermany.shape}''')

print(' Balance positivo y negativo de la clase con SMOTEENN - Germany (%):')
print(y_smote_ennGermany.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_ennGermany)
#plt.title(" Datos Balanceados SMOTEENN - Germany")
#plt.show()

########## print(X_smote_ennGermany)

#Aplicación de la tecnica de remuestreo SMOTETOMEK
smote_tomek = SMOTETomek(random_state=0)
X_smote_tomekGermany, y_smote_tomekGermany = smote_tomek.fit_resample(X_Germany, y_Germany)
print(f'''Cambio de X antes de SMOTETOMEK - Germany: {X_Germany.shape}
Cambio de X después de SMOTETOMEK - Germany: {X_smote_tomekGermany.shape}''')

print(' Balance positivo y negativo de la clase con SMOTETOMEK - Germany (%):')
print(y_smote_tomekGermany.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_tomekGermany)
plt.title(" Datos Balanceados SMOTETOMEK - Germany")
#plt.show()
#plt.close('all')
########## print(X_smote_tomekGermany)

