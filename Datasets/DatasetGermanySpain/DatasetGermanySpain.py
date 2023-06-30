from PrototipoGrupo2 import dfGermanySpain
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

#Dataframes con columna Churn y sin Churn
X_GermanySpain = dfGermanySpain.drop('Churn', axis=1)
y_GermanySpain = dfGermanySpain['Churn']

#BALANCEO DE CLASES

X_trainGermanySpain, X_testGermanySpain, y_trainGermanySpain, y_testGermanySpain = train_test_split(X_GermanySpain, y_GermanySpain, test_size= 0.3, random_state= 42)
#print(X_train.shape, X_test.shape)

X_validGermanySpain, X_testGermanySpain, y_validGermanySpain, y_testGermanySpain = train_test_split(X_testGermanySpain, y_testGermanySpain, test_size=0.5, random_state = 42)
print(X_trainGermanySpain.shape, X_testGermanySpain.shape, X_validGermanySpain.shape)

#Aplicación de la tecnica de submuestreo RUS
rus = RandomUnderSampler(random_state=0)
X_rusGermanySpain, y_rusGermanySpain = rus.fit_resample(X_trainGermanySpain, y_trainGermanySpain)

print(f'''Cambio de X antes de RUS - GermanySpain: {X_trainGermanySpain.shape}
Cambio de X después de RUS - GermanySpain: {X_rusGermanySpain.shape}''')

print(' Balance positivo y negativo de la clase con RUS - GermanySpain (%):')
print(y_rusGermanySpain.value_counts(normalize = True)*100)

sns.countplot(x = y_rusGermanySpain)
#plt.title(" Datos Balanceados RUS - GermanySpain")
#plt.show()
########## print(X_rusGermanySpain)

#Aplicación de la tecnica de sobremuestreo SMOTE
sm = SMOTE(random_state=42)
X_smoteGermanySpain, y_smoteGermanySpain = sm.fit_resample(X_trainGermanySpain, y_trainGermanySpain)

print(f'''Cambio de X antes de SMOTE - GermanySpain: {X_trainGermanySpain.shape}
Cambio de X después de SMOTE - GermanySpain: {X_smoteGermanySpain.shape}''')

print(' Balance positivo y negativo de la clase con SMOTE - GermanySpain (%):')
print(y_smoteGermanySpain.value_counts(normalize = True)*100)

sns.countplot(x = y_smoteGermanySpain)
#plt.title(" Datos Balanceados SMOTE - GermanySpain")
#plt.show()

########## print(X_smoteGermanySpain)

#Aplicación de la tecnica de remuestreo SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_smote_ennGermanySpain, y_smote_ennGermanySpain = smote_enn.fit_resample(X_trainGermanySpain, y_trainGermanySpain)
print(f'''Cambio de X antes de SMOTEENN - GermanySpain: {X_trainGermanySpain.shape}
Cambio de X después de SMOTEENN - GermanySpain: {X_smote_ennGermanySpain.shape}''')

print(' Balance positivo y negativo de la clase con SMOTEENN - GermanySpain (%):')
print(y_smote_ennGermanySpain.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_ennGermanySpain)
#plt.title(" Datos Balanceados SMOTEENN - GermanySpain")
#plt.show()

########## print(X_smote_ennGermanySpain)

#Aplicación de la tecnica de remuestreo SMOTETOMEK
smote_tomek = SMOTETomek(random_state=0)
X_smote_tomekGermanySpain, y_smote_tomekGermanySpain = smote_tomek.fit_resample(X_trainGermanySpain, y_trainGermanySpain)
print(f'''Cambio de X antes de SMOTETOMEK - GermanySpain: {X_trainGermanySpain.shape}
Cambio de X después de SMOTETOMEK - GermanySpain: {X_smote_tomekGermanySpain.shape}''')

print(' Balance positivo y negativo de la clase con SMOTETOMEK - GermanySpain (%):')
print(y_smote_tomekGermanySpain.value_counts(normalize = True)*100)

sns.countplot(x = y_smote_tomekGermanySpain)
#plt.title(" Datos Balanceados SMOTETOMEK - GermanySpain")
#plt.show()

plt.close('all')