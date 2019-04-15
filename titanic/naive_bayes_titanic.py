import pandas as pd

#base = pd.concat([df_train,df_test],keys=['train','test'])
base = pd.read_csv('train.csv')
#train = pd.read_csv('train.csv')
#test = pd.read_csv('test.csv')

base[base.Embarked.isnull()]

import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(base[:, 0:5])
base[:, 5] = imputer.transform(base[:, 0:5])
        
#Transformacao de categorico para discreto (Obrigatorio)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 4] = labelencoder_previsores.fit_transform(previsores[:, 4])
previsores[:, 11] = labelencoder_previsores.fit_transform(previsores[:, 11])


previsores = base.iloc[:, [2,4,5,6,7,9,11]].values
classe = base.iloc[:, 1].values
#Criando as dummy variables
#onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
#previsores = onehotencoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

#Escalonamento (Se fizer escalonamento para as variaveis dummy, da resultado ruim) Sem escalonamento 
# nas variaveis dummy ai fica bom
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#previsores = scaler.fit_transform(previsores)
previsores[:, [2,10,11,12]] = scaler.fit_transform(previsores[:, [2,10,11,12]])
#scaler.fit_transform(X[:,[0,1]])


#Dividir a base de dados em treinamento e teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)