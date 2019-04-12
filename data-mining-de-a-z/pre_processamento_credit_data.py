#%%#
#import os
#print(os.getcwd())
#workdir: /home/peluzio/Documentos/Mestrado Data Mining/data-mining-de-a-z
#workdir: C:\Users\hga\data-mining\data-mining-de-a-z
import os
os.chdir('C:\Users\hga\data-mining\data-mining-de-a-z')
import pandas as pd
base = pd.read_csv('credit_data.csv')

#clientid=ID do cliente (variavel nominal)
#income=salario         (variavel continua)
#age=idade              (variavel continua)
#loan=emprestimo        (variavel continua)
#default=meta/classe    (variavel discreta)

#%%#
# Trabalhar os dados
# existe idade negativa, repare no comando abaixo
base.describe()
base.head()

#%%#
#LOCaliza dentro da age na coluna age valores menores que 0
base.loc[base['age'] < 0]

#%%#
# temos 4 tecnicas para utilizar nesse caso
# 1 - apagar a coluna da idade (nao recomendavel)
#base.drop('age', 1, inplace=True) # parameters: nome da coluna, apagar coluna toda, rodar na propria base
# 2 - apagar somente os registros com o problema
#base.drop(base[base.age < 0].index, inplace = True)
# 3 - preencher os valores manualmente
# NAO VIAVEL
# 4- Preencher os valores com a media
base.mean() #Media de todos os campos
base['age'].mean() #Media da idade
#%%#
#Pegando a media das idades maiores que 0
base['age'][base.age > 0].mean()
# SUbtituindo valores de idade menores que 0 pela media das maiores
base.loc[base.age < 0, 'age'] = 40.92
#%%#
#TEstes
#base.loc[:9]
#base.loc[[1,2,3,4,5]]
#base.loc[2:6]
#base['age']
#base.loc[base.age, 'age']
#%%#
pd.isnull(base['age'])              #Verificando valores nulos
base.loc[pd.isnull(base['age'])]    #Verificando simplificado

#%%#
#Pegando apenas colunas dos previdores
#income, age, loan
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

#%%# DEPRECATED
#Biblioteca para preprocessar os dados nulos
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imputer = Imputer.fit(previsores[:, 0:3])
#previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

#%%#  Imputer esta deprecated, entao a gente usou esse aqui
#Biblioteca para preprocessar os dados nulos
import numpy as np
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])
#%%# Aprendendo usar loc iloc
previsores
#previsores.iloc[previsores[0] = 'NaN'].values
#%%#
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)









