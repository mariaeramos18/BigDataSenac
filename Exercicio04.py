import pandas as pd
import numpy as np

#obter os dados
dados = pd.read_csv('https://www.ispdados.rj.gov.br/Arquivos/UppEvolucaoMensalDeTitulos.csv' \
                    , sep=';', encoding='ISO-8859-1')

#Filtrar UPP
dados = dados[dados['upp'] == 'Vila Kennedy']

#variável dependente
y = dados['roubo_veiculo'].values

#variável independente
x = dados[['roubo_transeunte','recuperacao_veiculos']].values

#importar a classe de treino e teste
from sklearn.model_selection import train_test_split

#dividir os dados de treino e teste
X_train, X_test, y_train, y_test = \
train_test_split(
    x,
    y,
    test_size=0.3,
    shuffle=False
)

#importar a classe de regressão linear
from sklearn.linear_model import LinearRegression

#Modelo linear
modelo = LinearRegression()

#Treinando o modelo
#dados de treino
#y = a1x1 + a2x2 + b
#rubo_veiculo = a1*roubo_transeunte + a2*recuperacao_veiculos + b
modelo.fit(X_train.reshape(-1,2), y_train)

#Verificar a qualidade do treino: 0 a 1
#dados de teste
score = modelo.score(X_test.reshape(-1,2), y_test)
print('Coeficente angular: ', modelo.coef_)
print('Coeficiente linear (Intercepto): ', modelo.intercept_)
print('Score: ', score)

#Realizar a análise preditiva no meu teste
y_predicao = modelo.predict(X_test.reshape(-1,2))

#simular as qtdes de roubo_transeunte e recuperacao_veiculos
qtd_pred = np.array([
                    [12,15],
                    [23,25],
                    [27,32]
                    ])

#estimar o roubo_veiculo
roubo_veiculos_pred = modelo.predict(qtd_pred)

print('Roubo veículos Pred: ', roubo_veiculos_pred)

'''#Inserir dados de predição em um dataframe
#Para utilizá-los na tabela do painel do visualização
df_lesao_pred = pd.DataFrame({'Qtde. ameaças':qtd_ameaca_pred, \
                              'Lesão corp. dolosa':lesao_pred})'''

#RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_predicao))

#VISUALIÇÃO DE DADOS
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15,10), subplot_kw={'projection': '3d'})

roubo_transeunte = X_test[:,0]
recuperacao_veiculos = X_test[:,1]

ax.scatter(roubo_transeunte,recuperacao_veiculos,y_predicao,color='blue')
ax.plot_trisurf(roubo_transeunte,recuperacao_veiculos,y_predicao, color='red', alpha=0.5)

ax.set_xlabel('Roubo transeunte')
ax.set_ylabel('Recup. de veículos')
ax.set_zlabel('Roubo de veículos Pred')
plt.title('Roubo de veículos Pred - Vila Kennedy')
plt.tight_layout()
plt.show()