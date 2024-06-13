import pandas as pd
import numpy as np

#obter os dados
dados = pd.read_csv('https://www.ispdados.rj.gov.br/Arquivos/BaseDPEvolucaoMensalCisp.csv' \
                    , sep=';', encoding='ISO-8859-1')

#Filtrar ano
dados = dados[dados['ano'] <= 2023]

#correlação ameaça x lesao corporal dolosa
correlacao = dados['ameaca'].corr(dados['lesao_corp_dolosa'])

#variável dependente
y_lesao = dados['lesao_corp_dolosa'].values

#variável independente
x_ameaca = dados['ameaca'].values

#importar a classe de treino e teste
from sklearn.model_selection import train_test_split

#dividir os dados de treino e teste
X_ameaca_train, X_ameaca_test, y_lesao_train, y_lesao_test = \
train_test_split(
    x_ameaca,
    y_lesao,
    test_size=0.2,
    shuffle=False
)

#importar a classe de regressão linear
from sklearn.linear_model import LinearRegression

#Modelo linear
modelo = LinearRegression()

#Treinando o modelo
#dados de treino
#y = ax + b
modelo.fit(X_ameaca_train.reshape(-1,1), y_lesao_train)

#Verificar a qualidade do treino: 0 a 1
#dados de teste
score = modelo.score(X_ameaca_test.reshape(-1,1), y_lesao_test)
#print('y = ', modelo.coef_[0],'*', X_ameaca_test, '+', modelo.intercept_)
#print(score)

#Testar a análise preditiva
y_predicao = modelo.predict(X_ameaca_test.reshape(-1,1))

#verificar o custo das qtdes 12000, 15000 e 20000
qtd_ameaca_pred = np.array([300,450,550,700,800,1000])

#estimar o custo de produção
lesao_pred = modelo.predict(qtd_ameaca_pred.reshape(-1,1))

#Inserir dados de predição em um dataframe
#Para utilizá-los na tabela do painel do visualização
df_lesao_pred = pd.DataFrame({'Qtde. ameaças':qtd_ameaca_pred, \
                              'Lesão corp. dolosa':lesao_pred})

#RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_lesao_test, y_predicao))

#VISUALIZAÇÃO DE DADOS
import seaborn as sns
import matplotlib.pyplot as plt

#criando o painel
plt.subplots(1,2, figsize=(15,10))

#Regressão linear
plt.subplot(1,2,1)
#gráfio de regressão linear do seaborn
#sns.regplot(x=X_ameaca_test, y=y_lesao_test, color='red')
sns.scatterplot(x=X_ameaca_test,y=y_lesao_test, color='black')
sns.lineplot(x=X_ameaca_test,y=y_predicao, color='blue')
plt.xlabel('Ameaça')
plt.ylabel('Lesão corporal dolosa')
plt.title(f'Regressão linear - Ameaça x Lesão corp. dolosa.\n \
          Score: {score:.2f}\n\
          Correlação: {correlacao:.2f}')

#Tabela com os dados simulados
plt.subplot(1,2,2)
table = plt.table(
    cellText=df_lesao_pred.values.astype(int),
    colLabels=df_lesao_pred.columns,
    loc='center',
    cellLoc='center'
)
table.set_fontsize(12)
table.scale(1, 1.5)
plt.title(f'Previsão de lesão corporal. RMSE: {round(rmse,0)}')
plt.axis('off')

plt.tight_layout()
plt.show()