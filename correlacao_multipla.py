import pandas as pd
import numpy as np

#obter os dados
dados = pd.read_csv('https://www.ispdados.rj.gov.br/Arquivos/BaseDPEvolucaoMensalCisp.csv' \
                    , sep=';', encoding='ISO-8859-1')

#Filtrar ano
dados = dados[dados['ano'] <= 2023]

#manter somente as colunas dos registos de ocorrências
dados_correlacao = dados.iloc[:,9:61]

#correlacao
correlacao = dados_correlacao.corr()

#filtrar correlação acima de 0.85
#BAIXAR O DICIONÁRIO DE DADOS
correlacao = correlacao[correlacao >= 0.8]

#exportar para o Excel
endereco = r'C:\uc3_bigdata\AULA04'
correlacao.to_excel(endereco + r'\correlacao.xlsx')