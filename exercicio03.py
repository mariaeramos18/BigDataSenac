import pandas as pd

#obter os dados
dados = pd.read_csv('https://www.ispdados.rj.gov.br/Arquivos/UppEvolucaoMensalDeTitulos.csv' \
                    , sep=';', encoding='ISO-8859-1')

#filtrar UPP VK
dados = dados[dados['upp'] == 'Vila Kennedy']

#delimitar colunas para correlação
dados_correlacao = dados.iloc[:,4:23].join(dados.iloc[:,24:42])

#correlação
correlacao = dados_correlacao.corr()

#filtrar coeficiente de correlação >= 0.8
correlacao = correlacao[correlacao >= 0.8]

#exportar para o Excel
endereco = r'C:\uc3_bigdata\AULA04'
correlacao.to_excel(endereco + r'\correlacaoUPP.xlsx')