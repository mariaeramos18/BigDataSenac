#Importando bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#endereço do conjunto de dados
endereco = r'C:\Users\alexa\Desktop\Curso_Big_Data'

#df com os dados de salário
salario = pd.read_excel(endereco + r'\dados_salarios.xlsx')

#definir as variáveis
x = ['Salário','Tempo de Casa (meses)','Idade']

#Normalizar os dados
#variável de normalização
scaler = StandardScaler()

#criar um conjunto de dados normalizado
salarioNormalizado = scaler.fit_transform(salario[[x[0],x[1],x[2]]])

'''
MÉTODO DE COTOVELO: 
IDENTIFICAR A QUANTIDADE DE CLUSTERS
É PARA REALIZAR SOMENTE 1 VEZ OU QDO HOUVER NECESSIDADE
'''
'''#criando a lista da inércia
inercia = []

#criando o intervalo de valores k (cluster)
valores_k = range(1,10)

#aplicar o método de cotovelo
for k in valores_k:
    #incializar o modelo kmeans com o 'k' cluster
    kmeans = KMeans(n_clusters=k)

    #ajustar o modelo
    kmeans.fit(salarioNormalizado)

    #adicionar a inércia na lista
    inercia.append(kmeans.inertia_)

#plotar o gráfico
import matplotlib.pyplot as plt

plt.plot(valores_k,inercia)
plt.xlabel('Qtde de clusters (k)')
plt.ylabel('Inércia')
plt.title('Método de cotovelo')

plt.show()'''

#k clusters
#k = 3 (segundo a avaliação do gráfico do método de cotovelo)
kmeans = KMeans(n_clusters=3)

#treinar o modelo
kmeans.fit(salarioNormalizado)

#adicionar os cluster ao df original (salario)
salario['cluster'] = kmeans.labels_

#consolidar os dados do cluster
df_salario_cluster_media = salario.groupby('cluster') \
                            .mean().reset_index() \
                            .sort_values(x[0], ascending=False)

#Plotar em um gráfico
fig, ax = plt.subplots(figsize=(15,10), \
                       subplot_kw={'projection':'3d'})

tempo_casa = salario[x[1]]
idade = salario[x[2]]
vlrSalario = salario[x[0]]
cluster = salario['cluster']

ax.scatter(tempo_casa,idade,vlrSalario,c=cluster,cmap='viridis')

ax.set_xlabel('Tempo de casa(meses)')
ax.set_ylabel('Idade')
ax.set_zlabel('Salario')
plt.title('Clusters de salário')

#adicionar uma barra de cores
cbar = plt.colorbar(ax.scatter(tempo_casa,idade,\
                               vlrSalario,c=cluster,\
                               cmap='viridis'))
cbar.set_ticks(cluster.unique())

plt.tight_layout()
plt.show()