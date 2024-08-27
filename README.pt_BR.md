 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]
  <!--  START HEADER  -->  

<br><br>


# <p align="center">  💎 Previsão de Valorização de Diamantes
#### <p align="center"> [![Sponsor Mindful AI Assistants](https://img.shields.io/badge/Sponsor-Mindful%20AI%20%20Assistants-brightgreen?logo=GitHub)](https://github.com/sponsors/Mindful-AI-Assistants)

<br>

https://github.com/user-attachments/assets/7eeb93e6-9d3f-41d0-b40a-3d17e28670a9

<br>

Este repositório contém um projeto Python para analisar um conjunto de dados de características de diamantes e prever seu preço usando técnicas de aprendizado de máquina.

## Sumário

- [Introdução](#introducao)
- [Conjunto de Dados](#conjunto-de-dados)
- [Metodologia](#metodologia)
- [Descobertas](#descobertas)
- [Análise de Recursos Numéricos](#analise-de-recursos-numericos)
- [Análise de Recursos Categóricos](#analise-de-recursos-categoricos)
- [Insights](#insights)
- [Recomendações](#recomendacoes)
- [Conclusão](#conclusao)
- [Estrutura de Arquivos](#estrutura-de-arquivos)
- [Começando a Clonar](#comecando)
- [Contribuindo](#contribuindo)
- [Comandos Git](#GitCommands)
- [Códigos e Relatório de Análise de Dados](#Report)
- [Acesse o Site do Streamlit](https://diamondsvalues.streamlit.app/)  <!-- - [ Tap here and teleport to the site]() 🇺🇸  --> 
- [Códigos QR](#QRCodes)
- [Nossa Equipe](#nossa-equipe)
- [Código de Conduta](#codigo-de-conduta)
- [Licença](#licenca)

  #

## Introdução 

Este projeto explora o fascinante mundo dos diamantes e busca prever seus preços com base em uma variedade de fatores. Nosso objetivo é descobrir as relações ocultas entre as características dos diamantes e seus valores, contribuindo para uma compreensão mais profunda do mercado de diamantes. 

O propósito desta analise preditiva é criar um site que defina o preço de um diamante com base em suas características: carat (quilate), cut (corte), color (cor), clarity (claridade), price (preço), depth (profundidade), table (tabela), x (comprimento), y (largura) e z (profundidade). Entretanto, em casos extremos onde é necessário fazer uma estimativa rápida do valor de um diamante, não é viável perder tempo definindo todas essas características. Por isso, é necessário realizar um estudo da base de dados para determinar quais são as características mínimas necessárias para estimar o preço de um diamante de forma precisa. Para implementar o projeto, é essencial avaliar como cada característica do diamante influencia seu preço. Isso requer descobrir como a variabilidade de uma característica pode afetar a vari- abilidade do preço. Portanto, o uso de estratégias estatísticas será crucial para responder a essas questões e garantir a precisão das estimativas de valor dos diamantes.
Sent by you: traduza Ingles O propósito deste projeto é criar um site que defina o preço de um diamante com base em suas características: carat (quilate), cut (corte), color (cor), clarity (claridade), price (preço), depth (profundidade), table (tabela), x (comprimento), y (largura) e z (profundidade). Entretanto, em casos extremos onde é necessário fazer uma estimativa rápida do valor de um diamante, não é viável perder tempo definindo todas essas características. Por isso, é necessário realizar um estudo da base de dados para determinar quais são as características mínimas necessárias para estimar o preço de um diamante de forma precisa. Para implementar o projeto, é essencial avaliar como cada característica do diamante influencia seu preço. Isso requer descobrir como a variabilidade de uma característica pode afetar a vari- abilidade do preço. Portanto, o uso de estratégias estatísticas será crucial para responder a essas questões e garantir a precisão das estimativas de valor dos diamantes.

#
  
## Conjunto de Dados 📊

O conjunto de dados utilizado neste projeto é "Diamonds_values_faltantes.csv" e inclui as seguintes colunas:

| Nome da Coluna | Descrição |
|---|---|
| quilate | Peso do diamante em quilates |
| corte | Qualidade do corte do diamante (Ideal, Premium, Muito Bom, Bom, Regular) |
| cor | Cor do diamante (D, E, F, G, H, I, J) |
| clareza | Clareza do diamante (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1) |
| profundidade | Porcentagem da profundidade do diamante |
| mesa | Porcentagem da largura da mesa do diamante |
| preço | Preço do diamante em dólares americanos |
| x | Comprimento do diamante em milímetros |
| y | Largura do diamante em milímetros |
| z | Profundidade do diamante em milímetros |

#

## Metodologia 🛠️

### Carregamento e Exploração de Dados

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

import random
# Mudar o caminho da base de dados
path = r"DataBases\Diamonds_values_faltantes.csv"
diamonds = pd.read_csv(fr"{path}")

diamonds
```

###  Visualização do coeficiente de correlação linear e separação do conjunto de dados, para melhor implementação do KNN.

 Abaixo está a quantidade de valores faltantes por coluna

 <br>

```python
counter = {}
for x in range(diamonds.shape[1]):
    column_name = diamonds.columns[x]
    counter[column_name] = diamonds.shape[0] - len(diamonds[column_name].dropna())

counter_df = pd.DataFrame(list(counter.items()), columns=['Coluna', 'Quantidade de NaN'])
counter_df

plt.figure(figsize = (8, 6))
sns.heatmap((diamonds[["carat", "depth", "table", "price", "x", "y", "z"]]).corr(), vmin = -1, vmax = 1, annot = True, cmap = 'magma')
plt.title("Coeficiente de Correlação Linear")
plt.show()
```

## [Três Métodos para Estimar Preços de Diamantes:]()

1. **Solicite a Massa do Diamante ao Cliente:**

   $$\text{Quilate} = \frac{200}{\text{Massa} \ (\text{mg})}$$

<br>

2. **Quando o Usuário Fornece os Pontos do Diamante:**

   $$\text{Quilate} = \frac{100}{\text{Pontos} \ (\text{pt})}$$

3. **Usando Quatro Elementos para Estimar o Quilate do Diamante:**

   Para o segundo método de estimar o quilate do diamante, quatro elementos são necessários: Comprimento (mm), Largura (mm), Profundidade (mm) e Densidade \(\left( \text{mm}^3/\text{mm} \right)\). Usamos o cálculo da densidade do objeto para calcular primeiro a massa do diamante:

   $$\text{Densidade} = \frac{\text{Volume}}{\text{Massa}} \rightarrow \text{Massa} = \text{Densidade} \times \text{Volume}$$

   No entanto, não temos o volume do diamante. Para obtê-lo, vamos detalhar o cálculo do volume de um objeto da seguinte forma:

   $$\text{Volume} = \text{Comprimento} \times \text{Largura} \times \text{Profundidade}$$

   ### [Substituindo isso na fórmula original temos:]()

   $$\text{Massa} = \text{Densidade} \times (\text{Comprimento} \times \text{Largura} \times \text{Profundidade})$$

   Agora, precisamos encontrar o quilate do diamante. Para isso, utilizaremos a Fórmula 1 para estimar o quilate do diamante:

   $$\text{Quilate} = \frac{200}{\text{Massa}(\text{mg})}$$

   A fórmula geral se torna:

   $$\text{Quilate} = \frac{200}{\text{Densidade} \times \text{Volume}}$$

   OU

   $$\text{Quilate} = \frac{200}{\text{Comprimento} \times \text{Largura} \times \text{Profundidade} \times \text{Densidade}}$$

## [Engenharia de Recursos]()

Análise do mapa de calor acima com base no preço:
Podemos concluir que o preço não tem uma boa correlação com a porcentagem total do diamante (profundidade) e também não tem uma correlação alta com a mesa, sendo uma correlação inversamente proporcional de -0,0086 com a profundidade, e uma relação proporcional de 0,13 com a mesa.
 Podemos concluir também que o preço tem uma boa correlação linear com o quilate (quilate) de 0,92, x (comprimento) de 0,89, y (largura) de 0,89 e z (profundidade) de 0,88.

 Com base nessa análise do mapa de calor, podemos concluir que quanto maior o quilate (quilate), x (comprimento), y (largura) e z (profundidade), maior poderá ser o preço do diamante.

Entretanto, podem existir alguns casos, de se ter um diamante com um quilate muito alto porém com um preço baixo, assim como poderá existir diamantes com um quilate baixo mas com um preço alto. Tal, poderá também acontecer com o x (comprimento), y (largura) e z (profundidade), por causa disso nos questionamos o seguinte, quanto que o quilate (quilate), x (comprimento), y (largura) e z (profundidade) conseguem determinar o valor do diamante? Para responder isso, precisamos tirar o Coeficiente de Determinação.

```python
plt.figure(figsize = (8, 6))
sns.heatmap((diamonds[["carat", "depth", "table", "price", "x", "y", "z"]]).corr()**2, vmin = -1, vmax = 1, annot = True, cmap = 'magma')
plt.title("Coeficiente de Determinação")
plt.show()
```

### Análise do mapa de calor acima com base no preço:

Ao analisarmos o mapa de calor acima, podemos perceber que podemos definir o preço do diamante com maior confiabilidade usando a variável numérica quilate (quilate), com confiabilidade de 85%. Isso significa que, embora possamos dizer que quanto maior o quilate do diamante, maior o seu preço, infelizmente essa regra só é de fato válida para 85% dos dados.

Já para x (comprimento), y (largura) e z (profundidade), essa confiabilidade é de apenas 79% para comprimento e largura, e 78% para profundidade, o que não é uma determinação forte, e por isso poderão ser desconsideradas caso as variáveis categóricas, consigam definir com precisão o preço do diamante.

Abaixo estamos realizando o processo de separação da base de dados diamonds. Para que assim, o processo de machine learn seja mais efetivo.

 Cut tem 5 tipos de classificação Ideal, Premium, Good, Very Good e Fair

- Color tem 7 tipos de classificação E, I, J, H, F, G e D

- Clarity tem 8 tipos de classificação SI2, SI1, VS1, VS2, VVS2, VVS1, I1 and IF

- Implementação do K-NN

Colocando medições iguais a 0 de comprimento, largura e/ou profundidade de um diamante como NaN

```python
for x in range(diamonds.shape[0]):
    for y in range(7, diamonds.shape[1]):
        if diamonds.iloc[x, y] == 0: diamonds.iloc[x, y] = np.nan
        elif diamonds.iloc[x, y] >= 30: diamonds.iloc[x, y] = np.nan
diamonds
```

### Below is the implementation of K-NN in the numerical columns

Some books advise using the formula (K = log n) where n is the number of rows in the database.
To thus define the amount of K.'''

```python
classificacao = KNNImputer(n_neighbors = round(math.log(diamonds.shape[0])))
diamonds[["carat", "depth", "table", "price", "x", "y", "z"]] = classificacao.fit_transform(diamonds[["carat", "depth", "table", "price", "x", "y", "z"]])

diamonds
```

### Aplicando K-NN for categorical columns

```python
'''KNN for categorical values'''
encoder = OrdinalEncoder()
diamonds_encoder = encoder.fit_transform(diamonds)

knn_imputer = KNNImputer(n_neighbors = round(math.log(diamonds.shape[0])))
diamonds_imputer = knn_imputer.fit_transform(diamonds_encoder)

diamonds_imputer = pd.DataFrame(diamonds_imputer, columns = diamonds.columns)
diamonds_imputer = encoder.inverse_transform(diamonds_imputer)
```

### Angular Coefficient

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/22ab9ccb-3e3d-4884-9700-3cdd2811461a"


###  Replacing missing values in the main diamonds database

```python
for x in range(diamonds.shape[0]):
    for y in range(1, 4):
        if pd.isna(diamonds.iloc[x, y]): diamonds.iloc[x, y] = diamonds_imputer[x][y]

diamonds
```

### Below we are normalizing the numerical columns.

```python
standardization of numerical columns
diamonds[["carat", "x", "y", "z"]] = round(diamonds[["carat", "x", "y", "z"]], 2)
diamonds[["table", "price"]] = round(diamonds[["table", "price"]])
diamonds["depth"] = round(diamonds["depth"], 1)

diamonds
```


### Aplicando K-NN for categorical columns

```python
'''KNN for categorical values'''
encoder = OrdinalEncoder()
diamonds_encoder = encoder.fit_transform(diamonds)

knn_imputer = KNNImputer(n_neighbors = round(math.log(diamonds.shape[0])))
diamonds_imputer = knn_imputer.fit_transform(diamonds_encoder)

diamonds_imputer = pd.DataFrame(diamonds_imputer, columns = diamonds.columns)
diamonds_imputer = encoder.inverse_transform(diamonds_imputer)
```

###  Replacing missing values in the main diamonds database

```python
for x in range(diamonds.shape[0]):
    for y in range(1, 4):
        if pd.isna(diamonds.iloc[x, y]): diamonds.iloc[x, y] = diamonds_imputer[x][y]

diamonds
```

### Below we are normalizing the numerical columns.

```python
standardization of numerical columns
diamonds[["carat", "x", "y", "z"]] = round(diamonds[["carat", "x", "y", "z"]], 2)
diamonds[["table", "price"]] = round(diamonds[["table", "price"]])
diamonds["depth"] = round(diamonds["depth"], 1)

diamonds
```

### Coefficient of Determination


<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/9e748a04-bd7f-4639-b374-a8a62170f48f"/>

### Modelo de Predição de Preço

#### Salvando o banco de dados já limpo, sem valores ausentes

```python
path = r"DataBases\Diamonds_clean.csv"
try:
    pd.read_csv(f"{path}")
    print(f"This dataframe already exists in the directory: {path}")
except FileNotFoundError:
    diamonds.to_csv(fr"{path}", index=False)
    print(f'''Cleaned database added to directory:
          {path}
          successfully!!''')
```
          
### Análise da relação do preço com as colunas numéricas

#### INFORMAÇÕES IMPORTANTES:

1- Um quilate é equivalente a 200mg

2- Dois pontos são equivalentes a 0,01 quilates

O gráfico abaixo compara a relação do comprimento de um diamante com o quilate e com o preço

```python
plt.figure(figsize=(17, 10))

plt.subplot(2, 1, 1)
sns.scatterplot(data=diamonds, x="x", y="price")
plt.xlabel("Comprimento (mm)")
plt.ylabel("Preço")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis="y", alpha=0.5)

plt.subplot(2, 1, 2)
sns.scatterplot(data=diamonds, x="x", y="carat")
plt.xlabel("Comprimento (mm)")
plt.ylabel("Quilate")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis="y", alpha=0.5)

plt.show()
```

### Relação do comprimento de um diamante com o quilate e o preço

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/3ec26597-e22b-4910-b422-f11b4720effe"/>













#

## Códigos e Relatório de Análise de Dados

- [DiamondValuationPredictio.py](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/50d1ec6c1074be9848ad472ff1955aad541d1ae2/Codes/Codes%20English/Codes%20Portugues/Codes%20%20English/diamondvaluationenglish.py)

- [DiamondValuationPrediction.ipynb](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/1770a467185ea56dffdede68b81f730bcff8ead4/Codes/Codes%20English/Codes%20Portugues/Codes%20%20English/diamondValuationEnglish.ipynb)

- [Relatório de Análise de Dados](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/d70469f8420bba79f97ed01cbac5234659503a79/Data%20Analyze%20Report/Data%20Analyse%20Portugues/avaliacaoDiamantePortugues%20.pdf)

#

## Acesse o Site Streamlit

🚀 [Toque aqui e teletransporte-se para o Site Streamlit](https://diamondsvalues.streamlit.app/) 

#

## Códigos QR

<p align="center"> 👑 Código QR do Site no Streamlit </p>

<p align="center">
  <img src="Site.png" alt="QR Code 1" width="200"/>
  </p>

  <br>

<!--  
<p align="center"> 👑🇺🇸 QR Code of the Site on Streamlit (English)  (Portuguese) </p>

<p align="center">
  <img src="" />
  </p>

  <br>
  -->

<p align="center">:octocat: Código QR do Repositório GitHub </p>

  <p align="center">
  <img src="RepositorioGitHub.png" alt="QR Code 2" width="200"/>
</p>


#

###### <p align="center">[Copyright 2024 Mindful-AI-Assistants. Código lançado sob a licença MIT.]( https://github.com/Mindful-AI-Assistants/.github/blob/ad6948fdec771e022d49cd96f99024fcc7f1106a/LICENSE)
