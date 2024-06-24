 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]
  <!--  START HEADER  -->  

<br><br>


# <p align="center">  💎 Predição de Valoração de Diamantes

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

### Engenharia de Recursos





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
