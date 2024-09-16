 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]
  <!--  START HEADER  -->  

<br>

# <p align="center">  💎 Previsão de Valorização de Diamantes
#### <p align="center"> [![Sponsor Mindful AI Assistants](https://img.shields.io/badge/Sponsor-Mindful%20AI%20%20Assistants-brightgreen?logo=GitHub)](https://github.com/sponsors/Mindful-AI-Assistants)

<br>

https://github.com/user-attachments/assets/7eeb93e6-9d3f-41d0-b40a-3d17e28670a9

<br><br>


## [Sobre este Projeto](#)

Este repositório inclui um projeto em Python desenvolvido para contar histórias, ciência de dados e inteligência artificial na Universidade PUC-SP. O projeto envolve a análise de um conjunto de dados de características de diamantes e a utilização de técnicas de aprendizado de máquina para prever seus preços.

## [Tabela de Conteúdos](#)

- [Introdução](#introducao)
- [Conjunto de Dados](#dataset)
- [Metodologia](#metodologia)
- [Descobertas](#descobertas)
- [Análise de Recursos Numéricos](#analise-recursos-numericos)
- [Análise de Recursos Categóricos](#analise-recursos-categoricos)
- [Insights](#insights)
- [Recomendações](#recomendacoes)
- [Conclusão](#conclusao)
- [Estrutura dos Arquivos](#estrutura-arquivos)
- [Começando a Clonar](#clonar)
- [Contribuindo](#contribuindo)
- [Comandos Git](#comandos-git)
- [Análise de Dados, Códigos e Relatório](#analise-dados-codigos-relatorio)
- [Acesse o site do Streamlit](https://diamondsvalues.streamlit.app/)
- [QR Codes](#qr-codes)
- [Nosso Time](#nosso-time)
- [Código de Conduta](#codigo-de-conduta)
- [Licença](#licenca)

<br>

## [📚 Introdução](#)

Este projeto explora o fascinante mundo dos diamantes e busca prever seu preço com base em uma variedade de fatores. Nosso objetivo é descobrir as relações ocultas entre as características dos diamantes e seu valor, contribuindo para um entendimento mais profundo do mercado de diamantes.

O propósito desta análise preditiva é criar um site que determine o preço de um diamante com base em suas características: quilate, corte, cor, clareza, preço, profundidade, tabela, x (comprimento), y (largura) e z (profundidade). Em casos extremos onde uma estimativa rápida é necessária, não é viável definir todas essas características. Portanto, é necessário um estudo para determinar as características mínimas necessárias para estimar o preço com precisão.

## [Conjunto de Dados 📊](#)

### O conjunto de dados utilizado neste projeto é "Diamonds_values_faltantes.csv" e inclui as seguintes colunas:

| Nome da Coluna | Descrição |
|---|---|
| quilate | Peso do diamante em quilates |
| corte | Qualidade do corte do diamante (Ideal, Premium, Very Good, Good, Fair) |
| cor | Cor do diamante (D, E, F, G, H, I, J) |
| clareza | Clareza do diamante (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1) |
| profundidade | Percentual da profundidade do diamante |
| tabela | Percentual da largura da tabela do diamante |
| preço | Preço do diamante em dólares |
| x | Comprimento do diamante em milímetros |
| y | Largura do diamante em milímetros |
| z | Profundidade do diamante em milímetros |

## [Metodologia 🛠️](#)

### Carregamento e Exploração de Dados

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

path = r"DataBases\Diamonds_values_faltantes.csv"
diamonds = pd.read_csv(fr"{path}")

diamonds
```

## [Visualização do Coeficiente de Correlação Linear e Separação do Conjunto de Dados](#)

Abaixo está o número de valores ausentes por coluna:

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

## [Três Métodos para Estimar os Preços dos Diamantes:](#)

1. **Solicitar a Massa do Diamante ao Cliente:**

 $$\text{Carat} = \frac{200}{\text{Mass} \ (\text{mg})}$$

<br>

2. **Quando o Usuário Fornece os Pontos do Diamante:**

$$\text{Carat} = \frac{100}{\text{Points} \ (\text{pt})}$$
 

3. **Usando Quatro Elementos para Estimar o Quilate do Diamante:**

   Para o segundo método, quatro elementos são necessários: Comprimento, Largura, Profundidade e Densidade. Usamos a densidade do objeto para calcular a massa do diamante:

   $$\text{Density} = \frac{\text{Volume}}{\text{Mass}} \rightarrow \text{Mass} = \text{Density} \times \text{Volume}$$

   Substituindo na fórmula:

   $$\text{Carat} = \frac{200}{\text{Length} \times \text{Width} \times \text{Depth} \times \text{Density}}$$


## [Engenharia de Recursos](#)

### [Análise do Mapa de Calor Baseada no Preço:](#)

Concluímos que o preço tem boa correlação linear com o quilate (0.92), comprimento (0.89), largura (0.89) e profundidade (0.88), indicando que quanto maiores essas dimensões, maior o preço.

```python
plt.figure(figsize=(8, 6))
sns.heatmap((diamonds[["carat", "depth", "table", "price", "x", "y", "z"]]).corr()**2, vmin=-1, vmax=1, annot=True, cmap='magma')
plt.title("Coeficiente de Determinação")
plt.show()
```


### [👇 O gráfico abaixo compara a relação do quilate de um diamante com o preço:]()

```python
plt.figure(figsize=(17, 10))

plt.subplot(2, 1, 1)
sns.scatterplot(diamonds, x = "carat", y = "price")
plt.xlabel("Quilate")
plt.ylabel("Preço")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.subplot(2, 1, 2)
sns.histplot(diamonds, x = "carat", bins = 30, kde = True)
plt.xlabel("Quilate")
plt.ylabel("Distribuição")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.show()
```

### Relação do Quilate de um Diamante com o Preço

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/9ec15c21-f88e-486d-97ef-5f4d6d2ff165"/>

### [👇 O gráfico abaixo compara a relação entre a cor do diamante com o preço:]()

```python
plt.figure(figsize=(17, 10))

sns.boxplot(diamonds, x = "color", y = "price")
plt.xlabel("Cor")
plt.ylabel("Preço")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.show()
```

### Relação da Cor do Diamante com o Preço

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/7f5e6461-7958-4f9b-a3cb-8ff9a61590b5"/>









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
