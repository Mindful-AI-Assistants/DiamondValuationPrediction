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

Para o estudo da base de dados, utilizaremos diversas estratégias estatísticas, incluindo a regressão linear, e aplicaremos conhecimentos em química para a formulação de equações matemáticas, com o objetivo de definir o preço dos diamantes com base em suas características. Adicionalmente, para limpar a base de dados, que contém valores ausentes, e para prever os valores dos diamantes com base em suas características, empregaremos o algoritmo de agrupamento KNN (K-Nearest Neighbors). Este é um algoritmo de aprendizado supervisionado que será utilizado tanto para a limpeza quanto para as previsões. Para estimar os valores ausentes na base de dados, o algoritmo KNN calculará, individualmente, a distância entre os diamantes com valores ausentes e aqueles com valores presentes, com base nas características conhecidas dos diamantes. Em seguida, o KNN identificará os diamantes mais próximos do que está sendo analisado e utilizará essas informações para prever o valor ausente. O mesmo processo será aplicado para a previsão do preço dos diamantes.

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

### [Análise do mapa de calor acima com base no preço:]()

Ao analisar o mapa de calor acima, podemos ver que podemos definir o preço do diamante de forma mais confiável usando a variável numérica quilates, com 85% de confiabilidade. Isso significa que, embora possamos dizer que quanto maior o quilate do diamante, maior o seu preço, infelizmente, essa regra só é válida para 85% dos dados.

Para x (comprimento), y (largura) e z (profundidade), essa confiabilidade é de apenas 79% para comprimento e largura e 78% para profundidade, o que não é uma determinação forte. Portanto, eles podem ser desconsiderados se as variáveis categóricas puderem definir com precisão o preço do diamante.

Abaixo estamos realizando o processo de separação do banco de dados dos diamantes para que o processo de aprendizado de máquina seja mais eficaz.

 *- O corte tem 5 tipos de classificação: Ideal, Premium, Bom, Muito Bom e Justo*

 *- A cor tem 7 tipos de classificação: E, I, J, H, F, G e D*

 *- A clareza tem 8 tipos de classificação: SI2, SI1, VS1, VS2, VVS2, VVS1, I1 e IF*


###  [Implementação do Algoritmo K-NN]()

Definindo as medidas de comprimento, largura e/ou profundidade de um diamante iguais a 0 como NaN

```python
for x in range(diamonds.shape[0]):
    for y in range(7, diamonds.shape[1]):
        if diamonds.iloc[x, y] == 0: 
            diamonds.iloc[x, y] = np.nan
        elif diamonds.iloc[x, y] >= 30: 
            diamonds.iloc[x, y] = np.nan
diamonds
```

### [👇 Abaixo está a implementação do Algoritmo K-NN nas colunas numéricas]()

ps: Alguns livros recomendam usar a fórmula (K = log n), onde n é o número de linhas no banco de dados. Para assim definir a quantidade de K.

```python
classification = KNNImputer(n_neighbors=round(math.log(diamonds.shape[0])))
diamonds[["carat", "depth", "table", "price", "x", "y", "z"]] = classification.fit_transform(diamonds[["carat", "depth", "table", "price", "x", "y", "z"]])

diamonds
```

### [Aplicação do Algoritmo K-NN para Colunas Categóricas]()

```python
'''KNN para valores categóricos'''
encoder = OrdinalEncoder()
diamonds_encoder = encoder.fit_transform(diamonds)

knn_imputer = KNNImputer(n_neighbors = round(math.log(diamonds.shape[0])))
diamonds_imputer = knn_imputer.fit_transform(diamonds_encoder)
diamonds_imputer = pd.DataFrame(diamonds_imputer, columns = diamonds.columns)
diamonds_imputer = encoder.inverse_transform(diamonds_imputer)
```

### [Gráfico do Coeficiente Angular]()

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/22ab9ccb-3e3d-4884-9700-3cdd2811461a"

### [Substituindo valores ausentes no banco de dados principal dos diamantes]()

```python
for x in range(diamonds.shape[0]):
    for y in range(1, 4):
        if pd.isna(diamonds.iloc[x, y]): 
            diamonds.iloc[x, y] = diamonds_imputer[x][y]

diamonds
```

### 👇[Abaixo estamos normalizando as colunas numéricas]()

```python
padronização das colunas numéricas
diamonds[["carat", "x", "y", "z"]] = round(diamonds[["carat", "x", "y", "z"]], 2)
diamonds[["table", "price"]] = round(diamonds[["table", "price"]])
diamonds["depth"] = round(diamonds["depth"], 1)

diamonds
```

### [Gráfico do Coeficiente de Determinação]()

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/9e748a04-bd7f-4639-b374-a8a62170f48f"/>


## [Modelo de Predição de Preço]()

### [Salvando o banco de dados já limpo, sem valores ausentes]()

```Python
path = r"DataBases\Diamonds_clean.csv"
try:
    pd.read_csv(f"{path}")
    print(f"Este dataframe já existe no diretório: {path}")
except FileNotFoundError:
    diamonds.to_csv(fr"{path}", index=False)
    print(f'''Banco de dados limpo adicionado ao diretório:
          {path}
          com sucesso!!''')
```

### [Análise da Relação do Preço das Colunas Numéricas]()

#### **⭕️ INFORMAÇÕES IMPORTANTES:**

1- **O quilate equivale a 200mg**

2- **Pontos são equivalentes a 0.01 quilates**

### [👇 O gráfico abaixo compara a relação do comprimento de um diamante com o quilate e com o preço]()

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
### Relação do Comprimento de um Diamante com o Quilate e o Preço

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/3ec26597-e22b-4910-b422-f11b4720effe"/>


### [👇 O gráfico abaixo compara a relação da largura de um diamante com o quilate e com o preço]()

```python
plt.figure(figsize=(17, 10))

plt.subplot(2, 1, 1)
sns.scatterplot(diamonds, x = "y", y = "price")
plt.xlabel("Largura (mm)")
plt.ylabel("Preço")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.subplot(2, 1, 2)
sns.scatterplot(diamonds, x = "y", y = "carat")

plt.xlabel("Largura (mm)")
plt.ylabel("Quilate")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.show()
```

### Relação da Largura de um Diamante com o Quilate e o Preço

![4  Relação da largura de um diamante com o quilate e o preço](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/a2b83a69-1570-4c76-85ba-3b98726160d4)


### [👇 O gráfico abaixo compara a relação da profundidade de um diamante com o quilate e com o preço]()

```python
plt.figure(figsize=(17, 10))

plt.subplot(2, 1, 1)
sns.scatterplot(diamonds, x = "z", y = "price")
plt.xlabel("Profundidade (mm)")
plt.ylabel("Preço")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.subplot(2, 1, 2)
sns.scatterplot(diamonds, x = "z", y = "carat")
plt.xlabel("Profundidade (mm)")
plt.ylabel("Quilate")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.show()
```

### Relação da Profundidade de um Diamante

 com o Quilate e o Preço

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/8f17f4b1-2fbf-4eb2-80e4-8755c5422c51"/>

### [👇 O gráfico abaixo compara a relação do quilate de um diamante com o preço]()

```python
plt.figure(figsize=(17, 5))
sns.scatterplot(diamonds, x = "carat", y = "price")
plt.xlabel("Quilate")
plt.ylabel("Preço")
plt.title("Relação entre Preço e Quilate")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.show()
```

### Relação do Quilate de um Diamante com o Preço

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/ba6b45cd-e624-4dfa-85a9-8b8a1cde832b"/>

## [🔍 Descobertas]()

### [Análise de Recursos Numéricos]()

Quilate: O preditor mais forte do preço do diamante, com um coeficiente de determinação de 85%.
Comprimento, Largura, Profundidade: Embora correlacionados com o preço, essas características têm uma relação mais fraca em comparação com o quilate. Essas características são mais úteis para prever o peso do diamante (quilate) do que o preço.

### [Análise de Recursos Categóricos]()

Corte, Cor, Clareza: Essas características não estão diretamente correlacionadas com o preço. No entanto, a análise de sua distribuição em diferentes faixas de preço revela insights sobre como esses fatores influenciam a faixa de preço. Por exemplo, uma maior porcentagem de diamantes com um corte "Ideal" pode ser encontrada em faixas de preço mais altas.

## [💡 Insights]()

O preditor mais confiável do preço de um diamante é seu peso em quilates.

Embora comprimento, largura e profundidade estejam correlacionados com o preço, sua relação é mais fraca do que a do quilate, sugerindo que essas dimensões são mais úteis para determinar o peso.

Recursos categóricos, como corte, cor e clareza, não estão diretamente correlacionados com o preço, mas podem fornecer uma indicação geral da faixa de preço.
Uma combinação de recursos numéricos e categóricos pode ser usada para construir um modelo de previsão de preço mais preciso.


## [📈 Recomendações]() 

Explore modelos de aprendizado de máquina mais complexos (por exemplo, florestas aleatórias, máquinas de vetores de suporte) para potencialmente melhorar a precisão da previsão.

Analise a distribuição de recursos categóricos em diferentes faixas de preço para entender melhor sua influência.

Considere incorporar outros recursos relevantes, como certificação de diamantes, origem e gravidade específica, para aumentar o poder preditivo do modelo.

## [🎉 Conclusão]()

Este projeto de análise de dados identificou com sucesso as principais características que impactam o preço de um diamante e demonstrou a importância da engenharia de recursos na construção de modelos de previsão precisos. Ao entender as relações entre as características dos diamantes e o preço, esta análise pode informar estratégias de precificação para varejistas de diamantes e fornecer insights valiosos para os consumidores.

Nota: Este relatório é baseado no trecho de código fornecido. Mais detalhes sobre o modelo de previsão e seu desempenho não estão disponíveis e exigiriam informações adicionais.

Estrutura do Arquivo 📁

└── 🇺🇸 diamondValuationEnglish.ipynb

    └── 🇺🇸 diamondValuationEnglish.py
    
    └── 🇧🇷 avaliacaoDiamante.inpyb
    
    └── 🇧🇷 avaliacaoDi

## 👌 [Clone este Repositório]()

```bash
git clone https://github.com/[seu-usuário]/diamond-price-prediction.git
```

Instale os Pacotes Necessários:
```bash
pip install -r requirements.txt
```

## [🤝 Contribuição]()

Toda contribuição é altamente apreciada. Você pode contribuir de duas maneiras:

1. Crie uma issue e nos conte sua ideia 💡. Certifique-se de usar a etiqueta *nova ideia* nesse caso;
2. Faça um fork do projeto e envie uma *pull request* com sua nova ideia. Antes disso, por favor, certifique-se de ler e seguir o [Guia de Contribuição](https://github.com/Mindful-AI-Assistants/.github/blob/9e7e98f98af07a1d6c4bdeb349e1a9db04f8ed0e/CONTRIBUTING.md).

## [💻 Comandos Git]()

Crie um novo branch:

```bash
git checkout -b feature/minha-feature
```

### Adicione as mudanças à área de stage:

```bash
git add .
```

Faça o commit das mudanças:

```bash
git commit -m "feat: Implementada nova funcionalidade"
```

Envie as mudanças para o repositório remoto:

```bash
git push origin feature/minha-feature
```

Crie uma pull request:

[Link do Repositório](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/edit/main/README)

Mescle as mudanças:

```bash
git merge feature/minha-feature
```

Delete o branch:

```bash
git branch -d feature/minha-feature
```

## [👩🏽‍💻 Códigos e Relatório de Análise de Dados]()

### Códigos:

- [DiamondValuationPredictio.py](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/50d1ec6c1074be9848ad472ff1955aad541d1ae2/Codes/Codes%20English/Codes%20Portugues/Codes%20%20English/diamondvaluationenglish.py)

- [DiamondValuationPrediction.ipynb](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/1770a467185ea56dffdede68b81f730bcff8ead4/Codes/Codes%20English/Codes%20Portugues/Codes%20%20English/diamondValuationEnglish.ipynb)

<br>

### Conjunto de Dados:

- [Dataset Diamond Valuation - do kaggle](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/57c0e19bc9bd3efbdd8256442129261df9ac358f/Database/Diamonds_limpa%20(1).csv)

<br>

### Relatório de Análise de Dados:  

- [Relatório de Análise de Dados](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/86c2111a01c153279e7e8a7744f398041ef5d35b/Data%20Analyze%20Report/Data%20Analyse%20English/Data%20Analyse%20English.pdf)

## [👑 Acesse o Site no Streamlit]()

<br>

🚀 [Clique aqui e teleporte-se para o Site do Streamlit](https://diamondsvalues.streamlit.app/)

## [QR Codes]()

### <p align="center"> 👑 QR Code do Site no Streamlit </p>

<p align="center">
    <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/e52ee3b6-6260-4733-8a65-5c7e2bc8c5d7" alt="QR Code 1" width="200"/>
</p>

<br>

### <p align="center"> :octocat: QR Code do Repositório no GitHub </p>

<p align="center"> 
    <img src="https://github.com/user-attachments/assets/aa52cd93-ac2d-4728-8c53-a1ed18470905" alt="QR Code 2" width="200"/> 
</p>

<br>

<!--  
<p align="center"> 👑🇺🇸 QR Code do Site no Streamlit (Inglês)  (Português) </p>

<p align="center">
  <img src="" />
  </p>
-->

## [👥 Nosso Time]() 

- [Fabiana 🚀 Campanari](https://github.com/FabianaCampanari)

- [Gabriel Melo Dos Santos](https://github.com/Gabri3l-M)

- [José Augusto de Souza Oliveira](https://github.com/Jojose3)

- [Luan Augusto dos Santos Fabiano](https://github.com/LuanFabiano28)

- [Pedro Gallego Barenco](https://github.com/Pgbarenco)

- [Pedro Vyctor Carvalho De Almeida](https://github.com/ppvyctor)

## [🤝 Código de Conduta]()

Estamos comprometidos em promover uma comunidade acolhedora e inclusiva para todos os membros da equipe. Esperamos que todos sigam os seguintes princípios:

- Seja respeitoso: Trate os outros com cortesia e respeito, independentemente de sua origem, identidade ou opiniões.
- Seja construtivo: Foque em fornecer feedback útil e críticas construtivas.
- Seja mente aberta: Esteja aberto a diferentes perspectivas e ideias.
- Seja responsável: Assuma a responsabilidade por suas ações e palavras.
- Seja inclusivo: Promova um ambiente acolhedor e inclusivo para todos.

Se você testemunhar qualquer violação deste código de conduta, entre em contato com [suas informações de contato] para que possamos lidar com a situação de forma adequada.

## [💌 Contato]()

Para mais informações, entre em contato com [Mindful-AI-Assistants](mailto:fabicampanari@proton.me)

#

###### <p align="center">[Copyright 2024 Mindful-AI-Assistants. Código liberado sob a licença MIT.](https://github.com/Mindful-AI-Assistants/.github/blob/ad6948fdec771e022d49cd96f99024fcc7f1106a/LICENSE)






















#

