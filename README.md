
 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]
  <!--  START HEADER  -->  

<br><br>


# <p align="center">  💎 Diamond Valuation Prediction

<br>

This repository contains a Python project for  analyzing a dataset of diamond characteristics and predicting their price using machine learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Findings](#findings)
- [Analysis of Numerical Features](#analysis-of-numerical-features)
- [Analysis of Categorical Features](#analysis-of-categorical-features)
- [Insights](#insights)
- [Recommendations](#recommendations)
- [Conclusion](#conclusion)
- [File Structure](#file-structure)
- [Getting Started Cloning](#getting-started)
- [Contributing](#contributing)
- [Git Commands](#GitCommands)
- [Codes and Data Analysis Report](#Report)
- [Access the Streamlit Site](https://diamondsvalues.streamlit.app/)  <!-- - [ Tap here and teleport to the site]() 🇺🇸  --> 
- [QR Coides](#QRCodes)
- [Our Team](#our-team)
- [Code of Conduct](#code-of-conduct)
- [License](#license)

#
  
##  📚 Introduction 

This project explores the fascinating world of diamonds and seeks to predict their price based on a variety of factors. We aim to uncover the hidden relationships between diamond characteristics and their value, ultimately contributing to a deeper understanding of the diamond market.

The purpose of this predictive analysis is to create a website that determines the price of a diamond based on its characteristics: carat, cut, color, clarity, price, depth, table, x (length), y (width), and z (depth). 
However, in extreme cases where a quick estimate of a diamond’s value is needed, it is not feasible to spend time defining all these characteristics. Therefore, it is necessary to conduct a study of the database to determine what are the minimum necessary characteristics to accurately estimate the price of a diamond. To implement the project, it is essential to assess how each characteristic of the diamond influences its price. This requires discovering how the variability of a characteristic can affect the variability of the price. Therefore, the use of statistical strategies will be crucial to answer these questions and ensure the accuracy of the diamond value estimate
  
#

## Dataset 📊

The dataset used in this project is "Diamonds_values_faltantes.csv" and includes the following columns:

| Column Name | Description |
|---|---|
| carat | Weight of the diamond in carats |
| cut | Quality of the diamond's cut (Ideal, Premium, Very Good, Good, Fair) |
| color | Color of the diamond (D, E, F, G, H, I, J) |
| clarity | Clarity of the diamond (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1) |
| depth | Percentage of the diamond's depth |
| table | Percentage of the diamond's table width |
| price | Price of the diamond in US dollars |
| x | Length of the diamond in millimeters |
| y | Width of the diamond in millimeters |
| z | Depth of the diamond in millimeters |

#

## Metodology 🛠️

### Loading and Data Exploration

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

###  Visualization of the linear correlation coefficient and separation of the dataset, for better KNN implementation. 

Below is the number of missing values per column

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

### Resource Engineering

### Analysis of the heat map above based on price:

We can conclude that the price does not have a good correlation with the total percentage of the diamond (depth) and also does not have a high correlation with the table, with an inversely proportional correlation of -0.0086 with depth, and a proportional relationship of 0.13 with the table.
We can also conclude that the price has a good linear correlation with the carat of 0.92, x (length) of 0.89, y (width) of 0.89, and z (depth) of 0.88.

Based on this heat map analysis, we can conclude that the larger the carat, x (length), y (width), and z (depth), the higher the diamond's price can be.

However, there may be some cases where a diamond has a very high carat but a low price, just as there may be diamonds with a low carat but a high price. This can also happen with x (length), y (width), and z (depth). Because of this, we question the following: how well can the carat, x (length), y (width), and z (depth) determine the value of the diamond? To answer this, we need to derive the Coefficient of Determination.

```python
Copy code
plt.figure(figsize=(8, 6))
sns.heatmap((diamonds[["carat", "depth", "table", "price", "x", "y", "z"]]).corr()**2, vmin=-1, vmax=1, annot=True, cmap='magma')
plt.title("Coefficient of Determination")
plt.show()
```

### Analysis of the heat map above based on price:

When analyzing the heat map above, we can see that we can define the price of the diamond more reliably using the numerical variable carat, with 85% reliability. This means that although we can say that the higher the carat of the diamond, the higher its price, unfortunately, this rule is only valid for 85% of the data.

For x (length), y (width), and z (depth), this reliability is only 79% for length and width and 78% for depth, which is not a strong determination. Therefore, they may be disregarded if the categorical variables can accurately define the price of the diamond.

Below we are performing the process of separating the diamonds database so that the machine learning process is more effective.

- Cut has 5 classification types: Ideal, Premium, Good, Very Good, and Fair

- Color has 7 classification types: E, I, J, H, F, G, and D

- Clarity has 8 classification types: SI2, SI1, VS1, VS2, VVS2, VVS1, I1, and IF

### Implementation of K-NN

Setting length, width, and/or depth measurements of a diamond equal to 0 as NaN

```python
Copy code
for x in range(diamonds.shape[0]):
    for y in range(7, diamonds.shape[1]):
        if diamonds.iloc[x, y] == 0: 
            diamonds.iloc[x, y] = np.nan
        elif diamonds.iloc[x, y] >= 30: 
            diamonds.iloc[x, y] = np.nan
diamonds
```

Below is the implementation of K-NN in the numerical columns
Some books advise using the formula (K = log n) where n is the number of rows in the database.
To thus define the amount of K.

```python
Copy code
classification = KNNImputer(n_neighbors=round(math.log(diamonds.shape[0])))
diamonds[["carat", "depth", "table", "price", "x", "y", "z"]] = classification.fit_transform(diamonds[["carat", "depth", "table", "price", "x", "y", "z"]])

diamonds
```

### Applying K-NN for categorical columns

#### Applying K-NN for categorical columns

```python
Copy code
'''KNN for categorical values'''
encoder = OrdinalEncoder()
diamonds_encoder = encoder.fit_transform(diamonds)

knn_imputer = KNNImputer(n_neighbors = round(math.log(diamonds.shape[0])))
diamonds_imputer = knn_imputer.fit_transform(diamonds_encoder)
diamonds_imputer = pd.DataFrame(diamonds_imputer, columns = diamonds.columns)
diamonds_imputer = encoder.inverse_transform(diamonds_imputer)
```

IMAGW1

#### Replacing missing values in the main diamonds database

```python
Copy code
for x in range(diamonds.shape[0]):
    for y in range(1, 4):
        if pd.isna(diamonds.iloc[x, y]): 
            diamonds.iloc[x, y] = diamonds_imputer[x][y]

diamonds
```

Below we are normalizing the numerical columns.

```python
standardization of numerical columns
diamonds[["carat", "x", "y", "z"]] = round(diamonds[["carat", "x", "y", "z"]], 2)
diamonds[["table", "price"]] = round(diamonds[["table", "price"]])
diamonds["depth"] = round(diamonds["depth"], 1)

diamonds
```

IMa 2

### Price Prediction Model

#### Saving the already cleaned database without missing values

```Python
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

#### Analysis of the price relationship of the numerical columns

### **IMPORTANT INFORMATION:**

1- **Carat is equivalent to 200mg**

2- **Points are equivalent to 0.01 carats**

The graph below compares the relationship of the length of a diamond with the carat and with the price

```python
plt.figure(figsize=(17, 10))

plt.subplot(2, 1, 1)
sns.scatterplot(data=diamonds, x="x", y="price")
plt.xlabel("Length (mm)")
plt.ylabel("Price")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis="y", alpha=0.5)

plt.subplot(2, 1, 2)
sns.scatterplot(data=diamonds, x="x", y="carat")
plt.xlabel("Length (mm)")
plt.ylabel("Carat")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis="y", alpha=0.5)

plt.show()
```















#





#

## Codes and Data Analysis Report

- [DiamondValuationPredictio.py](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/50d1ec6c1074be9848ad472ff1955aad541d1ae2/Codes/Codes%20English/Codes%20Portugues/Codes%20%20English/diamondvaluationenglish.py)

- [DiamondValuationPrediction.ipynb](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/1770a467185ea56dffdede68b81f730bcff8ead4/Codes/Codes%20English/Codes%20Portugues/Codes%20%20English/diamondValuationEnglish.ipynb)

- [Data Analysis Report](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/d70469f8420bba79f97ed01cbac5234659503a79/Data%20Analyze%20Report/Data%20Analyse%20Portugues/avaliacaoDiamantePortugues%20.pdf)

#

## Access the Streamlist Site

🚀 [Tap here and teleport to the Streamlit Site](https://diamondsvalues.streamlit.app/) 

#

## QR Codes

<p align="center"> 👑 QR Code of the Site on Streamlit </p>

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

<p align="center">:octocat: QR Code of the GitHub Repository </p>

  <p align="center">
  <img src="RepositorioGitHub.png" alt="QR Code 2" width="200"/>
</p>


#

###### <p align="center">[Copyright 2024 Mindful-AI-Assistants. Code released under the  MIT license.]( https://github.com/Mindful-AI-Assistants/.github/blob/ad6948fdec771e022d49cd96f99024fcc7f1106a/LICENSE)
