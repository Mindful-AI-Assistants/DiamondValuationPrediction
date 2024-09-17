 \[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]
  <!--  START HEADER  -->  

<br>

# <p align="center">  💎 Diamond Valuation Prediction
#### <p align="center"> [![Sponsor Mindful AI Assistants](https://img.shields.io/badge/Sponsor-Mindful%20AI%20%20Assistants-brightgreen?logo=GitHub)](https://github.com/sponsors/Mindful-AI-Assistants)

<br>

https://github.com/user-attachments/assets/292156b0-1430-48f7-b6d7-2ce08a7d6fee

<br><br>

## [About This Projeto]()

This repository includes a Python project developed for <!--storytelling, data science, and artificial intelligence at PUC-SP University-->the [AI Project Showcase Competition 2024 organized by Ready Tensor AI](https://app.readytensor.ai/publications/sumbot_freecode_uaWsno2Z7r2a).
. The project involves analyzing a dataset of diamond characteristics and using machine learning techniques to predict their price.

## [Table of Contents]()

- [Introduction](#introduction)
- [Data Set](#dataset)
- [Methodology](#mMethodology)
- [Discoveries](#discoveries)
- [Numerical-resource-analysis](#numerical-resource-analysis)
- [Analysis of Categorical Resources](#analysis-of-categorical-resources)
- [Insights](#insights)
- [Recommendations](#recommendations)
- [Conclusion](#conclusion)
- [File Structure](#file-structure)
- [Starting to Clone](#starting-clone)
- [Contributing](#contributing)
- [Git Commands](#GitCommands)
- [Data Analysis, Codes and Report](#DataAnalysis-CodesReport)
- [Access the Streamlit Site](https://diamondsvalues.streamlit.app/)  <!-- - [ Tap here and teleport to the site]() 🇺🇸  --> 
- [QR Codes](#QRCodes)
- [Our Team](#our-team)
- [Code of Conduct](#code-of-conduct)
- [License](#license)


<br>
  
##  [📚 Introduction]()

This project explores the fascinating world of diamonds and aims to predict their price based on a variety of factors. Our goal is to uncover hidden relationships between diamond characteristics and their value, contributing to a deeper understanding of the diamond market.

The purpose of this predictive analysis is to create a website that determines the price of a diamond based on its characteristics: carat, cut, color, clarity, price, depth, table, x (length), y (width), and z (depth). In extreme cases where a quick estimate is required, it is not feasible to define all of these characteristics. Therefore, a study is necessary to determine the minimum characteristics needed to estimate the price accurately.

For the database study, we will use various statistical strategies, including linear regression, and apply chemistry knowledge to formulate mathematical equations to define diamond prices based on their characteristics. Additionally, to clean the database, which contains missing values, and to predict the value of diamonds based on their characteristics, we will employ the KNN (K-Nearest Neighbors) clustering algorithm. This is a supervised learning algorithm that will be used for both cleaning and predictions. To estimate missing values in the database, the KNN algorithm will individually calculate the distance between diamonds with missing values and those with known values, based on the known characteristics of the diamonds. Then, the KNN will identify the diamonds closest to the one being analyzed and use this information to predict the missing value. The same process will be applied to predict the price of diamonds.
  


## [Dataset 📊]()

### The dataset used in this project is "Diamonds_values_faltantes.csv" and includes the following columns:

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



## [Metodology 🛠️]()

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

## [Visualization of the **Linear Correlation Coefficient** and **Separation of the Dataset**, for better KNN Implementation]()

👇 Below is the number of missing values per column

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

## [Three Methods to Estimate Diamond Prices:]()

1. **Request the Diamond's Mass from the Client:**

 $$\text{Carat} = \frac{200}{\text{Mass} \ (\text{mg})}$$

<br>

2. **When the User Provides the Diamond's Points:**

$$\text{Carat} = \frac{100}{\text{Points} \ (\text{pt})}$$
 

3. **Using Four Elements to Estimate the Carat of the Diamond:**

   For the second method of estimating the diamond's carat, four elements are required: Length (mm), Width (mm), Depth (mm), and Density (\frac{mm}{mm^2}). We use the object's density calculation to first calculate the diamond's mass:

   $$\text{Density} = \frac{\text{Volume}}{\text{Mass}} \rightarrow \text{Mass} = \text{Density} \times \text{Volume}$$

   However, we don't have the diamond's volume. To obtain it, we'll break down the volume calculation of an object as follows:

   $$\text{Volume} = \text{Length} \times \text{Width} \times \text{Depth}$$
      
   ### [Substituting this into the original formula gives:]()

   $$\text{Mass} = \text{Density} \times (\text{Length} \times \text{Width} \times \text{Depth})$$
   
   Now, we need to find the diamond's carat. To do this, we'll use Formula 1 to estimate the diamond's carat:

   $$\text{Carat} = \frac{\text{Mass}(\text{mg})}{200}$$
   
   The general formula becomes:

   $$\text{Carat} = \frac{\text{Density} \times \text{Volume}}{200}$$
   
   OR

   $$\text{Carat} = \frac{\text{Length} \times \text{Width} \times \text{Depth} \times \text{Density}}{200}$$

 
## [Resource Engineering]()

### [Analysis of the Heat Map Above Based on Price:]()

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

### [Analysis of the heat map above based on price:]()

When analyzing the heat map above, we can see that we can define the price of the diamond more reliably using the numerical variable carat, with 85% reliability. This means that although we can say that the higher the carat of the diamond, the higher its price, unfortunately, this rule is only valid for 85% of the data.

For x (length), y (width), and z (depth), this reliability is only 79% for length and width and 78% for depth, which is not a strong determination. Therefore, they may be disregarded if the categorical variables can accurately define the price of the diamond.

Below we are performing the process of separating the diamonds database so that the machine learning process is more effective.

 *- Cut has 5 classification types: Ideal, Premium, Good, Very Good, and Fair*

 *- Color has 7 classification types: E, I, J, H, F, G, and D*

 *- Clarity has 8 classification types: SI2, SI1, VS1, VS2, VVS2, VVS1, I1, and IF*


###  [Implementation of K-NN Algorithm]()

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

### [👇 Below is the implementation of K-NN Algorithm in the numerical columns]()

ps: Some books advise using the formula (K = log n) where n is the number of rows in the database.
To thus define the amount of K.

```python
Copy code
classification = KNNImputer(n_neighbors=round(math.log(diamonds.shape[0])))
diamonds[["carat", "depth", "table", "price", "x", "y", "z"]] = classification.fit_transform(diamonds[["carat", "depth", "table", "price", "x", "y", "z"]])

diamonds
```

### [Applying K-NN Algorithm for Categorical Columns Algorithm]()

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

### [Angular Coefficient Graphic]()

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/22ab9ccb-3e3d-4884-9700-3cdd2811461a"

    <br>
    
### [Replacing missing values in the main diamonds database]()

```python
Copy code
for x in range(diamonds.shape[0]):
    for y in range(1, 4):
        if pd.isna(diamonds.iloc[x, y]): 
            diamonds.iloc[x, y] = diamonds_imputer[x][y]

diamonds
```

### 👇[Below we are normalizing the numerical columns]()

```python
standardization of numerical columns
diamonds[["carat", "x", "y", "z"]] = round(diamonds[["carat", "x", "y", "z"]], 2)
diamonds[["table", "price"]] = round(diamonds[["table", "price"]])
diamonds["depth"] = round(diamonds["depth"], 1)

diamonds
```

### [Coefficient of Determination Graphic]()

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/9e748a04-bd7f-4639-b374-a8a62170f48f"/>


## [Price Prediction Model]()

### [Saving the already cleaned database without missing values]()

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

### [Analysis of the Price Relationship of the Numerical Columns]()

#### **⭕️ IMPORTANT INFORMATION:**

1- **Carat is equivalent to 200mg**

2- **Points are equivalent to 0.01 carats**

### [👇 The graph below compares the relationship of the length of a diamond with the carat and with the price]()

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
### Relationship of a Diamond’s Length with the Carat and Price Graphic

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/3ec26597-e22b-4910-b422-f11b4720effe"/>


### [👇 The graph below compares the relationship of the width of a diamond with the carat and with the price]()

```python
plt.figure(figsize=(17, 10))

plt.subplot(2, 1, 1)
sns.scatterplot(diamonds, x = "y", y = "price")
plt.xlabel("Width (mm)")
plt.ylabel("Price")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.subplot(2, 1, 2)
sns.scatterplot(diamonds, x = "y", y = "carat")

plt.xlabel("Width (mm)")
plt.ylabel("Carat")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.show()
```

###  Relationship of a Diamond’s Width with the Carat and Price

![4  Relationship of a diamond’s width with the carat and price](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/a2b83a69-1570-4c76-85ba-3b98726160d4)


### [👇 The graph below compares the relationship of the depth of a diamond with the carat and with the price]()

```python
plt.figure(figsize=(17, 10))

plt.subplot(2, 1, 1)
sns.scatterplot(diamonds, x = "z", y = "price")
plt.xlabel("Depth (mm)")
plt.ylabel("Price")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.subplot(2, 1, 2)
sns.scatterplot(diamonds, x = "z", y = "carat")
plt.xlabel("Depth (mm)")
plt.ylabel("Carat")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.show()
```

### Relationship of the Depth of a Diamond with the Carat and with the Price

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/8f17f4b1-2fbf-4eb2-80e4-8755c5422c51"/>

### [👇 The graph below compares the relationship of the carat of a diamond with the price]()

```python
plt.figure(figsize=(17, 5))
sns.scatterplot(diamonds, x = "carat", y = "price")
plt.xlabel("Carat")
plt.ylabel("Price")
plt.title("Price and Carat Relationship")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.grid(axis = "y", alpha = 0.5)

plt.show()
```

### Relationship of the carat of a diamond with the price

<p align="center">
  <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/ba6b45cd-e624-4dfa-85a9-8b8a1cde832b"/=


##  [🔍 Discoveries]()

### [Analysis of Numerical Resources]()

Carat: The strongest predictor of diamond price, with a determination coefficient of 85%.
Length, Width, Depth: Although correlated with the price, these features have a weaker relationship compared to the carat. These features are more useful for predicting the weight of the diamond (carat) than the price.

### [Analysis of Categorical Features]()

Cut, Color, Clarity: These features are not directly correlated with the price. However, analyzing their distribution in different price ranges reveals insights about how these factors influence the price range. For example, a higher percentage of diamonds with an "Ideal" cut can be found in higher price ranges.

## [💡 Insights]()

The most reliable predictor of a diamond's price is its weight in carats.

Although length, width, and depth are correlated with the price, their relationship is weaker than the carat, suggesting that these dimensions are more useful for determining weight.

Categorical features such as cut, color, and clarity are not directly correlated with the price, but can provide a general indication of the price range.
A combination of numerical and categorical features can be used to build a more accurate price prediction model.


## [📈 Recommendations]() 

Explore more complex machine learning models (for example, random forests, support vector machines) to potentially improve prediction accuracy.

Analyze the distribution of categorical features in different price ranges to better understand their influence.

Consider incorporating other relevant features, such as diamond certification, origin, and specific gravity, to increase the predictive power of the model.

## [🎉 Conclusion]()

This data analysis project has successfully identified the key features that impact the price of a diamond and demonstrated the importance of feature engineering in building accurate prediction models. By understanding the relationships between the characteristics of diamonds and the price, this analysis can inform pricing strategies for diamond retailers and provide valuable insights for consumers.

Note: This report is based on the provided code snippet. More details about the prediction model and its performance are not available and would require additional information.

File Structure 📁

└── 🇺🇸 diamondValuationEnglish.ipynb

    └── 🇺🇸 diamondValuationEnglish.py
    
    └── 🇧🇷 avaliacaoDiamante.inpyb
    
    └── 🇧🇷 avaliacaoDiamante


## 👌 [Clone this Repository]()

```bash
git clone https://github.com/[your-username]/diamond-price-prediction.git
```

Install Required Packages:
```
pip install -r requirements.txt
```

## [🤝 Contribution]()

Any contributions are highly appreciated.  You can contribute in two ways:

   1. Create an issue and tell us your idea 💡. Make sure that you use the new idea label in this case;

   2. Fork the project and submit a full requesto with your new idea. Before doing that, please make sure that you read and follow the [Contributions Guide](https://github.com/Mindful-AI-Assistants/.github/blob/9e7e98f98af07a1d6c4bdeb349e1a9db04f8ed0e/CONTRIBUTIBNG.md).



## [💻 Git Commands]()

Create a new branch:

 ```
git checkout -b feature/my-feature
```

### Add changes to staging area: git add

Commit changes:
```
git commit -m "feat: Implemented new feature"
```

Push changes to remote: 

```
git push origin feature/my-feature
```

Create a pull request

```
[Repo Link]
(https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/edit/main/READ)
```

Merge changes:

```
git merge feature/my-feature
```

Delete branch:

```
git branch -d feature/my-feature
```

## [👩🏽‍💻 Codes and Data Analysis Report]()

### Codes:

- [DiamondValuationPredictio.py](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/50d1ec6c1074be9848ad472ff1955aad541d1ae2/Codes/Codes%20English/Codes%20Portugues/Codes%20%20English/diamondvaluationenglish.py)

- [DiamondValuationPrediction.ipynb](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/1770a467185ea56dffdede68b81f730bcff8ead4/Codes/Codes%20English/Codes%20Portugues/Codes%20%20English/diamondValuationEnglish.ipynb)

<br>

### Dataset:

- [Dataset Diamond Valuation - from kaggle](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/57c0e19bc9bd3efbdd8256442129261df9ac358f/Database/Diamonds_limpa%20(1).csv)

<br>

### Data Analysis Report:  

- [Data Analysis Report](https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/blob/86c2111a01c153279e7e8a7744f398041ef5d35b/Data%20Analyze%20Report/Data%20Analyse%20English/Data%20Analyse%20English.pdf)



## [👑 Access the Streamlit Site]()

<br>

🚀 [Tap here and teleport to the Streamlit Site](https://diamondsvalues.streamlit.app/) 


## [pQR Codes]()

### <p align="center"> 👑 QR Code of the Site on Streamlit </p>

<p align="center">
    <img src="https://github.com/Mindful-AI-Assistants/DiamondValuationPrediction/assets/113218619/e52ee3b6-6260-4733-8a65-5c7e2bc8c5d7" alt="QR Code 1" width="200"/>
</p>

 <br> 

### <p align="center">:octocat: QR Code of the GitHub Repository </p>

<p align="center"> 
<img src="https://github.com/user-attachments/assets/aa52cd93-ac2d-4728-8c53-a1ed18470905" alt="QR Code 2" width="200"/> </p>



  <br> 

<!--  
<p align="center"> 👑🇺🇸 QR Code of the Site on Streamlit (English)  (Portuguese) </p>

<p align="center">
  <img src="" />
  </p>
-->




## [👥 Our Team]() 

- [Fabiana 🚀 Campanari](https://github.com/FabianaCampanari)

- [Pedro Vyctor Carvalho De Almeida](https://github.com/ppvyctor)



## [🤝 Codes odf Conduct]()

We are committed to fostering a welcoming and inclusive community for all team members. We expect everyone to adhere to the following principles:

- Be respectful: Treat others with courtesy and respect, regardless of their background, identity, or opinions.

- Be constructive: Focus on providing helpful feedback and constructive criticism.

- Be open-minded: Be open to different perspectives and ideas.

- Be open-minded: Be open to different perspectives and ideas.

- Be accountable: Take responsibility for your actions and words.

- Be inclusive: Promote a welcoming and inclusive environment for everyone.

If you witness any violation of this code of conduct, please contact [your contact information] so we can address the situation appropriately.



## [💌 Contact]()

For more information, contact [Mindful-AI-Assistants](mailto:fabicampanari@proton.me)
  

#

###### <p align="center">[Copyright 2024 Mindful-AI-Assistants. Code released under the  MIT license.]( https://github.com/Mindful-AI-Assistants/.github/blob/ad6948fdec771e022d49cd96f99024fcc7f1106a/LICENSE)
