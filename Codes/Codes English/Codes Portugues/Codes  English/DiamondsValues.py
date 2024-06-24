import pandas as pd
import numpy as np
import streamlit as st
import math
import requests
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from diamondvaluationenglish import cadernoJupyter

densidade = 0.0
volume = 0
carat = 0.0
depth = 0.0
table = 0.0
x = 0.0
y = 0.0
z = 0.0
price = np.nan
cut = ""
color = ""
clarity = ""

st.sidebar.title("MENU")
button1 = st.sidebar.button("Discover the value of a diamond 🤑💲")
button2 = st.sidebar.button("Precise study on diamond pricing. 📘")

for x in range(37):
    st.sidebar.write("")

st.sidebar.write("## Download used database:")

download1, download2 = st.sidebar.columns(2)

download1.download_button("Database of missing values",
                          pd.read_csv(r"DataBases/Diamonds_values_faltantes.csv").to_csv(index = False).encode("utf-8"),
                          "Diamonds_values_faltantes.csv", mime = "text/csv",
                          help = 'This is the database that has missing and erroneous values. We use this database in the option "Precise study on diamond pricing. 📘" where we treat the database and conduct a study using it.')

download2.download_button("Download Clean database", 
                          pd.read_csv(r"DataBases/Diamonds_values_faltantes.csv").to_csv(index = False).encode("utf-8"),
                          "Diamonds_limpa.csv", mime = "text/csv",
                          help = 'This is the database is the same as on the left, however this has been treated, and is now used for predictions of diamonds in the option "Find Your Diamond Value: Estimate Price Accurately! 💎".')

if button1 or (button1 == False and button2 == False):
    st.title("Discover the Value of Your Diamond: Estimate the Price Accurately! 💎\n")
    st.write("---")
    
    diamonds = pd.read_csv(r"DataBases/Diamonds_limpa.csv")
    
    # Definindo a variável cut
    aux = [x for x in list(set(diamonds["cut"].dropna()))]
    aux.insert(0, "Choose a option")
    cut = st.selectbox("Please input the diamond's cut below:", tuple(aux))
    
    
    if cut != "Choose a option":
        # Definindo uma cor ao diamante
        aux = [x for x in list(set(diamonds["color"].dropna()))]
        aux.insert(0, "Choose a option") 
        color = st.selectbox("Please input the diamond's color below:", tuple(aux))
        
        if color != "Choose a option":
            # Definindo a claridade (pureza) do diamante 
            aux = [x for x in list(set(diamonds["clarity"].dropna()))]
            aux.insert(0, "Choose a option")
            clarity = st.selectbox("Please input the diamond's clarity below:", tuple(aux))
            
            if clarity != "Choose a option":
                for _ in range(2):
                    st.write("")

                #Defina o depth (porcentagem total da profundidade) do diamante
                depth = st.number_input("Enter below the Depth (total percentage of the depth) of the diamond", min_value = 0.0, max_value=100.0)

                # Definindo um table (maior faceta plana de um diamante)
                table = st.number_input("Type below the table (largest flat facet) of the diamond", min_value = 0.0, max_value=150.00)
                
                for _ in range(2):
                    st.write("")

                # Definindo as opções de escolha de carat
                option = st.selectbox('''Choose how you want to set the diamond carat: 
                                    (NOTE: If the choice is density, it will be mandatory to enter the diamond width and depth length) *(Required)''', 
                                    ("Select a option", "Carat", "Diamonds dots (dt)", "Mass(mg) of diamond", "Density(mg/mm³) and Volume(mm³)", "Density(mg/mm³) of diamond"))
                
                if option == "Carat":
                    carat = st.number_input("Enter below the carat value of the diamond:", min_value=0.0, max_value=10.0)
                    
                elif option == "Diamonds dots (dt)":
                    carat = st.number_input("Enter below the diamond points:", help = '100pt = 1 Quilate', min_value=0, max_value=10000)
                    carat = round(carat / 100, 2)
                    
                elif option == "Mass(mg) of diamond":
                    carat = st.number_input("Enter below the diamond mass(mg):", help = "200mg = 1 Quilate", min_value=0, max_value=2000)
                    carat = round(carat/200, 2)
                
                elif option in ["Density(mg/mm³) and Volume(mm³)", "Density(mg/mm³) of diamond"] :
                    st.markdown("### **Because the choice was the density, we will need the measurements of the diamond to calculate the carat.**")
                    densidade = st.number_input("Enter below the diamond Density(Mg/mm³):", min_value=0.0)
                    
                    if densidade == 0:
                        st.write(f'The density "{densidade}" cannot be equal to 0.')
                        
                    if option == "Density(mg/mm³) and Volume(mm³)": 
                        volume = st.number_input("Enter the volume(mm³) of the diamond next to it:", min_value = 0, max_value = 20000)
                        carat = round((densidade * volume) / 200, 2)
                        

                if option == "Select a option":
                    pass
                    
                elif carat == 0.0 and option != "Density(mg/mm³) of diamond":
                    st.markdown("##### **Please set a Carat!!**")
                
                else:
                    # Definir comprimento do diamante
                    x = st.number_input("Enter below the Length (mm) of the diamond:", min_value=0.00, max_value=20.00)
                    
                    y = st.number_input("Enter below the Width (mm) of the diamond:", min_value=0.00,  max_value=20.00)
                    
                    z = st.number_input("Enter below the Diamond Depth (mm):", min_value=0.00, max_value=18.00)

                    st.write("---")
                    # A função abaixo é para prever o preço do diamante
                    st.markdown(f"## **Characteristics of registered diamond:**")
                    if cut == "": 
                        st.markdown("- Cut: ?")
                    else:
                        st.markdown(f"- Cut: {cut}")
                    if color == "":
                        st.markdown("- Color: ?")
                    else:
                        st.markdown(f"- Color: {color}")
                    if clarity == "":
                        st.markdown("- Clarity: ?")
                    else:
                        st.markdown(f"- Clarity: {clarity}")
                    st.markdown(f"- Total percentage of diamond depth: {depth}")
                    st.markdown(f"- Larger flat facet of diamond: {table}")
                    
                    if option == "Density(mg/mm³) of diamond":
                        if (x != 0.0 and y != 0.0) and z != 0.0:
                            st.markdown(f"- Carat: {round((x * y * z * densidade) / 200, 2)}")
                        else:
                            st.markdown(f"- Carat: {carat}")

                    else:
                        st.markdown(f"- Carat: {carat}")


                    st.markdown(f"- length: {x}")
                    st.markdown(f"- width: {y}")
                    st.markdown(f"- depth: {z}") 
                    
                    
                    if option == "Density(mg/mm³) of diamond":
                        if ((x == 0.0 or y == 0.0) or z == 0.0) or densidade == 0.0:
                            st.markdown("### **It is necessary to define:**")
                            if densidade == 0.0: st.markdown('- The density of the diamond.')
                            if x == 0.0: st.markdown('- The Length of the diamond.')
                            if y == 0.0: st.markdown('- The Width of the diamond.')
                            if z == 0.0: st.markdown('- The Depth of the diamond.')
                        else:
                            carat = round((x * y * z * densidade) / 200, 2)
                    else:
                        if x == 0.0: x = np.nan
                        if y == 0.0: y = np.nan
                        if z == 0.0: z = np.nan
                        
                    
                    if carat != 0.0:
                        st.write("---")
                        if depth == 0: depth = np.nan
                        if table == 0: table = np.nan
                        
                        
                        if st.button("Predict the price of diamond!! 💰💲"):
                            st.write("Analyzing the diamond to set its price")
                            st.write("")
                            
                            diamonds.loc[diamonds.shape[0]] = {"carat": carat,
                                                                    "cut": cut, "color": color, "clarity": clarity,
                                                                    "depth": depth, "table": table,
                                                                    "x": x, "y": y, "z": z}
                                            
                            for y2 in range(1, 4):
                                diamonds.iloc[:, y2] = pd.factorize(diamonds.iloc[:, y2])[0]

                            # 1. Dividir o conjunto de dados
                            diamonds_train, diamonds_test = train_test_split(diamonds, test_size=0.2, random_state=42)

                            # 2. Aplicar o KNN para imputar valores faltantes na coluna "price" do conjunto de treinamento
                            knn_imputer = KNNImputer(n_neighbors=round(math.log(diamonds.shape[0])), metric='nan_euclidean')
                            
                            # Imputar valores faltantes na coluna "price" do conjunto de teste usando o mesmo imputer
                            diamonds_train_imputed = knn_imputer.fit_transform(diamonds_train)
                            diamonds_aux = knn_imputer.fit_transform(diamonds)
                            diamonds_test_imputed = knn_imputer.transform(diamonds_test)

                            valor_diamonds = pd.DataFrame(diamonds_aux, columns = diamonds.columns)
                            # O valor calculado está em dolar, mas queremos transformar isso para real
                            
                            # API da cotação do dolar
                            respose = requests.get(r"https://economia.awesomeapi.com.br/last/USD-BRL,USD-EUR")
                            cotacao = respose.json()
                            cotacao_dolar_real = cotacao["USDBRL"]["bid"] # Valor do dolar atualmente
                            cotacao_dolar_euro = cotacao["USDEUR"]["bid"] # Valor do euro ao transformado a partir do dolar (Dolar-Euro)
                            
                            # Modificando a forma de apresentar a data do valor da cotação atribuida
                            data_dolar_real = cotacao["USDBRL"]["create_date"].split(" ")[0].split("-")
                            data_dolar_real = reversed(data_dolar_real)
                            data_dolar_real = "/".join(data_dolar_real)
                            
                            data_dolar_euro = cotacao["USDEUR"]["create_date"].split(" ")[0].split("-")
                            data_dolar_euro = reversed(data_dolar_euro)
                            data_dolar_euro = "/".join(data_dolar_euro)
                            
                            st.markdown(f'''
                            ### **The value of the diamond with the given characteristics is:**
                            - Dólar: ${round(valor_diamonds.loc[valor_diamonds.shape[0]-1, "price"], 2)} 
                            - Euro: €{round(valor_diamonds.loc[valor_diamonds.shape[0]-1, "price"] * float(cotacao_dolar_euro), 2)}
                            - Real: R${round(valor_diamonds.loc[valor_diamonds.shape[0]-1, "price"] * float(cotacao_dolar_real), 2)}''')
                            
                            left, right = st.columns(2)
                            
                            with left:
                                st.markdown(f"##### **Quotation of the Dolar-Real: Dolar-Real:**")
                                st.markdown(f'''
                                - Quotation: R$ {cotacao_dolar_real}
                                - Date: {data_dolar_real}
                                - Hour: {cotacao["USDBRL"]["create_date"].split(" ")[1]}''')
                            
                            with right:
                                st.markdown(f"##### **Quotation of the Dolar-Real: Dolar-Euro:**")
                                st.markdown(f'''
                                - Quotation: € {cotacao_dolar_euro}
                                - Date: {data_dolar_euro}
                                - Hour: {cotacao["USDEUR"]["create_date"].split(" ")[1]}''')


elif button2:
    cadernoJupyter()