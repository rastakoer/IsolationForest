import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------
# Config streamlit
#-----------------------------------------------------------------------------------
st.set_page_config(page_title="Isolation", page_icon=":tada:", layout="wide")


# ---------------------------------------------------------------------
# SIDEBAR
#-----------------------------------------------------------------------
st.sidebar.header("Isolation d\'anomalie")
nb_estimateurs = st.sidebar.slider('Nombre d\'estimateurs ',0,100,50)
contamine=st.sidebar.slider('Pourcentage d\'anomalie',0.01,0.20,0.10)
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier csv", type="csv")

# ---------------------------------------------------------------------
# Page principale
#-----------------------------------------------------------------------
if uploaded_file is not None:
    # CHARGEMENT DU DATAFRAME
    data = pd.read_csv(uploaded_file)
    # Nettoyage du dataframe à faire  
    # + valeur non numérique à transformer.......
    # => Quand on aura le temps d'approfondir le sujet
    # AFFICHADE DU DATASET
    st.subheader(f"Nombre de lignes dans le dataset avant modification : {data.shape[0]}")
    st.dataframe(data)

    # RECUPERATION DES PARAMETRES ET ENTRAINEMENT DE MODELE
    model = IsolationForest(n_estimators=nb_estimateurs,contamination=contamine, 
                            max_samples = 100, random_state = 42)
    model.fit(data)
    y_ano = model.predict(data)
    y_ano = pd.DataFrame(y_ano, columns = ['ANOMALIE'])

    # CREATION ET AFFICHAGE DES ANOMALIES
    anomalie = data.iloc[y_ano[y_ano['ANOMALIE'] == -1].index.values]
    anomalie.reset_index(drop = True, inplace = True)
    st.subheader(f"Nombre d\'anomalies: {anomalie.shape[0]}")
    st.dataframe(anomalie)
    
    # CREATION ET AFFICHAGE DU DATASET SANS ANOMALIES
    sans_anomalie = data.iloc[y_ano[y_ano['ANOMALIE'] == 1].index.values]
    sans_anomalie.reset_index(drop = True, inplace = True)
    st.subheader(f"Lignes du dataset sans anomalies: {sans_anomalie.shape[0]}")
    st.dataframe(sans_anomalie)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.download_button(label="Sauvegarder les données sans anomalie",
                            data=sans_anomalie.to_csv().encode('utf-8'),
                            file_name='csv_clean.csv',
                            mime='text/csv',
                            )

else :
    st.header("Detection d\'anomalie dans un dataset à l\'aide d\'IsolationForest")
    st.markdown("- Telecharger un fichier csv avec uniquement des données numériques, l'appli sera mise à jour lors d'une étude plus appronfondie")
    st.markdown("- Selectionner le nombre d'estimateurs ansi que le pourcentage d'outliers voulu")


        
        
        
