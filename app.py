import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
import os
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model

local_model_path = "meu_modelo.h5"

# Sequência conforme utilizado no treinamento do seu modelo
SEQUENCE_LENGTH = 60

# Título principal e informações do repositório
st.title("Tech Challenge 4 - Previsão do Preço do Petróleo")
st.markdown("**Usuário:** vs-pereira | **Repositório:** [Tech-Challenge-4](https://github.com/vs-pereira/Tech-Challenge-4)")

# Menu lateral para selecionar a aba
abas = ["Contexto", "Dashboard", "Metodologia", "Resultados", "Simulação"]
aba_selecionada = st.sidebar.selectbox("Escolha uma aba", abas)

# Função para previsão utilizando o modelo LSTM carregado
def predict_future_price(selected_date):
    """
    Simula a previsão do preço do petróleo.
    Aqui, para exemplificação, geramos uma sequência dummy aleatória.
    Em um cenário real, essa função deve:
      - Obter os dados históricos relevantes até 'selected_date'
      - Realizar o pré-processamento (normalização, criação de sequências, etc.)
      - Gerar a sequência de entrada de forma consistente com o treinamento
    """
    # Cria uma sequência dummy com valores aleatórios para simulação
    dummy_sequence = np.random.rand(1, SEQUENCE_LENGTH, 1)
    pred = model.predict(dummy_sequence)
    return np.round(pred[0, 0], 2)

# Aba 1: Contexto
if aba_selecionada == "Contexto":
    st.header("Contexto")
    st.write("""
    **Problema do Tech Challenge:**
    - Você foi contratado(a) para uma consultoria que analisa os dados históricos do preço do petróleo.
    - A base de dados possui duas colunas: data e preço (em dólares).
    - O desafio consiste em desenvolver um dashboard interativo que gere insights relevantes para a tomada de decisão, além de um modelo de Machine Learning para previsão dos preços do petróleo.
    
    **Contextualização da Variação dos Preços de Petróleo:**
    - Os preços do petróleo são influenciados por fatores como eventos geopolíticos, crises econômicas, e a demanda global por energia.
    - Essas variações refletem a complexidade do mercado internacional e sua dinâmica em resposta a acontecimentos globais.
    """)
    st.write("Para mais detalhes, consulte o notebook [Tech_Challenge_4_.ipynb](https://github.com/vs-pereira/Tech-Challenge-4/blob/main/Tech_Challenge_4_.ipynb).")

# Aba 2: Dashboard
elif aba_selecionada == "Dashboard":
    st.header("Dashboard")
    st.write("""
    **Link do Dashboard:**  
    [Insira aqui o link do seu dashboard]
    
    **5 Insights (a serem definidos):**
    1. Insight 1: _[Descrição do insight 1]_.
    2. Insight 2: _[Descrição do insight 2]_.
    3. Insight 3: _[Descrição do insight 3]_.
    4. Insight 4: _[Descrição do insight 4]_.
    5. Insight 5: _[Descrição do insight 5]_.
    """)

# Aba 3: Metodologia do Modelo Desenvolvido
elif aba_selecionada == "Metodologia":
    st.header("Metodologia do Modelo Desenvolvido")
    st.write("""
    **Abordagem via LSTM:**
    - Utilizei uma rede neural LSTM (Long Short-Term Memory) para capturar as dependências temporais na série histórica do preço do petróleo.
    - Os dados foram previamente normalizados e transformados em sequências para treinamento.
    - O modelo foi treinado utilizando 80% dos dados para treinamento e 20% para teste, garantindo sua robustez.
    - Todo o processo foi realizado no Google Colab e o código completo encontra-se no notebook [Tech_Challenge_4_.ipynb](https://github.com/vs-pereira/Tech-Challenge-4/blob/main/Tech_Challenge_4_.ipynb).
    """)

# Aba 4: Resultados
elif aba_selecionada == "Resultados":
    st.header("Resultados")
    st.write("""
    **Resultados Alcançados:**
    - O modelo LSTM apresentou alta precisão, com R² elevado e baixas métricas de erro (MSE, MAE, MAPE e RMSE).
    - Esses resultados indicam que o modelo é capaz de prever com confiabilidade o preço do petróleo.
    
    **Detalhes das Métricas:**
    - *R² Score:* [Insira o valor]
    - *MSE:* [Insira o valor]
    - *MAE:* [Insira o valor]
    - *MAPE:* [Insira o valor]
    - *RMSE:* [Insira o valor]
    
    Para mais detalhes, consulte o anexo em PDF disponível no repositório.
    """)

# Aba 5: Simulação
elif aba_selecionada == "Simulação":
    st.header("Simulação - Previsão do Preço do Petróleo")
    st.write("""
    **Simulação:**
    Insira uma data para obter a previsão do preço do petróleo com base no modelo LSTM.
    """)
    
    # Interface para simulação
    data_simulacao = st.date_input("Selecione a data para previsão", value=datetime.date.today())
    if st.button("Prever"):
        previsao = predict_future_price(data_simulacao)
        st.write(f"A previsão do preço do petróleo para **{data_simulacao}** é de **US$ {previsao}**.")
        
        # Exemplo de gráfico para visualização (simulação)
        datas = pd.date_range(start=data_simulacao, periods=5, freq='D')
        valores = np.random.uniform(50, 100, size=5)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(datas, valores, marker='o', linestyle='--', color='green')
        ax.set_xlabel("Data")
        ax.set_ylabel("Preço (US$)")
        ax.set_title("Simulação de Previsão do Preço do Petróleo")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Rodapé opcional com imagem ou logo
try:
    image_url = "https://raw.githubusercontent.com/vs-pereira/Tech-Challenge-4/main/foto%20capa.jpg"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    st.image(image, width=85, output_format="PNG", use_column_width=True)
except Exception as e:
    st.write("Imagem não disponível.")

st.markdown('<p class="footer">Tech Challenge 4 - vs-pereira</p>', unsafe_allow_html=True)
