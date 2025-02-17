import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import datetime
import requests
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import math

# ====================================================================
# 1. Carregamento e Pré‑processamento da Base Histórica
# ====================================================================

# Tenta carregar o arquivo CSV com os dados históricos
try:
    df = pd.read_csv("base_preco_petroleo.csv")
except Exception as e:
    st.error("Erro ao carregar a base histórica. Verifique se o arquivo 'base_preco_petroleo.csv' está disponível.")
    st.stop()

# Converte a coluna 'Data' para datetime (considerando o formato dia/mês/ano)
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
# Ordena os dados pela data e reseta o índice
df.sort_values('Data', inplace=True)
df.reset_index(drop=True, inplace=True)

# Converte os preços: remove espaços, substitui vírgula por ponto e converte para float
df['Preco do Petroleo'] = (
    df['Preco do Petroleo']
    .astype(str)
    .str.strip()
    .str.replace(',', '.')
    .astype(float)
)

# ====================================================================
# 2. Carregamento do Modelo e Preparação para Previsão
# ====================================================================

# Tamanho da sequência usado no treinamento (conforme seu notebook: 10)
SEQUENCE_LENGTH = 10

# Tenta carregar o modelo salvo
try:
    model = load_model("meu_modelo.h5", compile=False)
except Exception as e:
    st.error(f"Erro ao carregar o modelo LSTM. Verifique se o arquivo 'meu_modelo.h5' está disponível.\nDetalhes: {e}")
    st.stop()

# Normaliza os preços utilizando o MinMaxScaler (mesmo procedimento usado no treinamento)
scaler = MinMaxScaler()
prices = df['Preco do Petroleo'].values.reshape(-1, 1)
prices_normalized = scaler.fit_transform(prices)

# Função que, a partir dos últimos dados normalizados, gera uma previsão para N passos à frente
def predict_future(model, data, num_prediction, sequence_length, scaler):
    # Inicia com os últimos "sequence_length" pontos (convertendo-os para float)
    prediction_list = [float(item) for item in data[-sequence_length:]]
    for _ in range(num_prediction):
        x = np.array(prediction_list[-sequence_length:], dtype=float).reshape((1, sequence_length, 1))
        out = model.predict(x)[0][0]
        prediction_list.append(out)
    # Seleciona apenas os valores previstos (excluindo os dados de entrada)
    prediction_array = np.array(prediction_list[sequence_length:]).reshape(-1, 1)
    # Desnormaliza as previsões
    prediction_array = scaler.inverse_transform(prediction_array)
    return prediction_array

# Função para prever o preço para uma data selecionada pelo usuário
def predict_price_for_date(selected_date):
    """
    Se a data selecionada for anterior (ou igual) à última data histórica,
    retorna o preço histórico (ou o mais próximo). Caso contrário,
    calcula quantos dias faltam e gera a previsão para o último dia.
    """
    selected_date = pd.to_datetime(selected_date)
    last_date = df['Data'].iloc[-1]
    if selected_date <= last_date:
        # Retorna o preço histórico (para datas existentes, pode-se aprimorar a busca)
        price_val = df[df['Data'] <= selected_date]['Preco do Petroleo'].iloc[-1]
        return np.round(price_val, 2)
    else:
        # Número de dias a prever (diferença entre a data selecionada e a última data histórica)
        delta_days = (selected_date - last_date).days
        forecast = predict_future(model, prices_normalized, delta_days, SEQUENCE_LENGTH, scaler)
        predicted_value = forecast[-1, 0]
        return np.round(predicted_value, 2)

# ====================================================================
# 3. Estrutura do App Streamlit (Abas)
# ====================================================================

st.title("Tech Challenge 4 - Previsão do Preço do Petróleo")
st.markdown("**Usuário:** vs-pereira | **Repositório:** [Tech-Challenge-4](https://github.com/vs-pereira/Tech-Challenge-4)")

# Menu lateral para selecionar a aba
abas = ["Contexto", "Dashboard", "Metodologia", "Resultados", "Simulação"]
aba_selecionada = st.sidebar.selectbox("Escolha uma aba", abas)

if aba_selecionada == "Contexto":
    st.header("Contexto")
    st.write("""
    **Problema do Tech Challenge:**
    - Você foi contratado(a) para uma consultoria que analisa os dados históricos do preço do petróleo.
    - A base de dados possui duas colunas: data e preço (em dólares).
    - O desafio consiste em desenvolver um dashboard interativo que gere insights relevantes para a tomada de decisão, além de um modelo de Machine Learning para previsão dos preços do petróleo.
    
    **Contextualização da Variação dos Preços de Petróleo:**
    - Os preços do petróleo são influenciados por fatores como eventos geopolíticos, crises econômicas e a demanda global por energia.
    - Essas variações refletem a complexidade do mercado internacional e sua dinâmica em resposta a acontecimentos globais.
    """)
    st.write("Para mais detalhes, consulte o notebook [Tech_Challenge_4_.ipynb](https://github.com/vs-pereira/Tech-Challenge-4/blob/main/Tech_Challenge_4_.ipynb).")

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

elif aba_selecionada == "Metodologia":
    st.header("Metodologia do Modelo Desenvolvido")
    st.write("""
    **Abordagem via LSTM:**
    - Utilizei uma rede neural LSTM (Long Short-Term Memory) para capturar as dependências temporais na série histórica do preço do petróleo.
    - Os dados foram previamente normalizados e transformados em sequências para treinamento.
    - O modelo foi treinado utilizando 80% dos dados para treinamento e 20% para teste, garantindo sua robustez.
    - Todo o processo foi realizado no Google Colab; o código completo encontra-se no notebook [Tech_Challenge_4_.ipynb](https://github.com/vs-pereira/Tech-Challenge-4/blob/main/Tech_Challenge_4_.ipynb).
    """)

elif aba_selecionada == "Resultados":
    st.header("Resultados")
    st.write("""
    **Resultados Alcançados:**
    - O modelo LSTM apresentou alta precisão, com R² elevado e baixas métricas de erro (MSE, MAE, MAPE e RMSE).
    - Esses resultados indicam que o modelo é capaz de prever com confiabilidade o preço do petróleo.
    
    **Detalhes das Métricas:**
    - *R² Score:* 0.9138
    - *MSE:* 2.6362
    - *MAE:* 1.2833
    - *MAPE:* 1.6196%
    - *RMSE:* 1.6236
    
    Para mais detalhes, consulte o anexo em PDF disponível no repositório.
    """)

def predict_future_for_date(selected_date, num_prediction):
    """
    Gera previsões para os próximos 'num_prediction' dias a partir dos dados históricos 
    até a data 'selected_date'.
    """
    # Converte a data recebida para datetime (caso não seja)
    selected_date = pd.to_datetime(selected_date)
    
    # Filtra os dados históricos até 'selected_date'
    df_filtered_date = df[df['Data'] <= selected_date].copy()
    if len(df_filtered_date) < SEQUENCE_LENGTH:
        st.error("Não há dados históricos suficientes até a data selecionada.")
        st.stop()
    
    # Obtém os últimos 'SEQUENCE_LENGTH' preços dos dados filtrados
    last_prices = df_filtered_date['Preco do Petroleo'].values[-SEQUENCE_LENGTH:]
    last_prices_normalized = scaler.transform(last_prices.reshape(-1, 1))
    
    # Cria a lista de previsão iniciando a partir da sequência obtida
    prediction_list = list(last_prices_normalized.flatten())
    for _ in range(num_prediction):
        x_input = np.array(prediction_list[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
        out = model.predict(x_input)[0][0]
        prediction_list.append(out)
    
    # Seleciona apenas os valores previstos e desnormaliza-os
    forecast = scaler.inverse_transform(np.array(prediction_list[SEQUENCE_LENGTH:]).reshape(-1, 1))
    return forecast

elif aba_selecionada == "Simulação":
    st.header("Simulação - Previsão do Preço do Petróleo")
    st.write("""
    Insira a data desejada para a previsão do preço do barril de petróleo.
    O intervalo permitido é dos 15 dias seguintes a 10/02/2025 (ou seja, de 11/02/2025 a 25/02/2025).
    """)
    
    # Define o intervalo permitido para a previsão
    min_date = datetime.date(2025, 2, 11)
    max_date = datetime.date(2025, 2, 25)
    data_simulacao = st.date_input("Selecione a data para previsão", 
                                   value=min_date, 
                                   min_value=min_date, 
                                   max_value=max_date)
    
    if st.button("Prever"):
        # Última data histórica (assumindo que df['Data'] esteja ordenado)
        last_hist_date = pd.to_datetime(df['Data'].iloc[-1])  # ex.: 10/02/2025
        selected_date_dt = pd.to_datetime(data_simulacao)
        num_days_forecast = (selected_date_dt - last_hist_date).days
        if num_days_forecast <= 0:
            st.error("Selecione uma data após o último ponto do histórico (10/02/2025).")
            st.stop()
        
        # Gera a previsão para os dias a partir de last_hist_date + 1 até a data selecionada.
        # A função predict_future_for_date utiliza os dados históricos até last_hist_date.
        forecast_array = predict_future_for_date(last_hist_date, num_days_forecast)
        # A previsão para o dia selecionado é o último elemento do array forecast_array.
        predicted_price = forecast_array[-1, 0]
        
        # Formata a data para dd/mm/aaaa
        formatted_date = selected_date_dt.strftime("%d/%m/%Y")
        st.write(f"A previsão do preço do petróleo para **{formatted_date}** é de **US$ {predicted_price:.2f}**.")
        
        # Prepara o gráfico
        # Histórico: dados desde 01/06/2020 até a data selecionada
        start_hist_plot = pd.to_datetime("2020-06-01")
        df_plot = df[(df['Data'] >= start_hist_plot) & (df['Data'] <= selected_date_dt)].copy()
        
        # Datas da previsão: de last_hist_date + 1 até selected_date
        forecast_dates = pd.date_range(start=last_hist_date + datetime.timedelta(days=1),
                                       periods=num_days_forecast)
        
        # Cria o gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_plot['Data'], df_plot['Preco do Petroleo'], marker='o', label="Histórico", color="blue")
        ax.plot(forecast_dates, forecast_array, marker='x', linestyle='--', label="Previsão", color="green")
        ax.set_xlabel("Data")
        ax.set_ylabel("Preço (US$)")
        ax.set_title("Previsão do Preço do Petróleo")
        
        # Formata o eixo x para mostrar datas no formato dd/mm/aaaa
        from matplotlib.dates import DateFormatter
        ax.xaxis.set_major_formatter(DateFormatter("%d/%m/%Y"))

# ====================================================================
# 4. Rodapé e Imagem
# ====================================================================

try:
    image_url = "https://raw.githubusercontent.com/vs-pereira/Tech-Challenge-4/main/foto%20capa.jpg"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    st.image(image, width=85, use_column_width=True)
except Exception as e:
    st.write("Imagem não disponível.")

st.markdown('<p class="footer">Tech Challenge 4 - vs-pereira</p>', unsafe_allow_html=True)
