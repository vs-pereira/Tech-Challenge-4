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
from matplotlib.dates import DateFormatter

# ====================================================================
# 1. Carregamento e Pré‑processamento da Base Histórica
# ====================================================================

try:
    df = pd.read_csv("base_preco_petroleo.csv")
except Exception as e:
    st.error("Erro ao carregar a base histórica. Verifique se o arquivo 'base_preco_petroleo.csv' está disponível.")
    st.stop()

df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
df.sort_values('Data', inplace=True)
df.reset_index(drop=True, inplace=True)
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

SEQUENCE_LENGTH = 10

try:
    model = load_model("meu_modelo.h5", compile=False)
except Exception as e:
    st.error(f"Erro ao carregar o modelo LSTM. Verifique se o arquivo 'meu_modelo.h5' está disponível.\nDetalhes: {e}")
    st.stop()

scaler = MinMaxScaler()
prices = df['Preco do Petroleo'].values.reshape(-1, 1)
prices_normalized = scaler.fit_transform(prices)

def predict_future(model, data, num_prediction, sequence_length, scaler):
    prediction_list = [float(item) for item in data[-sequence_length:]]
    for _ in range(num_prediction):
        x = np.array(prediction_list[-sequence_length:], dtype=float).reshape((1, sequence_length, 1))
        out = model.predict(x)[0][0]
        prediction_list.append(out)
    prediction_array = np.array(prediction_list[sequence_length:]).reshape(-1, 1)
    prediction_array = scaler.inverse_transform(prediction_array)
    return prediction_array

def predict_price_for_date(selected_date):
    selected_date = pd.to_datetime(selected_date)
    last_date = df['Data'].iloc[-1]
    if selected_date <= last_date:
        price_val = df[df['Data'] <= selected_date]['Preco do Petroleo'].iloc[-1]
        return np.round(price_val, 2)
    else:
        delta_days = (selected_date - last_date).days
        forecast = predict_future(model, prices_normalized, delta_days, SEQUENCE_LENGTH, scaler)
        predicted_value = forecast[-1, 0]
        return np.round(predicted_value, 2)

def predict_future_for_date(selected_date, num_prediction):
    """
    Gera previsões para os próximos 'num_prediction' dias a partir dos dados históricos 
    até a data 'selected_date'.
    """
    selected_date = pd.to_datetime(selected_date)
    df_filtered_date = df[df['Data'] <= selected_date].copy()
    if len(df_filtered_date) < SEQUENCE_LENGTH:
        st.error("Não há dados históricos suficientes até a data selecionada.")
        st.stop()
    last_prices = df_filtered_date['Preco do Petroleo'].values[-SEQUENCE_LENGTH:]
    last_prices_normalized = scaler.transform(last_prices.reshape(-1, 1))
    prediction_list = list(last_prices_normalized.flatten())
    for _ in range(num_prediction):
        x_input = np.array(prediction_list[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 1)
        out = model.predict(x_input)[0][0]
        prediction_list.append(out)
    forecast = scaler.inverse_transform(np.array(prediction_list[SEQUENCE_LENGTH:]).reshape(-1, 1))
    return forecast

# ====================================================================
# 3. Estrutura do App Streamlit (Abas)
# ====================================================================

st.title("Previsão do Preço do Petróleo")
st.markdown("**Usuário:** vs-pereira | **Repositório:** [Tech-Challenge-4](https://github.com/vs-pereira/Tech-Challenge-4)")

abas = ["Contexto", "Dashboard", "Metodologia", "Resultados", "Simulação"]
aba_selecionada = st.sidebar.selectbox("Escolha uma aba", abas)

if aba_selecionada == "Contexto":
    st.header("Explorando as Profundezas das Oscilações no Mercado do Petróleo")
    st.write("""
    O mercado global de petróleo é uma arena dinâmica e multifacetada, onde cada variação no preço do barril reflete um intricado jogo de forças econômicas, políticas e tecnológicas. As flutuações observadas não são números isolados, mas o resultado de interações complexas entre eventos geopolíticos, crises financeiras, transformações na demanda energética e inovações disruptivas na cadeia produtiva do petróleo.
    """)
    
    # Exibe imagem ilustrativa sobre o mercado de petróleo
    st.image("https://raw.githubusercontent.com/vs-pereira/Tech-Challenge-4/main/imagem_petroleo.jpg", 
             caption="Oscilações no Mercado do Petróleo", width=500)
    
    st.write("""
    Este dashboard interativo oferece uma análise aprofundada desses fenômenos, explorando desde a influência dos conflitos internacionais e mudanças nas políticas energéticas até o impacto de avanços tecnológicos que redefinem tanto a extração quanto a utilização dessa commodity estratégica. A combinação de dados históricos e modelos avançados de machine learning possibilita identificar padrões e prever tendências de curto prazo com alta confiabilidade, convertendo dados complexos em insights claros e acionáveis para investidores, gestores e formuladores de políticas.
    """)
    
    st.write("""
    Para uma compreensão integrada do mercado, utilizamos ferramentas poderosas para a visualização e análise dos dados. A estrutura do MVP foi construída com uma abordagem que alia storytelling e análise quantitativa, possibilitando uma tomada de decisão mais informada e estratégica em um cenário global em constante transformação.
    """)
    
    # Exibe imagem ilustrativa da indústria do petróleo
    st.image("https://raw.githubusercontent.com/vs-pereira/Tech-Challenge-4/main/imagem_industria.jpg", 
             caption="Indústria do Petróleo", width=500)
    
    st.write("Utilizamos também as seguintes ferramentas para compor nosso dashboard e plataforma interativos:")
    
    # Exibe imagens das ferramentas utilizadas
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://raw.githubusercontent.com/vs-pereira/Tech-Challenge-4/main/imagem_lookerstudio.jpg", 
                 caption="Looker Studio", width=200)
    with col2:
        st.image("https://raw.githubusercontent.com/vs-pereira/Tech-Challenge-4/main/imagem_streamlit.jpg", 
                 caption="Streamlit", width=200)
    
    st.write("Para mais detalhes, consulte o notebook [Tech_Challenge_4_.ipynb](https://github.com/vs-pereira/Tech-Challenge-4/blob/main/Tech_Challenge_4_.ipynb).")

elif aba_selecionada == "Dashboard":
    st.header("Dashboard")
    
    # Exibe a imagem do dashboard (certifique-se de que "dash.png" esteja no repositório)
    st.image("dash.png", use_container_width=True)
    
    st.write("""
    **Link do Dashboard:**  
    [Acesse o Dashboard Interativo](https://lookerstudio.google.com/reporting/03e2fa7d-0f1f-44c3-ad76-c381c03d85a0/page/qlD)
    
    **Insights Sobre a Variação dos Preços do Petróleo (últimos 20 anos):**

    1. **2008: Máxima Alta e Queda Abrupta**  
       Em 2008, o mercado experimentou extremos – atingindo um pico de USD143,95 e, logo em seguida, uma queda para USD33,73.  
       - **Fatores:**  
         - Período de alta especulativa e demanda aquecida que elevou os preços antes da crise.  
         - A crise financeira global reduziu drasticamente a demanda e criou um excesso de oferta.

    2. **2011 – 2013: Altas Médias de Preços**  
       Durante esse período, os preços médios anuais se mantiveram elevados devido a uma combinação de fatores.  
       - **Fatores:**  
         - Instabilidade geopolítica provocada pela Primavera Árabe, que gerou incertezas nas regiões produtoras.  
         - Políticas da OPEP que restringiram a oferta em meio a temores de interrupções no fornecimento.  
         - Demanda global robusta, especialmente de economias emergentes.
    
    3. **2016: Segundo Menor Preço**  
       Em 2016, o preço do petróleo caiu para cerca de USD26,01, refletindo um período de reequilíbrio no mercado.  
       - **Fatores:**  
         - Aumento da produção de shale oil nos EUA e relutância inicial da OPEP em cortar a produção levaram a um excesso de oferta.  
         - Incertezas macroeconômicas globais afetaram a demanda por energia.  
         - Mercado ainda se recuperando da queda iniciada em 2014.

    4. **2020: Menor Preço da Série**  
       O ano de 2020 ficou marcado pelo menor preço já registrado, em torno de USD9,12, devido a uma combinação sem precedentes de fatores.  
       - **Fatores:**  
         - A pandemia de COVID-19 resultou em lockdowns globais e queda acentuada na demanda.  
         - Guerra de preços entre grandes produtores agravou o excesso de oferta.  
         - Problemas logísticos e de armazenamento contribuíram para a queda.
    
    5. **2022: Segunda Maior Alta**  
       Em 2022, os preços subiram para USD133,18, impulsionados por fatores que alteraram significativamente o equilíbrio entre oferta e demanda.  
       - **Fatores:**  
         - Conflito na Ucrânia e sanções à Rússia reduziram a oferta global.  
         - Recuperação econômica pós-pandemia impulsionou a demanda por energia.  
         - Decisões da OPEP+ para manter cortes na produção.
    """)

elif aba_selecionada == "Metodologia":
    st.header("Metodologia do Modelo Desenvolvido")
    
    # Exibe uma imagem ilustrativa do Python, que deve estar no repositório
    st.image("https://raw.githubusercontent.com/vs-pereira/Tech-Challenge-4/main/python_image.png", use_container_width=True)
    
    st.write("""
    ### Origem e Análise dos Dados

    Os dados históricos do preço do petróleo utilizados nesta análise foram extraídos do IPEA, abrangendo o período de **20/05/1987 até 10/02/2025**. Esse conjunto de dados reúne informações diárias que refletem as variações do preço do barril ao longo dos anos, permitindo uma análise aprofundada das tendências e dos fatores que influenciam esse mercado.

    Inicialmente, os dados foram organizados e salvos em um arquivo Microsoft Excel, o que facilitou a sua manipulação e análise com Python. Utilizamos as bibliotecas **Pandas** e **NumPy** para realizar uma limpeza minuciosa dos dados, corrigindo inconsistências e estruturando a série temporal de forma que fosse possível identificar padrões, sazonalidades e eventos extraordinários que pudessem explicar oscilações históricas.

    Em seguida, foram construídas diversas visualizações interativas que ajudaram a demonstrar a evolução dos preços ao longo do tempo e a revelar relações entre os dados. Esse processo foi fundamental para entender como eventos econômicos e geopolíticos impactaram o mercado do petróleo.

    ### Aplicação do Modelo LSTM

    Para prever os preços futuros do petróleo, implementamos um modelo de rede neural do tipo **LSTM** (Long Short-Term Memory). Este modelo é especialmente indicado para trabalhar com dados sequenciais, pois consegue capturar dependências de longo prazo presentes em séries temporais.

    O processo de modelagem envolveu as seguintes etapas:
    
    - **Normalização dos Dados:** Utilizamos o **MinMaxScaler** para transformar os preços em uma escala entre 0 e 1. Essa etapa é crucial para garantir que o modelo LSTM opere com dados em uma faixa uniforme, facilitando o treinamento e melhorando a estabilidade dos cálculos.
    
    - **Criação de Sequências:** A série histórica foi dividida em sequências de um comprimento fixo (no nosso caso, 10 dias). Cada sequência serve como entrada para o modelo, enquanto o valor seguinte na sequência é utilizado como saída esperada. Essa abordagem permite ao modelo aprender padrões temporais e fazer previsões baseadas em janelas deslizantes.
    
    - **Divisão dos Dados:** Os dados foram divididos em 80% para treinamento e 20% para teste. Essa separação assegura que o modelo seja treinado com a maioria dos dados disponíveis e, ao mesmo tempo, validado com um subconjunto separado para avaliar sua performance.
    
    - **Treinamento do Modelo:** Com o conjunto de dados preparado, treinamos a rede LSTM, ajustando os pesos e viéses para minimizar a função de perda (Erro Quadrático Médio, MSE). O treinamento foi realizado no **Google Colab**, que oferece recursos computacionais avançados (incluindo GPUs) para acelerar o processo.

    - **Avaliação do Modelo:** Após o treinamento, diversas métricas foram calculadas para avaliar a performance do modelo, incluindo:
        - **Coeficiente de Determinação (R²)**
        - **Erro Médio Quadrático (MSE)**
        - **Erro Médio Absoluto (MAE)**
        - **Erro Percentual Absoluto Médio (MAPE)**
        - **Raiz do Erro Médio Quadrático (RMSE)**
        
      Essas métricas ajudam a mensurar a capacidade do modelo em prever os preços e a identificar eventuais discrepâncias entre os valores reais e os previstos.

    ### Execução no Google Colab e Integração

    Toda a análise, desde a extração dos dados até o treinamento do modelo, foi realizada no **Google Colab**. Essa plataforma permitiu a execução de códigos Python de forma colaborativa e com acesso a recursos computacionais robustos, sem a necessidade de configurar um ambiente local complexo.

    Por fim, os resultados do modelo LSTM foram integrados a um dashboard interativo que combina storytelling com análises preditivas. Essa solução proporciona insights valiosos, permitindo uma compreensão aprofundada das oscilações do preço do petróleo e ajudando na tomada de decisões estratégicas no mercado energético.
    """)

elif aba_selecionada == "Resultados":
    st.header("Resultados")
    st.write("""
    **Resultados Alcançados:**

    O modelo LSTM demonstrou robustez e uma notável capacidade preditiva ao analisar os dados históricos do preço do petróleo. 
    A rede neural foi capaz de capturar as complexas dinâmicas do mercado, evidenciadas por um elevado ajuste aos dados e baixos índices de erro. 
    Esses resultados fundamentam a utilização do modelo para a previsão de preços futuros, contribuindo para decisões estratégicas em cenários de volatilidade.
    
    **Métricas de Desempenho:**
    - *R² Score:* 0.8693
    - *MSE:* 3.9970
    - *MAE:* 1.6485
    - *MAPE:* 2.0504%
    - *RMSE:* 1.9993
    
    Esses indicadores mostram que o modelo apresenta uma performance sólida e confiável para a previsão dos preços do petróleo.
    
    Para mais detalhes, consulte o anexo em PDF disponível no repositório.
    """)

elif aba_selecionada == "Simulação (deploy)":
    st.header("Simulação - Previsão do Preço do Petróleo")
    st.write("""
    Insira a data desejada para a previsão do preço do barril de petróleo.  
    *Observação:* O intervalo permitido é dos 15 dias seguintes a 10/02/2025, ou seja, de 11/02/2025 a 25/02/2025.
    """)
    
    # Define o intervalo permitido
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
        forecast_array = predict_future_for_date(last_hist_date, num_days_forecast)
        predicted_price = forecast_array[-1, 0]
        formatted_date = selected_date_dt.strftime("%d/%m/%Y")
        st.write(f"A previsão do preço do petróleo para **{formatted_date}** é de **US$ {predicted_price:.2f}**.")
        
        # Prepara o gráfico:
        # Histórico desde 01/06/2020 até a data selecionada
        start_hist_plot = pd.to_datetime("2025-01-01")
        df_plot = df[(df['Data'] >= start_hist_plot) & (df['Data'] <= selected_date_dt)].copy()
        
        # Datas da previsão: de last_hist_date + 1 até selected_date
        forecast_dates = pd.date_range(start=last_hist_date + datetime.timedelta(days=1),
                                       periods=num_days_forecast)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_plot['Data'], df_plot['Preco do Petroleo'], marker='o', label="Histórico", color="blue")
        ax.plot(forecast_dates, forecast_array, marker='x', linestyle='--', label="Previsão", color="green")
        ax.set_xlabel("Data")
        ax.set_ylabel("Preço (US$)")
        ax.set_title("Previsão do Preço do Petróleo")
        ax.xaxis.set_major_formatter(DateFormatter("%d/%m/%Y"))
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

# ====================================================================
# 4. Rodapé e Imagem
# ====================================================================

try:
    image_url = "https://raw.githubusercontent.com/vs-pereira/Tech-Challenge-4/main/foto%20capa.jpg"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    st.image(image, width=85)
except Exception as e:
    st.write("Imagem não disponível.")

st.markdown('<p class="footer">Vitor Squecola Pereira (vitor.squecola@gmail.com) e Daniel Udala (daniel.udala@outlook.com)</p>', unsafe_allow_html=True)
