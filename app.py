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
    st.write("""
    ### Origem e Análise dos Dados

    Os dados históricos do preço do petróleo foram obtidos de fontes públicas e confiáveis, que agregam informações de diversos mercados internacionais. Após a extração, os dados foram cuidadosamente processados e armazenados em um arquivo Microsoft Excel. Em seguida, utilizamos Python, com as bibliotecas **Pandas** e **NumPy**, para realizar uma análise exploratória detalhada. Essa etapa envolveu a limpeza dos dados, a correção de inconsistências, a identificação de outliers e a análise de tendências e sazonalidades que influenciam o comportamento do mercado.

    Visualizações interativas foram construídas para examinar a evolução do preço do petróleo ao longo do tempo, permitindo a identificação de padrões e a compreensão das dinâmicas de oferta e demanda, bem como dos impactos de eventos geopolíticos e crises econômicas.

    ### Aplicação do Modelo LSTM

    Para a previsão dos preços futuros, implementamos uma rede neural do tipo **LSTM** (Long Short-Term Memory), reconhecida por sua capacidade de capturar dependências temporais e relacionamentos de longo prazo em séries temporais. Os dados foram previamente normalizados utilizando o **MinMaxScaler**, garantindo que a rede neural trabalhasse com uma escala consistente, o que é crucial para o desempenho do modelo.

    O conjunto de dados foi dividido em 80% para treinamento e 20% para validação, permitindo uma avaliação robusta da performance do modelo. Foram calculadas diversas métricas de desempenho, incluindo:
    
    - **Coeficiente de Determinação (R²):** Mede a proporção da variância dos dados que é explicada pelo modelo. Valores próximos de 1 indicam um ajuste excelente.
    - **Erro Médio Quadrático (MSE):** Indica a média dos quadrados dos erros de previsão, destacando a dispersão dos resíduos.
    - **Erro Médio Absoluto (MAE):** Fornece a média dos erros absolutos, representando a magnitude média dos desvios.
    - **Erro Percentual Absoluto Médio (MAPE):** Expressa a precisão das previsões em termos percentuais.
    - **Raiz do Erro Médio Quadrático (RMSE):** É a raiz quadrada do MSE, interpretada nas mesmas unidades do preço.

    Esses indicadores foram essenciais para validar a eficácia do modelo, garantindo que as previsões geradas fossem confiáveis para a tomada de decisões estratégicas.

    ### Execução no Google Colab

    Todo o processo de análise e treinamento do modelo foi conduzido no **Google Colab**, uma plataforma que oferece acesso gratuito a recursos computacionais robustos, incluindo GPUs. Essa infraestrutura permitiu a execução eficiente do código Python, facilitou a colaboração entre pesquisadores e possibilitou a manipulação de grandes conjuntos de dados com rapidez e precisão.

    ### Integração e Aplicação

    A integração da análise detalhada dos dados históricos com o modelo LSTM permitiu o desenvolvimento de um dashboard interativo que une storytelling e análises preditivas avançadas. Essa solução oferece insights valiosos sobre as oscilações do preço do petróleo, permitindo que empresas, investidores e formuladores de políticas antecipem tendências, mitiguem riscos e identifiquem oportunidades no mercado energético.
    """)


elif aba_selecionada == "Resultados":
    st.header("Resultados")
    st.write("""
    **Resultados Alcançados:**
    - O modelo LSTM apresentou alta precisão com R² elevado e baixas métricas de erro.
    
    **Métricas:**
    - *R² Score:* 0.8693
    - *MSE:* 3.9970
    - *MAE:* 1.6485
    - *MAPE:* 2.0504%
    - *RMSE:* 1.9993
    
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
