# ==============================
# IMPORTAÇÕES
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==============================
# CONFIGURAÇÃO INICIAL (OBRIGATORIAMENTE PRIMEIRO)
# ==============================
st.set_page_config(
    page_title="Predição de Obesidade",
    layout="centered"
)

# ==============================
# SIDEBAR - NAVEGAÇÃO
# ==============================
aba = st.sidebar.radio(
    "Navegação",
    ["🩺 Sistema Preditivo", "📊 Dashboard Analítico"]
)

# ==============================
# CARREGAR MODELOS
# ==============================
modelo_clinico = joblib.load("modelo_clinico.pkl")
modelo_preventivo = joblib.load("modelo_preventivo.pkl")
encoder_target = joblib.load("encoder_target.pkl")

# =========================================================
# 🩺 SISTEMA PREDITIVO
# =========================================================
if aba == "🩺 Sistema Preditivo":

    st.title("🩺 Sistema de Predição de Obesidade")
    st.write(
        """
        Este sistema auxilia profissionais de saúde na identificação do nível de obesidade.
        O sistema possui dois modos:
        - **Diagnóstico Clínico** (com peso e altura)
        - **Análise Preventiva** (sem peso e altura)
        """
    )

    # ------------------------------
    # SELETOR DE MODELO
    # ------------------------------
    tipo_analise = st.selectbox(
        "Selecione o tipo de análise:",
        (
            "Diagnóstico Clínico (com peso e altura)",
            "Análise Preventiva (sem peso e altura)"
        )
    )

    st.divider()

    # ------------------------------
    # INPUTS COMUNS
    # ------------------------------
    age = st.slider("Idade", 14, 61, 30)
    gender = st.selectbox("Gênero", ["Male", "Female"])
    family_history = st.selectbox("Histórico familiar de obesidade", ["yes", "no"])
    favc = st.selectbox("Consumo frequente de alimentos calóricos", ["yes", "no"])
    fcvc = st.slider("Consumo de vegetais (1 = raramente, 3 = sempre)", 1, 3, 2)
    ncp = st.slider("Número de refeições principais por dia", 1, 4, 3)
    caec = st.selectbox("Consumo entre refeições", ["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Fuma?", ["yes", "no"])
    ch2o = st.slider("Consumo diário de água (1 = <1L, 3 = >2L)", 1, 3, 2)
    scc = st.selectbox("Monitora consumo de calorias?", ["yes", "no"])
    faf = st.slider("Frequência de atividade física", 0, 3, 1)
    tue = st.slider("Tempo em dispositivos eletrônicos", 0, 2, 1)
    calc = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox(
        "Meio de transporte",
        ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"]
    )

    # ------------------------------
    # INPUTS ESPECÍFICOS DO MODELO CLÍNICO
    # ------------------------------
    if tipo_analise == "Diagnóstico Clínico (com peso e altura)":
        weight = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Altura (m)", 1.30, 2.10, 1.70)

    # ------------------------------
    # BOTÃO DE PREDIÇÃO
    # ------------------------------
    if st.button("🔍 Realizar predição"):

        if tipo_analise == "Diagnóstico Clínico (com peso e altura)":
            bmi = weight / (height ** 2)

            input_data = pd.DataFrame([{
                "Age": age,
                "Height": height,
                "Weight": weight,
                "FCVC": fcvc,
                "NCP": ncp,
                "CH2O": ch2o,
                "FAF": faf,
                "TUE": tue,
                "BMI": bmi,
                "Gender": gender,
                "family_history": family_history,
                "FAVC": favc,
                "CAEC": caec,
                "SMOKE": smoke,
                "SCC": scc,
                "CALC": calc,
                "MTRANS": mtrans
            }])

            pred = modelo_clinico.predict(input_data)[0]

        else:
            input_data = pd.DataFrame([{
                "Age": age,
                "FCVC": fcvc,
                "NCP": ncp,
                "CH2O": ch2o,
                "FAF": faf,
                "TUE": tue,
                "Gender": gender,
                "family_history": family_history,
                "FAVC": favc,
                "CAEC": caec,
                "SMOKE": smoke,
                "SCC": scc,
                "CALC": calc,
                "MTRANS": mtrans
            }])

            pred = modelo_preventivo.predict(input_data)[0]

        classe = encoder_target.inverse_transform([pred])[0]

        st.success(f"🧠 **Nível de obesidade previsto:** {classe}")
        st.info("⚠️ Sistema de apoio à decisão clínica. Não substitui avaliação médica.")

# =========================================================
# 📊 DASHBOARD ANALÍTICO
# =========================================================
if aba == "📊 Dashboard Analítico":

    st.title("📊 Dashboard Analítico - Obesidade")
    st.write(
        """
        Painel analítico com base em dados históricos para apoio
        à tomada de decisão clínica e ações preventivas.
        """
    )

    # ------------------------------
    # CARREGAR DADOS
    # ------------------------------
    df_dash = pd.read_csv("Obesity.csv")

    cols_round = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
    for col in cols_round:
        df_dash[col] = df_dash[col].round().astype(int)

    df_dash["BMI"] = df_dash["Weight"] / (df_dash["Height"] ** 2)

    # ------------------------------
    # DISTRIBUIÇÃO DA OBESIDADE
    # ------------------------------
    st.subheader("Distribuição dos níveis de obesidade")
    fig1, ax1 = plt.subplots()
    df_dash["Obesity"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Quantidade de pacientes")
    ax1.set_xlabel("Nível de obesidade")
    st.pyplot(fig1)

    # ------------------------------
    # IMC x OBESIDADE
    # ------------------------------
    st.subheader("IMC por nível de obesidade")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    df_dash.boxplot(column="BMI", by="Obesity", ax=ax2, rot=90)
    ax2.set_title("")
    ax2.set_ylabel("IMC")
    st.pyplot(fig2)

    # ------------------------------
    # ATIVIDADE FÍSICA
    # ------------------------------
    st.subheader("Atividade física x IMC")
    fig3, ax3 = plt.subplots()
    df_dash.groupby("FAF")["BMI"].mean().plot(kind="bar", ax=ax3)
    ax3.set_xlabel("Frequência de atividade física")
    ax3.set_ylabel("IMC médio")
    st.pyplot(fig3)

    # ------------------------------
    # CONSUMO DE ÁGUA
    # ------------------------------
    st.subheader("Consumo de água x IMC")
    fig4, ax4 = plt.subplots()
    df_dash.groupby("CH2O")["BMI"].mean().plot(kind="bar", ax=ax4)
    ax4.set_xlabel("Consumo diário de água")
    ax4.set_ylabel("IMC médio")
    st.pyplot(fig4)

    # ------------------------------
    # HISTÓRICO FAMILIAR
    # ------------------------------
    st.subheader("Histórico familiar x IMC")
    fig5, ax5 = plt.subplots()
    df_dash.groupby("family_history")["BMI"].mean().plot(kind="bar", ax=ax5)
    ax5.set_xlabel("Histórico familiar")
    ax5.set_ylabel("IMC médio")
    st.pyplot(fig5)
