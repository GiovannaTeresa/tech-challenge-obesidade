import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ==============================
# CONFIGURAÇÃO DA PÁGINA
# ==============================
st.set_page_config(
    page_title="Predição de Obesidade",
    layout="centered"
)

# ==============================
# MENU LATERAL
# ==============================
aba = st.sidebar.radio(
    "Navegação",
    ["🩺 Sistema Preditivo", "📊 Dashboard Analítico"]
)

# ==============================
# TREINAMENTO DOS MODELOS
# ==============================
@st.cache_resource
def treinar_modelos():
    df = pd.read_csv("Obesity.csv")

    # Feature engineering
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    y = df["Obesity"]
    X = df.drop("Obesity", axis=1)

    encoder_y = LabelEncoder()
    y_enc = encoder_y.fit_transform(y)

    # ==========================
    # MODELO CLÍNICO
    # ==========================
    num_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    modelo_clinico = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    modelo_clinico.fit(X, y_enc)

    # ==========================
    # MODELO PREVENTIVO
    # ==========================
    X_prev = X.drop(["Weight", "Height", "BMI"], axis=1)

    num_prev = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    cat_prev = [c for c in X_prev.columns if c not in num_prev]

    preprocessor_prev = ColumnTransformer([
        ("num", StandardScaler(), num_prev),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_prev)
    ])

    modelo_preventivo = Pipeline([
        ("prep", preprocessor_prev),
        ("model", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    modelo_preventivo.fit(X_prev, y_enc)

    return modelo_clinico, modelo_preventivo, encoder_y


modelo_clinico, modelo_preventivo, encoder_target = treinar_modelos()

# ==============================
# SISTEMA PREDITIVO
# ==============================
if aba == "🩺 Sistema Preditivo":

    st.title("🩺 Sistema de Predição de Obesidade")
    st.write(
        """
        Sistema de apoio à decisão clínica para identificação do **nível de obesidade**.
        Pode ser utilizado tanto em contexto **clínico** quanto **preventivo**.
        """
    )

    tipo_analise = st.selectbox(
        "Tipo de análise:",
        (
            "Diagnóstico Clínico (com peso e altura)",
            "Análise Preventiva (sem peso e altura)"
        )
    )

    st.divider()

    # INPUTS COMUNS
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

    if tipo_analise == "Diagnóstico Clínico (com peso e altura)":
        weight = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Altura (m)", 1.30, 2.10, 1.70)

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
        st.info("⚠️ Este sistema é um apoio à decisão e não substitui avaliação médica.")

# ==============================
# DASHBOARD ANALÍTICO
# ==============================
if aba == "📊 Dashboard Analítico":

    st.title("📊 Dashboard Analítico – Obesidade")
    st.write(
        """
        Visão analítica com foco em **insights clínicos e preventivos**,
        apoiando a tomada de decisão da equipe médica.
        """
    )

    # ==============================
    # CARREGAR DADOS
    # ==============================
    df = pd.read_csv("Obesity.csv")
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # ==============================
    # 1. DISTRIBUIÇÃO DOS NÍVEIS
    # ==============================
    st.subheader("Distribuição dos níveis de obesidade")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    df["Obesity"].value_counts().plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Nível de obesidade")
    ax1.set_ylabel("Quantidade de indivíduos")
    ax1.set_title("Distribuição dos níveis de obesidade")
    st.pyplot(fig1)

    st.markdown(
        "**Insight:** Maior concentração em sobrepeso e obesidade tipo I, "
        "indicando necessidade de ações preventivas precoces."
    )

    # ==============================
    # 2. IMC x OBESIDADE (BOXPLOT)
    # ==============================
    st.subheader("Relação entre IMC e nível de obesidade")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    df.boxplot(column="BMI", by="Obesity", ax=ax2, rot=45)
    ax2.set_xlabel("Nível de obesidade")
    ax2.set_ylabel("IMC")
    ax2.set_title("Distribuição do IMC por nível de obesidade")
    plt.suptitle("")
    st.pyplot(fig2)

    st.markdown(
        "**Insight:** O IMC apresenta clara separação entre os níveis de obesidade, "
        "reforçando sua relevância clínica."
    )

    # ==============================
    # 3. ATIVIDADE FÍSICA x IMC (FAIXAS)
    # ==============================
    st.subheader("Atividade física x IMC médio")
    
    df["Atividade_fisica"] = df["FAF"].map({
        0: "Sedentário",
        1: "Baixa",
        2: "Moderada",
        3: "Alta"
    })
    
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    df.groupby("Atividade_fisica")["BMI"].mean().loc[
        ["Sedentário", "Baixa", "Moderada", "Alta"]
    ].plot(kind="bar", ax=ax3)
    
    ax3.set_xlabel("Nível de atividade física")
    ax3.set_ylabel("IMC médio")
    ax3.set_title("IMC médio por nível de atividade física")
    st.pyplot(fig3)

    

    # ==============================
    # 4. CONSUMO DE ÁGUA (AGRUPADO)
    # ==============================
    st.subheader("Consumo de água x IMC médio")

    df["Consumo_agua"] = df["CH2O"].map({
        1: "< 1 litro/dia",
        2: "1–2 litros/dia",
        3: "> 2 litros/dia"
    })

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    df.groupby("Consumo_agua")["BMI"].mean().plot(kind="bar", ax=ax4)
    ax4.set_xlabel("Consumo diário de água")
    ax4.set_ylabel("IMC médio")
    ax4.set_title("IMC médio por consumo de água")
    st.pyplot(fig4)

    st.markdown(
        "**Insight:** Menor consumo de água está associado a maior IMC médio, "
        "sugerindo impacto de hábitos simples na saúde metabólica."
    )

    # ==============================
    # 5. HISTÓRICO FAMILIAR
    # ==============================
    st.subheader("Histórico familiar de obesidade x IMC médio")

    fig5, ax5 = plt.subplots(figsize=(6, 4))
    df.groupby("family_history")["BMI"].mean().plot(kind="bar", ax=ax5)
    ax5.set_xlabel("Histórico familiar")
    ax5.set_ylabel("IMC médio")
    ax5.set_title("IMC médio por histórico familiar")
    st.pyplot(fig5)

    st.markdown(
        "**Insight:** Indivíduos com histórico familiar de obesidade "
        "apresentam maior IMC médio, indicando influência genética associada "
        "a fatores comportamentais."
    )

