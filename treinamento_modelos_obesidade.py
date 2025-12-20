# =========================================================
# TREINAMENTO DOS MODELOS DE PREDIÇÃO DE OBESIDADE
# Tech Challenge - Ciência de Dados
# =========================================================

# ======================
# IMPORTAÇÕES
# ======================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ======================
# CARREGAMENTO DOS DADOS
# ======================
df = pd.read_csv("Obesity.csv")

# ======================
# TRATAMENTO DE DADOS
# ======================

# Arredondar colunas com ruído decimal
cols_round = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
for col in cols_round:
    df[col] = df[col].round().astype(int)

# Feature engineering: IMC
df["BMI"] = df["Weight"] / (df["Height"] ** 2)

# ======================
# VARIÁVEL ALVO
# ======================
X = df.drop("Obesity", axis=1)
y = df["Obesity"]

label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

# ======================
# MODELO 1 - CLÍNICO
# ======================

numeric_features_clinico = [
    "Age", "Height", "Weight",
    "FCVC", "NCP", "CH2O",
    "FAF", "TUE", "BMI"
]

categorical_features = [
    "Gender", "family_history", "FAVC", "CAEC",
    "SMOKE", "SCC", "CALC", "MTRANS"
]

preprocessor_clinico = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features_clinico),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

modelo_clinico = Pipeline(steps=[
    ("preprocessor", preprocessor_clinico),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

modelo_clinico.fit(X_train, y_train)
y_pred = modelo_clinico.predict(X_test)

print("Acurácia modelo clínico:",
      accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Salvar modelo clínico
joblib.dump(modelo_clinico, "modelo_clinico.pkl")

# ======================
# MODELO 2 - PREVENTIVO
# ======================

X_prev = df.drop(
    ["Obesity", "Weight", "Height", "BMI"],
    axis=1
)

numeric_features_prev = [
    "Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"
]

preprocessor_prev = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features_prev),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

modelo_preventivo = Pipeline(steps=[
    ("preprocessor", preprocessor_prev),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ))
])

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_prev,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

modelo_preventivo.fit(X_train2, y_train2)
y_pred2 = modelo_preventivo.predict(X_test2)

print("Acurácia modelo preventivo:",
      accuracy_score(y_test2, y_pred2))
print(classification_report(y_test2, y_pred2))

# Salvar modelo preventivo
joblib.dump(modelo_preventivo, "modelo_preventivo.pkl")

# ======================
# SALVAR ENCODER DA TARGET
# ======================
joblib.dump(label_encoder_y, "encoder_target.pkl")

print("Modelos treinados e salvos com sucesso.")
