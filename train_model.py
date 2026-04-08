import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# =============================
# 1. Criar pasta de saída
# =============================
os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =============================
# 2. Carregar dataset
# =============================
df = pd.read_csv("data/dataset.csv", sep=";")

print("Dataset carregado")
print(df.head())
print("\nColunas:")
print(df.columns.tolist())

# =============================
# 3. Limpar nomes de colunas
# =============================
df.columns = df.columns.str.strip()

# corrigir possível tab escondido
df.columns = df.columns.str.replace("\t", "", regex=False)

# =============================
# 4. Variável alvo
# =============================
target = "Target"

df[target] = df[target].astype(str).str.strip()

df[target] = df[target].map({
    "Dropout": 1,
    "Graduate": 0,
    "Enrolled": 0
})

print("\nValores únicos em Target após conversão:")
print(df[target].unique())

df = df.dropna(subset=[target])
df[target] = df[target].astype(int)

# =============================
# 5. Separar X e y
# =============================
X = df.drop(columns=[target])
y = df[target]

# =============================
# 6. One-hot encoding
# =============================
X = pd.get_dummies(X, drop_first=False)

# salvar colunas usadas no treino
joblib.dump(X.columns.tolist(), "model/colunas.pkl")

# =============================
# 7. Split treino/teste
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============================
# 8. Treinar XGBoost
# =============================
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# =============================
# 9. Predições
# =============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC: {auc:.4f}")

# =============================
# 10. Salvar modelo
# =============================
joblib.dump(model, "model/modelo.pkl")
print("\nModelo salvo com sucesso em model/modelo.pkl")

# =============================
# 11. Matriz de confusão
# =============================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusão - XGBoost")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("outputs/matriz_confusao.png")
plt.close()

# =============================
# 12. Curva ROC
# =============================
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Curva ROC - XGBoost")
plt.xlabel("Taxa de Falso Positivo")
plt.ylabel("Taxa de Verdadeiro Positivo")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/curva_roc.png")
plt.close()

# =============================
# 13. Importância das variáveis
# =============================
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Top 15 Variáveis Mais Importantes - XGBoost")
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.tight_layout()
plt.savefig("outputs/importancia_variaveis.png")
plt.close()

print("Gráficos salvos em /outputs")