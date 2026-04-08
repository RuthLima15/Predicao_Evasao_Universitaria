import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

st.set_page_config(
    page_title="Predição de Evasão Universitária",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.2rem;
    max-width: 1250px;
}
</style>
""", unsafe_allow_html=True)

st.title("Sistema Inteligente de Predição de Evasão Universitária")
st.write("Aplicação para estimar o risco de evasão no ensino superior.")

@st.cache_data
def carregar_dataset():
    df = pd.read_csv("data/dataset.csv", sep=";")
    df.columns = df.columns.str.strip().str.replace("\t", "", regex=False)
    return df

@st.cache_resource
def carregar_modelo():
    modelo = joblib.load("model/modelo.pkl")
    colunas = joblib.load("model/colunas.pkl")
    return modelo, colunas

@st.cache_data
def preparar_dados():
    df = carregar_dataset().copy()

    df["Target"] = df["Target"].astype(str).str.strip()
    df["Target"] = df["Target"].map({
        "Dropout": 1,
        "Graduate": 0,
        "Enrolled": 0
    })

    df = df.dropna(subset=["Target"])
    df["Target"] = df["Target"].astype(int)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    X = pd.get_dummies(X, drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return df, X, y, X_train, X_test, y_train, y_test

def estilizar_figura(fig, altura=380):
    fig.update_layout(
        height=altura,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=13)
    )
    return fig

df_raw = carregar_dataset()
model, colunas_modelo = carregar_modelo()
df_modelo, X, y, X_train, X_test, y_train, y_test = preparar_dados()

st.sidebar.header("📥 Dados do estudante")

idade = st.sidebar.number_input("Idade no ingresso", min_value=15, max_value=70, value=20)
nota_admissao = st.sidebar.number_input("Nota de admissão", min_value=0.0, max_value=200.0, value=120.0)

devedor = st.sidebar.selectbox(
    "Possui dívida?",
    [0, 1],
    format_func=lambda x: "Não" if x == 0 else "Sim"
)

mensalidade_em_dia = st.sidebar.selectbox(
    "Mensalidade em dia?",
    [0, 1],
    format_func=lambda x: "Não" if x == 0 else "Sim"
)

bolsista = st.sidebar.selectbox(
    "Possui bolsa?",
    [0, 1],
    format_func=lambda x: "Não" if x == 0 else "Sim"
)

aprovadas_1 = st.sidebar.number_input(
    "Disciplinas aprovadas no 1º semestre",
    min_value=0, max_value=30, value=5
)

media_1 = st.sidebar.number_input(
    "Média do 1º semestre",
    min_value=0.0, max_value=20.0, value=10.0
)

aprovadas_2 = st.sidebar.number_input(
    "Disciplinas aprovadas no 2º semestre",
    min_value=0, max_value=30, value=5
)

media_2 = st.sidebar.number_input(
    "Média do 2º semestre",
    min_value=0.0, max_value=20.0, value=10.0
)

genero = st.sidebar.selectbox(
    "Gênero",
    [0, 1],
    format_func=lambda x: "Feminino" if x == 0 else "Masculino"
)

prever = st.sidebar.button("🔍 Prever risco", use_container_width=True)

entrada = {
    "Age at enrollment": idade,
    "Admission grade": nota_admissao,
    "Debtor": devedor,
    "Tuition fees up to date": mensalidade_em_dia,
    "Scholarship holder": bolsista,
    "Curricular units 1st sem (approved)": aprovadas_1,
    "Curricular units 1st sem (grade)": media_1,
    "Curricular units 2nd sem (approved)": aprovadas_2,
    "Curricular units 2nd sem (grade)": media_2,
    "Gender": genero
}

input_df = pd.DataFrame([entrada])
input_df = pd.get_dummies(input_df)

for col in colunas_modelo:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[colunas_modelo]

st.markdown("##  Resultado da predição")

if prever:
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidade de evasão", f"{prob:.2%}")
    c2.metric("Classe prevista", "Em risco" if pred == 1 else "Sem risco")
    c3.metric("Nível de risco", "Alto" if prob >= 0.65 else "Médio" if prob >= 0.35 else "Baixo")

    st.progress(float(prob))

    if prob < 0.35:
        st.success("Baixo risco de evasão.")
        st.info("Sugestão: manter acompanhamento acadêmico regular.")
    elif prob < 0.65:
        st.warning("Médio risco de evasão.")
        st.info("Sugestão: realizar monitoramento mais próximo e orientação acadêmica.")
    else:
        st.error("Alto risco de evasão.")
        st.info("Sugestão: encaminhar para apoio pedagógico, psicopedagógico e avaliação socioeconômica.")

    st.markdown("### Dados informados")
    tabela_entrada = pd.DataFrame([{
        "Idade no ingresso": idade,
        "Nota de admissão": nota_admissao,
        "Possui dívida?": "Sim" if devedor == 1 else "Não",
        "Mensalidade em dia?": "Sim" if mensalidade_em_dia == 1 else "Não",
        "Possui bolsa?": "Sim" if bolsista == 1 else "Não",
        "Disciplinas aprovadas no 1º semestre": aprovadas_1,
        "Média do 1º semestre": media_1,
        "Disciplinas aprovadas no 2º semestre": aprovadas_2,
        "Média do 2º semestre": media_2,
        "Gênero": "Feminino" if genero == 0 else "Masculino"
    }])
    st.dataframe(tabela_entrada, use_container_width=True)
else:
    st.info("Preencha os dados na barra lateral e clique em 'Prever risco'.")

aba1, aba2, aba3, aba4 = st.tabs([
    " Base de dados",
    " Gráficos",
    " Métricas",
    " Importância das variáveis"
])

with aba1:
    st.subheader("Amostra da base de dados")
    st.dataframe(df_raw.head(20), use_container_width=True)

    info_df = pd.DataFrame({
        "Indicador": ["Número de linhas", "Número de colunas"],
        "Valor": [df_raw.shape[0], df_raw.shape[1]]
    })
    st.subheader("Informações gerais")
    st.dataframe(info_df, use_container_width=True)

with aba2:
    st.subheader("Distribuição da variável alvo")

    target_plot = df_raw.copy()
    target_plot["Target"] = target_plot["Target"].astype(str).str.strip()

    contagem = target_plot["Target"].value_counts().reset_index()
    contagem.columns = ["Classe", "Quantidade"]

    st.dataframe(contagem, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            contagem,
            x="Classe",
            y="Quantidade",
            text="Quantidade",
            title="Distribuição das classes"
        )
        fig1.update_traces(textposition="outside")
        st.plotly_chart(estilizar_figura(fig1), use_container_width=True)

    with col2:
        fig2 = px.histogram(
            df_raw,
            x="Age at enrollment",
            nbins=20,
            title="Distribuição da idade no ingresso"
        )
        st.plotly_chart(estilizar_figura(fig2), use_container_width=True)

    fig3 = px.scatter(
        df_raw,
        x="Admission grade",
        y="Curricular units 1st sem (approved)",
        opacity=0.6,
        title="Nota de admissão x disciplinas aprovadas no 1º semestre"
    )
    st.plotly_chart(estilizar_figura(fig3, altura=420), use_container_width=True)

with aba3:
    st.subheader("Avaliação do modelo")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(4)

    auc = roc_auc_score(y_test, y_prob)

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{auc:.4f}")
    c2.metric("Amostras de teste", len(y_test))
    c3.metric("Número de variáveis", X.shape[1])

    st.markdown("### Tabela de métricas")
    st.dataframe(report_df, use_container_width=True)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Real 0", "Real 1"],
        columns=["Previsto 0", "Previsto 1"]
    )

    st.markdown("### Matriz de confusão")
    st.dataframe(cm_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig4 = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Matriz de confusão",
            labels=dict(x="Previsto", y="Real", color="Quantidade"),
            x=["Previsto 0", "Previsto 1"],
            y=["Real 0", "Real 1"]
        )
        st.plotly_chart(estilizar_figura(fig4), use_container_width=True)

    with col2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc:.4f})"
        ))
        fig5.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Referência",
            line=dict(dash="dash")
        ))
        fig5.update_layout(
            title="Curva ROC",
            xaxis_title="Taxa de falsos positivos",
            yaxis_title="Taxa de verdadeiros positivos"
        )
        st.plotly_chart(estilizar_figura(fig5), use_container_width=True)

with aba4:
    st.subheader("Variáveis mais importantes do modelo")

    importancias = pd.Series(model.feature_importances_, index=X.columns)
    importancias = importancias.sort_values(ascending=False).head(15)

    tabela_importancia = pd.DataFrame({
        "Variável": importancias.index,
        "Importância": importancias.values.round(4)
    })

    st.dataframe(tabela_importancia, use_container_width=True)

    fig6 = px.bar(
        tabela_importancia.sort_values("Importância"),
        x="Importância",
        y="Variável",
        orientation="h",
        title="Top 15 variáveis mais importantes"
    )
    st.plotly_chart(estilizar_figura(fig6, altura=520), use_container_width=True)

st.markdown("---")
st.caption("Protótipo acadêmico para apoio à identificação precoce de risco de evasão universitária.")