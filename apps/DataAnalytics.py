import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import os


def data_analytics(df):
    html_file = 'assets/relatorio_analise.html'
    if os.path.exists(html_file):
        os.remove(html_file)

    # Carregue seus dados do arquivo Excel
    dados = df

    profile_report = ProfileReport(
        dados,
        sort=None,
        html={
            "style": {"full_width": True}
        },
        progress_bar=True,
        correlations={
            "pearson": {"calculate": True},   # Correlação de Pearson para variáveis contínuas
            "spearman": {"calculate": True},  # Correlação de Spearman para variáveis ordinais
            "kendall": {"calculate": True},   # Correlação de Kendall para uma medida não paramétrica
            "phi_k": {"calculate": True},     # Correlação Phik para variáveis categóricas e mistas
            "cramers": {"calculate": True},   # Correlação de Cramér para variáveis categóricas
            "auto": {"calculate": False}      # Desativa o cálculo automático para evitar duplicatas
        },

        explorative=True,
        interactions={"continuous": False},  # Desativa a seção de Interactions
        missing_diagrams={"bar": False, "matrix": False, "heatmap": False, "dendrogram": False},
        # Desativa todas as visualizações de Missing Values
        samples={"head": False, "tail": False},  # Desativa a seção de Sample
        title="Profiling Report"
    )

    # Gere o relatório em HTML
    profile_report.to_file(html_file)
