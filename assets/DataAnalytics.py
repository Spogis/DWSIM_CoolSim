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
            "auto": {"calculate": True},
        },
        explorative=True,
        interactions={"continuous": True},
        title="Profiling Report"
    )

    # Gere o relatório em HTML
    profile_report.to_file(html_file)
