import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, Patch
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objs as go
import plotly.express as px

def update_graph_parcoords():
    df_ODES_Dataset = pd.read_excel('datasets/ODEs_Dataset.xlsx')
    if 'index' in df_ODES_Dataset.columns:
        df_ODES_Dataset = df_ODES_Dataset.drop(columns=['index'])

    fig = px.parallel_coordinates(df_ODES_Dataset, color=df_ODES_Dataset.columns[-1])
    return fig


# Função para atualizar o gráfico de coordenadas paralelas
def update_graph_parcoords_min_max(df, filters):
    for col, (min_val, max_val) in filters.items():
        df = df[(df[col] >= min_val) & (df[col] <= max_val)]
    fig = px.parallel_coordinates(df, color=df.columns[-1])
    return fig


def parallel_chart():
    df = pd.read_excel('datasets/ODEs_Dataset.xlsx')
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    fig = update_graph_parcoords()

    # Inicializa os filtros com valores máximos e mínimos
    filters = {col: [df[col].min(), df[col].max()] for col in df.columns}
    fig = update_graph_parcoords_min_max(df, filters)

    layout = html.Div([
        dcc.Graph(id='graph-parcoords', figure=fig),

        html.Br(),
        dash_table.DataTable(
            id='table_data_analysis_min_max',
            columns=[{'name': i, 'id': i, 'type': 'numeric'} for i in df.columns],
            data=[
                {col: df[col].max() for col in df.columns},  # Max values
                {col: df[col].min() for col in df.columns},  # Min values
            ],
            editable=True,
            style_table={'overflowX': 'scroll'}
        ),

        html.Br(),
        html.Div([
            html.Button('Download Descriptive Statistics', id='btn-stat-download', n_clicks=0,
                        style={'backgroundColor': 'orange', 'color': 'white', 'fontWeight': 'bold',
                               'fontSize': '20px',
                               'marginRight': '10px'}),
            dcc.Download(id="download-excel")
        ], style={'display': 'flex', 'flex-direction': 'column', 'alignItems': 'center', 'justifyContent': 'center',
                  'marginBottom': '10px'}),

        html.Br(),
        dash_table.DataTable(id='table_data_analysis',
                             columns=[{'id': 'index', 'name': 'index'}] + [{'id': i, 'name': i} for i in
                                                                           df.columns],
                             style_table={'overflowX': 'scroll'}),
        dcc.Store(id='activefilters', data={})
    ])

    return layout
