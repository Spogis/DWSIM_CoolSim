import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, Patch
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objs as go
import plotly.express as px

def update_graph_parcoords():
    df_ODES_Dataset = pd.read_excel('datasets/ARGET_ATRP_ODEs_Dataset.xlsx')
    if 'index' in df_ODES_Dataset.columns:
        df_ODES_Dataset = df_ODES_Dataset.drop(columns=['index'])

    fig = px.parallel_coordinates(df_ODES_Dataset, color=df_ODES_Dataset.columns[-1])
    return fig


def parallel_chart():
    df = pd.read_excel('datasets/ARGET_ATRP_ODEs_Dataset.xlsx')
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    fig = update_graph_parcoords()
    layout = html.Div([
        dcc.Graph(id='graph-parcoords', figure=fig),
        html.Br(),
        html.Div([
            html.Button('Download Descriptive Statistics', id='btn-stat-download', n_clicks=0,
                        style={'backgroundColor': 'orange', 'color': 'white', 'fontWeight': 'bold',
                               'fontSize': '20px',
                               'marginRight': '10px'}),
            dcc.Download(id="download-excel")
        ], style={'display': 'flex', 'flex-direction': 'column', 'alignItems': 'center', 'justifyContent': 'center',
                  'marginBottom': '10px'}),

        dash_table.DataTable(id='table_data_analysis',
                             columns=[{'id': 'index', 'name': 'index'}] + [{'id': i, 'name': i} for i in
                                                                           df.columns],
                             style_table={'overflowX': 'scroll'}),
        dcc.Store(id='activefilters', data={})
    ])

    return layout
