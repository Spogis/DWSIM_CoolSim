import dash
from dash import html, dcc, dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc

def initial_columns():
    df_drop_values = pd.read_excel('datasets/ODEs_Dataset.xlsx')
    if 'index' in df_drop_values.columns:
        df_drop_values = df_drop_values.drop(columns=['index'])

    # Geramos as opções do dropdown com todos os nomes das colunas.
    drop_options = [{'label': col, 'value': col} for col in df_drop_values.columns]

    return drop_options


def layout_mlp_setup(input_columns, output_columns, drop_options, MLP_Type):
    if MLP_Type == "Direct MLP":
        input_columns = ['POX/C', 'C/A', 'POX/M']
        output_columns = ['X', 'PDI', 'Mn']
    if MLP_Type == "Inverse MLP":
        input_columns = ['X', 'PDI', 'Mn']
        output_columns = ['POX/C', 'C/A', 'POX/M']

    layout = html.Div([
        html.Br(),
        html.Label('MLP Setup:'),
        dcc.Dropdown(
            id='MLP-setup-selector',
            multi=False,
            options=["Direct MLP", "Inverse MLP", "Custom MLP"],
            value=MLP_Type,
        ),
        html.Br(),
        html.Label('MLP Input Layer:'),
        dcc.Dropdown(
            id='column-input-selector',
            multi=True,
            options=drop_options,
            value=input_columns,
        ),
        html.Br(),
        dash_table.DataTable(
            id='input-table',
            page_size=3,
        ),
        html.Br(),
        html.Label('MLP Output Layer:'),
        dcc.Dropdown(
            id='column-output-selector',
            multi=True,
            options=drop_options,
            value=output_columns,
        ),
        html.Br(),
        dash_table.DataTable(
            id='output-table',
            page_size=3,
        ),
    ], style={'width': '80%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto',
              'padding': '20px'})

    return layout

