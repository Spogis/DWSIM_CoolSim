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
        input_columns = ['Evaporator Temperature', 'Condenser Temperature', 'Adiabatic Efficiency', 'Capacity']
        output_columns = ['Compressor Energy', 'Electric Current', 'Discharge Temperature', 'Refrigerant Mass Flow']
    if MLP_Type == "Inverse MLP":
        input_columns = ['Compressor Energy', 'Electric Current', 'Discharge Temperature', 'Refrigerant Mass Flow']
        output_columns = ['Evaporator Temperature', 'Condenser Temperature', 'Adiabatic Efficiency', 'Capacity']
    

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

