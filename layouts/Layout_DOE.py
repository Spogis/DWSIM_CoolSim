import dash
from dash import html, dcc, dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc

initial_df_data = pd.read_excel('datasets/ARGET_ATRP_Input.xlsx')

# Inicializa os dados da tabela com as novas colunas e uma linha de exemplo
df = pd.DataFrame({
    'Variable Name': [''],
    'Mean': [''],
    'Standard Deviation': [''],
    'Max': [''],
    'Min': [''],
    'Step (If Variable is Discrete)': [''],
    'Trust Level': [0.95],  # 95% como valor inicial
    'Variable Type': ['Continuous']  # Valor inicial padrão
})

# Definição dos tipos de variáveis para o dropdown
variable_types = ['Continuous', 'Discrete']
trust_level_values = ['0.9', '0.95', '0.99']


def layout_DOE():
    layout = html.Div([
        html.Br(),
        dash_table.DataTable(
            id='table',
            columns=[
                {'id': 'Variable Name', 'name': 'Variable Name', 'editable': True},
                {'id': 'Mean', 'name': 'Mean', 'type': 'numeric', 'editable': True},
                {'id': 'Standard Deviation', 'name': 'Standard Deviation', 'type': 'numeric', 'editable': True},
                {'id': 'Max', 'name': 'Max', 'type': 'numeric', 'editable': True},
                {'id': 'Min', 'name': 'Min', 'type': 'numeric', 'editable': True},
                {'id': 'Step (If Variable is Discrete)', 'name': 'Step (If Variable is Discrete)', 'type': 'numeric',
                 'editable': True},
                {'id': 'Trust Level', 'name': 'Trust Level', 'presentation': 'dropdown', 'editable': True,
                 'format': Format(precision=2, scheme=Scheme.percentage)},
                {'id': 'Variable Type', 'name': 'Variable Type', 'presentation': 'dropdown', 'editable': True}
            ],
            data=initial_df_data.to_dict('records'),
            editable=True,
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Variable Type} = Discrete',  # Se Variable Type for Discrete
                        'column_id': 'Step (If Variable is Discrete)'
                    },
                    'backgroundColor': '#FAFAFA',
                    'border': '1px solid blue'
                },
            ],
            style_cell={
                'minWidth': '150px', 'width': '150px', 'maxWidth': '150px',  # Defina a largura das colunas aqui
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'whiteSpace': 'normal'
            },
            style_header={
                'textAlign': 'center'  # Centraliza o texto no cabeçalho também
            },
            row_deletable=False,
            dropdown={
                'Variable Type': {
                    'options': [
                        {'label': i, 'value': i}
                        for i in variable_types
                    ]
                },
                'Trust Level': {
                    'options': [
                        {'label': i, 'value': i}
                        for i in trust_level_values
                    ]
                },
            },
        ),
        html.Br(),

        html.Div([
            html.Br(),

            html.Button('Create DOE', id='create-doe-btn', n_clicks=0,
                        style={'width': '400px', 'backgroundColor': 'green', 'color': 'white',
                               'fontWeight': 'bold', 'fontSize': '20px', 'margin': 'auto'}),

        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '10px'}),


        html.Div([
            html.Div("Number of Simulations:", style={'width': '200px', 'textAlign': 'center', 'marginRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="numero_de_simulacoes",
                type='number',
                value=1000,
                disabled=False,
                step=1,
                style={'width': '150px', 'textAlign': 'center', 'fontWeight': 'bold'}
            )
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '10px'}),
    ])

    return layout
