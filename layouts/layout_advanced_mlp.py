import dash
from dash import html, dcc, dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_advanced_mlp():
    layout = html.Div([
        html.Div([
            html.Button(
                html.Img(src='/assets/optuna-logo.png', style={'height': '100%', 'width': '100%'}),
                id='run-OPTMLP-button',
                disabled=False,
                style={
                    'width': '182x',  # Ajuste a largura conforme necessário
                    'height': '38px',  # Ajuste a altura conforme necessário
                    'padding': '0',  # Remove o padding
                    'border': 'none',  # Remove a borda
                    'backgroundColor': 'transparent',  # Fundo transparente para não interferir na imagem
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center'
                }
            ),
        ], style={'display': 'flex', 'flex-direction': 'column', 'alignItems': 'center', 'justifyContent': 'center',
                  'marginBottom': '10px'}),
        html.Br(),
        dbc.Spinner(html.Div(id="loading-output2"), spinner_style={"width": "3rem", "height": "3rem"}),
        html.H2("r² score:"),
        dcc.Textarea(
            id='r2-opt-mlp-textarea',
            style={'width': '100%', 'height': '100px', 'resize': 'none', 'fontWeight': 'bold'},
            readOnly=True
        ),
        html.Br(),
        html.H2("Best Hyperparameters:"),
        dcc.Textarea(
            id='best-hps-textarea',
            style={'width': '100%', 'height': '200px', 'resize': 'none', 'fontWeight': 'bold'},
            readOnly=True
        ),
        html.H2("Best Model Architecture:"),
        dcc.Textarea(
            id='model-summary-textarea',
            style={'width': '100%', 'height': '300px', 'resize': 'none', 'fontWeight': 'bold'},
            readOnly=True
        ),
        html.Br(),
        html.Div(id='button-output-advanced'),
    ], style={'width': '100%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto',
              'padding': '20px'})

    return layout
