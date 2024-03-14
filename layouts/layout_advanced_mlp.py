import dash
from dash import html, dcc, dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_advanced_mlp():
    layout = html.Div([
        html.Div([
            html.Button('OPTIMIZE KERAS MLP!',
                        id='run-OPTMLP-button',
                        disabled=False,
                        style={'width': '400px', 'backgroundColor': 'green', 'color': 'white',
                               'fontWeight': 'bold', 'fontSize': '20px', 'marginRight': '50px'}),
        ], style={'display': 'flex', 'flex-direction': 'column', 'alignItems': 'center', 'justifyContent': 'center',
                  'marginBottom': '10px'}),
        html.Br(),
        dbc.Spinner(html.Div(id="loading-output2"), spinner_style={"width": "3rem", "height": "3rem"}),
        html.H2("r² score:"),
        dcc.Textarea(
            id='r2-opt-mlp-textarea',
            style={'width': '100%', 'height': 200, 'resize': 'none', 'fontWeight': 'bold'},
            readOnly=True
        ),
        html.Br(),
        html.H2("Best Hyperparameters:"),
        dcc.Textarea(
            id='best-hps-textarea',
            style={'width': '100%', 'height': 200, 'resize': 'none', 'fontWeight': 'bold'},
            readOnly=True
        ),
        html.H2("Best Model Architecture:"),
        dcc.Textarea(
            id='model-summary-textarea',
            style={'width': '100%', 'height': 200, 'resize': 'none', 'fontWeight': 'bold'},
            readOnly=True
        ),
        html.Br(),
        html.Div(id='button-output-advanced'),
    ], style={'width': '100%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto',
              'padding': '20px'})

    return layout
