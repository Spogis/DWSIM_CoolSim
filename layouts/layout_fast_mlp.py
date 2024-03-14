import dash
from dash import html, dcc, dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_fast_mlp():
    layout = html.Div([
        html.Div([
            html.Button('RUN PREDEFINED KERAS MLP!',
                        id='run-MLP-button',
                        disabled=False,
                        style={'width': '400px', 'backgroundColor': 'green', 'color': 'white',
                               'fontWeight': 'bold', 'fontSize': '20px'}),
        ], style={'display': 'flex', 'flex-direction': 'column', 'alignItems': 'center', 'justifyContent': 'center',
                  'marginBottom': '10px'}),
        html.Br(),
        dbc.Spinner(html.Div(id="loading-output1"), spinner_style={"width": "3rem", "height": "3rem"}),

        html.H2("r² score:"),
        dcc.Textarea(
            id='r2-simple-mlp-textarea',
            style={'width': '100%', 'height': '100px', 'resize': 'none', 'fontWeight': 'bold'},
            readOnly=True
        ),
        html.Br(),
        html.Div(id='button-output'),
    ], style={'width': '100%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '20px'})

    return layout
