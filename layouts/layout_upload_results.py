import dash
from dash import html, dcc, dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_upload_results():
    layout = html.Div([
        html.Br(),
        html.Div([
            dcc.Upload(
                id='upload-results',
                children=html.Div(['Drag or ', html.A('Select a File with the Simulation Results!')]),
                style={
                    'width': '400px', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '3px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                    'textAlign': 'center'
                },
                multiple=False  # Permite a seleção de um único arquivo por vez
            ),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '10px'}),

        html.Br(),
        html.Div([
            html.Div(id='output-data-upload', style={'width': '500px', 'textAlign': 'center', 'marginRight': '10px', 'fontWeight': 'bold'}),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '10px'}),
    ])

    return layout
