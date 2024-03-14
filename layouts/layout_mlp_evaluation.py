import dash
from dash import html, dcc, dash_table
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_mlp_evaluation(input_columns):
    modified_string = ', '.join(map(str, input_columns))
    layout = html.Div([
        html.Div([
            html.H5("Input Values (coma, space or semicolon):"),
            dcc.Textarea(id='input-variables-textarea',
                         value=modified_string,
                         readOnly=True,
                         style={'width': '50%', 'height': '30px', 'resize': 'none', 'fontWeight': 'bold'}),
            html.Br(),

            dcc.Textarea(id='input-text',
                         value='',
                         style={'width': '50%', 'height': '30px', 'resize': 'none', 'fontWeight': 'bold'}),
            html.Br(),

            html.Button('Predict Values!',
                        id='predict-button',
                        disabled=False,
                        style={'width': '400px', 'backgroundColor': 'green', 'color': 'white',
                               'fontWeight': 'bold', 'fontSize': '20px'}),
            html.Br(),
            dcc.Textarea(id='output-text',
                         readOnly=True,
                         style={'width': '50%', 'height': '100px', 'resize': 'none', 'fontWeight': 'bold'}),

        ], style={'display': 'flex', 'flex-direction': 'column', 'alignItems': 'center', 'justifyContent': 'center',
                  'marginBottom': '5px'}),

    ], style={'width': '100%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '20px'})

    return layout
