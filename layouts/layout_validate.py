import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, Patch
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_validate(M, MWm, Hours):
    layout = html.Div([
        html.Br(),
        html.Div([
            html.Br(),
            html.Button('Run Validation (ODE Solver + MLP)!', id='validation-btn', n_clicks=0,
                        style={'backgroundColor': 'orange', 'color': 'white', 'fontWeight': 'bold',
                               'fontSize': '20px'}),
        ], style={'textAlign': 'center'}),

        html.Div([
            html.Br(),
            dcc.Interval(id="progress-interval2", n_intervals=0, interval=500),
            dbc.Progress(id="progress2", value=0, style={'width': '300px', 'height': '20px', 'margin': 'auto'}),
        ], style={'textAlign': 'center'}),
        html.Br(),
        dbc.Spinner(html.Div(id="loading-output5"), spinner_style={"marginTop": "40px"}),

        html.Br(),
        html.Br(),
        html.Div([
            html.Div("X r² score:",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="Xrscore",
                type='number',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("PDI r² score:",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="PDIrscore",
                type='number',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("Mn r² score:",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="Mnrscore",
                type='number',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '20px'}),
    ], style={'textAlign': 'center'})

    return layout
