import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, Patch
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_simulate():
    layout = html.Div([
        html.Br(),

        html.Div([
            html.Br(),
            html.Button('Run Simulations (DOE)!', id='simulation-btn', n_clicks=0,
                        style={'backgroundColor': 'orange', 'color': 'white', 'fontWeight': 'bold',
                               'fontSize': '20px'}),
        ], style={'textAlign': 'center'}),

        html.Div([
            html.Br(),
            dcc.Interval(id="progress-interval", n_intervals=0, interval=500),
            dbc.Progress(id="progress", value=0, style={'width': '300px', 'height': '20px', 'margin': 'auto'}),
        ], style={'textAlign': 'center'}),
        html.Br(),
        dbc.Spinner(html.Div(id="loading-output3"), spinner_style={"marginTop": "40px"}),

    ], style={'textAlign': 'center'})

    return layout
