import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, Patch
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_simulate(M, MWm, Hours):
    layout = html.Div([
        html.Br(),

        html.Div([
            html.Div("Reaction Time (h):",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="reaction_time_value",
                type='number',
                value=Hours,
                disabled=False,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("Styrene Monomer Concentration [mol.L-1]):",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="styrene_monomer_value",
                type='number',
                value=M,
                disabled=False,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("Monomer Molar Mass [g⋅mol-1]:",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="monomer_molar_mass_value",
                type='number',
                value=MWm,
                disabled=False,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '20px'}),

        html.Div([
            html.Br(),
            html.Button('Run Simulations (ODE Solver)!', id='simulation-btn', n_clicks=0,
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
