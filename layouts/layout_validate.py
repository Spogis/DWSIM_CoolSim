import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, Patch
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_validate():
    status = 'assets/status2.txt'
    progress_value = 0
    with open('assets/status2.txt', 'w') as file:
        file.write(str(progress_value))

    layout = html.Div([
        html.Br(),
        html.Div([
            html.Div("Number of Test Cases:",
                     style={'width': '250px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

            dcc.Input(
                id="Validation_Cases",
                type='number',
                value=20,
                disabled=False,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Button('Run MLP Test (DWSIM + MLP)!', id='validation-btn', n_clicks=0,
                        style={'backgroundColor': 'orange', 'color': 'white', 'fontWeight': 'bold',
                               'fontSize': '20px'}),

        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '20px'}),

        html.Div([
            html.Br(),
            dcc.Interval(id="progress-interval2", n_intervals=0, interval=500),
            dbc.Progress(id="progress2", value=0, style={'width': '300px', 'height': '20px', 'margin': 'auto'}),
        ], style={'textAlign': 'center'}),

        html.Br(),
        html.Div([
            html.Div("Compressor Energy MAPE:",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="Compressor_Energy_rscore",
                type='text',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("Electric Current MAPE:",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="Electric_Current_rscore",
                type='text',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("Discharge Temperature MAPE:",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="Discharge_Temperature_rscore",
                type='text',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("Refrigerant Mass Flow MAPE:",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="Refrigerant_Mass_Flow_rscore",
                type='text',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '20px'}),

        html.Br(),
        html.Div([
            html.Div("Evaporator Temperature MAPE:",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="Evaporator_Temperature_rscore",
                type='text',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("Condenser Temperature MAPE:",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="Condenser_Temperature_rscore",
                type='text',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("Adiabatic Efficiency MAPE:",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
            dcc.Input(
                id="Adiabatic_Efficiency_rscore",
                type='text',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '20px'}),

        html.Br(),
        html.Br(),
        dbc.Spinner(html.Div(id="loading-output5"), spinner_style={"marginTop": "40px"}),

    ], style={'textAlign': 'center'})

    return layout
