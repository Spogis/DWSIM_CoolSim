import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, Patch
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_solve_once():
    layout = html.Div([
        html.Br(),
        html.Div([
            html.H2("Process Conditions:",
                    style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'fontWeight': 'bold'}),

            html.Div([
                html.Div("Delta T Approach [°C]:",
                         style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
                dcc.Input(
                    id="approach_temperature_value",
                    type='number',
                    value=10,
                    min=5,
                    max=20,
                    disabled=False,
                    style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
                ),

                html.Div("Desired Temperature [°C]:",
                         style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
                dcc.Input(
                    id="desired_temperature_value",
                    type='number',
                    value=23,
                    min=18,
                    disabled=False,
                    style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
                ),

                html.Div("External Ambient Temperature [°C]:",
                         style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
                dcc.Input(
                    id="external_temperature_value",
                    type='number',
                    value=35,
                    max=48,
                    disabled=False,
                    style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
                ),

                html.Div("Adiabatic Efficiency [%]:",
                         style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),
                dcc.Input(
                    id="adiabatic_efficiency_value",
                    type='number',
                    value=75,
                    min=50,
                    max=100,
                    disabled=False,
                    style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
                ),
            ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '20px'}),
        ], style={'padding': '20px', 'margin': 'auto', 'width': '1200px', 'box-shadow': '0px 0px 10px #ccc',
                  'border-radius': '15px'}),

        html.Br(),
        html.Div([
            html.Div([
                html.Div("Evaporator Temperature [°C]:", id="tooltip-x",
                         style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

                dcc.Input(
                    id="evaporator_temperature_value",
                    type='text',
                    value="",
                    disabled=True,
                    style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
                ),

                html.Div("Condenser Temperature [°C]:", id="tooltip-pdi",
                         style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

                dcc.Input(
                    id="condenser_temperature_value",
                    type='text',
                    value="",
                    disabled=True,
                    style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
                ),
            ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '20px'}),
        ], style={'padding': '20px', 'margin': 'auto', 'width': '1200px', 'box-shadow': '0px 0px 10px #ccc',
                  'border-radius': '15px'}),

        html.Br(),
        html.Div([
            html.H2("DWSIM Simulation Values:",
                    style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'fontWeight': 'bold'}),

            html.Div([
                html.Div("Compressor Energy [W]:", id="tooltip-x",
                         style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

                dcc.Input(
                    id="compressor_energy_value",
                    type='text',
                    value="",
                    disabled=True,
                    style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
                ),

                html.Div("Electric Current [A]:", id="tooltip-pdi",
                         style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

                dcc.Input(
                    id="electric_current_value",
                    type='text',
                    value="",
                    disabled=True,
                    style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
                ),

                html.Div("Discharge Temperature [°C]:", id="tooltip-mn",
                         style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

                dcc.Input(
                    id="discharge_temperature_value",
                    type='text',
                    value="",
                    disabled=True,
                    style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
                ),

                html.Div("Refrigerant Mass Flow [kg/min]:", id="tooltip-mn",
                         style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

                dcc.Input(
                    id="refrigerant_mass_flow_value",
                    type='text',
                    value="",
                    disabled=True,
                    style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
                ),
                ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '20px'}),
        ], style={'padding': '20px', 'margin': 'auto', 'width': '1200px', 'box-shadow': '0px 0px 10px #ccc',
                  'border-radius': '15px'}),

        html.Div([
            html.Br(),
            html.Button('Run Simulation!', id='dwsim-once-btn', n_clicks=0,
                        style={'backgroundColor': 'orange', 'color': 'white', 'fontWeight': 'bold',
                               'fontSize': '20px'}),
        ], style={'textAlign': 'center'}),

        html.Br(),
        dbc.Spinner(html.Div(id="loading-output4"), spinner_style={"marginTop": "40px"}),

        html.Div([
            html.Br(),
            html.Br(),
            dcc.Graph(id='output-dwsim-fig', style={'display': 'none'})
        ], style={'padding': '20px', 'margin': 'auto', 'width': '1200px',
                  'display': 'flex', 'justifyContent': 'center',
                  'alignItems': 'center'}),



    ], style={'textAlign': 'center'})

    return layout
