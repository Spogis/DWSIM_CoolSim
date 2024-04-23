import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, Patch
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_solve_once(M, MWm, Hours):
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

            html.Div("Styrene Monomer Concentration [mol⋅L-1]:",
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

        html.Br(),

        html.Div([
            html.Div("P0X/C:", id="tooltip-pox-c",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

            dbc.Tooltip(
                "P0X/C  = Initiator Concentration (P0X) / Active Catalyst Concentration (C)",
                target="tooltip-pox-c",
                placement="top"
            ),

            dcc.Input(
                id="POX_C_value",
                type='number',
                value=100.0,
                disabled=False,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("C/A:", id="tooltip-c-a",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

            dbc.Tooltip(
                "C/A  = Active Catalyst Concentration (C) / Reducing agent in reduced form Concentration (A)",
                target="tooltip-c-a",
                placement="top"
            ),

            dcc.Input(
                id="C_A_value",
                type='number',
                value=0.1,
                disabled=False,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("POX/M:", id="tooltip-pox-m",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

            dbc.Tooltip(
                "POX/M = Initiator Concentration (P0X)  / Monomer Concentration (M)",
                target="tooltip-pox-m",
                placement="top"
            ),

            dcc.Input(
                id="POX_M_value",
                type='number',
                value=0.001,
                disabled=False,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '20px'}),

        html.Br(),

        html.Div([
            html.Div("X:", id="tooltip-x",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

            dbc.Tooltip(
                "X = Monomer conversion",
                target="tooltip-x",
                placement="top"
            ),

            dcc.Input(
                id="final_X_value",
                type='number',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("PDI:", id="tooltip-pdi",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

            dbc.Tooltip(
                "PDI = Polydispersity",
                target="tooltip-pdi",
                placement="top"
            ),

            dcc.Input(
                id="final_PDI_value",
                type='number',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),

            html.Div("Mn [g⋅mol-1]:", id="tooltip-mn",
                     style={'width': '150px', 'textAlign': 'center', 'paddingRight': '10px', 'fontWeight': 'bold'}),

            dbc.Tooltip(
                "Mn = Number-average molar mass",
                target="tooltip-mn",
                placement="top"
            ),

            dcc.Input(
                id="final_Mn_value",
                type='number',
                value="",
                disabled=True,
                style={'width': '80px', 'textAlign': 'center', 'fontWeight': 'bold'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '20px'}),

        html.Div([
            html.Br(),
            html.Button('Run Simulation!', id='simulation-once-btn', n_clicks=0,
                        style={'backgroundColor': 'orange', 'color': 'white', 'fontWeight': 'bold',
                               'fontSize': '20px'}),
        ], style={'textAlign': 'center'}),

        html.Br(),
        dbc.Spinner(html.Div(id="loading-output4"), spinner_style={"marginTop": "40px"}),

        html.Div([
            html.Br(),
            html.Br(),
            dcc.Graph(id='graph1', style={'display': 'none'}),
            dcc.Graph(id='graph2', style={'display': 'none'}),
            dcc.Graph(id='graph3', style={'display': 'none'}),
            dcc.Graph(id='graph4', style={'display': 'none'}),
            dcc.Graph(id='graph5', style={'display': 'none'}),
            dcc.Graph(id='graph6', style={'display': 'none'}),
        ], style={'align': 'center', 'width': '50%', 'margin-left': 'auto', 'margin-right': 'auto'}),

    ], style={'textAlign': 'center'})

    return layout
