import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, Patch
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc


def layout_report():
    df = pd.read_excel('datasets/ODEs_Dataset.xlsx')
    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    layout = html.Div([
        html.Br(),
        html.Div([
            html.Button('Update Report', id='create-report-btn', n_clicks=0,
                        style={'backgroundColor': 'orange', 'color': 'white', 'fontWeight': 'bold', 'fontSize': '20px',
                               'margin': 'auto'}),
            html.Br(),
            html.Br(),
            dbc.Spinner(spinner_style={"marginTop": "20px"},
                        children=[html.Div(id="macro-output")]),
        ], style={'textAlign': 'center'}),

        html.Div([
            html.Br(),
            html.Iframe(id='html-viewer', src="assets/relatorio_analise.html", width='80%', height='600'),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '10px'}),
    ])

    return layout

