import dash
from dash import html, dcc, dash_table
import os


def layout_configuration(dwsimpath):
    layout = html.Div([
        html.Div([
            # Título e input lado a lado
            html.H2("DWSIM Installation Folder:", style={'margin-right': '20px'}),
            dcc.Input(
                id='dwsim-folder-input',
                type='text',
                value=dwsimpath,
                placeholder='Enter DWSIM Installation Folder',
                style={'width': '400px', 'height': '40px', 'font-size': '16px', 'padding': '5px'}
            )
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin-bottom': '20px'}),

        # html.Div(id='output-folder', style={
        #     'text-align': 'center',
        #     'font-size': '18px',
        #     'color': '#4A4A4A',
        #     'border': '1px solid #ccc',
        #     'padding': '10px',
        #     'border-radius': '5px',
        #     'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
        # })
    ])
    return layout
