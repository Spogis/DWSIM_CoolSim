import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, Input, Output, State, ctx, dash_table, Patch
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import math
import io
from scipy.integrate import solve_ivp
from datetime import datetime
from dash.dash_table.Format import Format, Scheme
import base64
import io

from doe.DOE import *
from assets.DataAnalytics import *
from assets.odes import *

from layouts.Layout_DOE import *
from layouts.layout_parallel_chart import *
from layouts.layout_report import *
from layouts.layout_simulate import *
from layouts.layout_about import *

with open('assets/status.txt', 'w') as file:
    file.write(str(0.0))

# Inicializa o app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "ARGET ATRP"

server = app.server

initial_df_data = pd.read_excel('datasets/ARGET_ATRP_Input.xlsx', sheet_name='DOE')

app.layout = html.Div([
    html.Br(),
    html.Br(),
    html.Div([
        html.Img(src='assets/logo.png', style={'height': '100px', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ], style={'text-align': 'center', 'margin-bottom': '10px'}),

    html.Div([
        dcc.Tabs(id='tabs', value='DOE', children=[
            dcc.Tab(label='Generate DOE', value='DOE'),
            dcc.Tab(label='Generate Dataset with Simulated Cases', value='Simulate'),
            dcc.Tab(label='Exploratory Data Analysis', value='Data_Analytics'),
            dcc.Tab(label='Parallel Chart', value='Parallel_Chart'),
            dcc.Tab(label='MLP Setup', value='MLP_Setup'),
            dcc.Tab(label='MLP Training', value='MLP_Training'),
            dcc.Tab(label='MLP Evaluation', value='MLP_Evaluation'),
            dcc.Tab(label='About', value='About'),
        ], style={'align': 'center', 'width': '100%', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ]),
    dcc.Store(id='store-data'),
    html.Div(id='tabs-content'),
])


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def update_tab_content(selected_tab):
    if selected_tab == 'DOE':
        return layout_DOE()
    elif selected_tab == 'Simulate':
        return layout_simulate()
    elif selected_tab == 'Data_Analytics':
        return layout_report()
    elif selected_tab == 'Parallel_Chart':
        return parallel_chart()
    elif selected_tab == 'MLP_Setup':
        return ""
    elif selected_tab == 'MLP_Training':
        return ""
    elif selected_tab == 'MLP_Evaluation':
        return ""
    elif selected_tab == 'About':
        return layout_about()

@app.callback(Output('table', 'style_data_conditional', allow_duplicate=True),
              Input('table', 'data'),
              prevent_initial_call=True)
def update_editability(rows):
    # Verifica se alguma linha tem 'Variable Type' definido como 'Discrete'
    conditions = []
    for i, row in enumerate(rows):
        if row['Variable Type'] == 'Discrete':
            conditions.append({
                'if': {'row_index': i, 'column_id': 'Step (If Variable is Discrete)'},
                'backgroundColor': '#FAFAFA',
                'border': '1px solid blue',
                # Aqui você pode aplicar estilos adicionais se necessário
            })
    return conditions


@app.callback(Output('create-doe-btn', 'children', allow_duplicate=True),
              Input('create-doe-btn', 'n_clicks'),
              [State('table', 'data'),
               State('numero_de_simulacoes', 'value')],
              prevent_initial_call=True)
def create_doe(n_clicks, rows, num_simulacoes):
    df_to_save = pd.DataFrame(rows)
    filepath = 'datasets/ARGET_ATRP_Input.xlsx'

    with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
        df_to_save.to_excel(writer, sheet_name='DOE', index=False)

        # Salvando o número de simulações em outra aba
        df_infos = pd.DataFrame({'Number of Simulations': [num_simulacoes]})
        df_infos.to_excel(writer, sheet_name='infos', index=False)

    NumberOfSimulations = num_simulacoes
    Run_DOE(filepath, NumberOfSimulations)

    return 'DOE Table Generated Successfully!'

@app.callback(
    Output('table_data_analysis', 'data'),
    Input("activefilters", "data")
)
def udpate_table(data):
    df_ODES_Dataset = pd.read_excel('datasets/ARGET_ATRP_ODEs_Dataset.xlsx')
    if 'index' in df_ODES_Dataset.columns:
        df_ODES_Dataset = df_ODES_Dataset.drop(columns=['index'])

    if data:
        dff = df_ODES_Dataset.copy()
        for col in data:
            if data[col]:
                rng = data[col][0]
                if isinstance(rng[0], list):
                    # if multiple choices combine df
                    dff3 = pd.DataFrame(columns=df.columns)
                    for i in rng:
                        dff2 = dff[dff[col].between(i[0], i[1])]
                        dff3 = pd.concat([dff3, dff2])
                    dff = dff3
                else:
                    # if one choice
                    dff = dff[dff[col].between(rng[0], rng[1])]
        descriptive_stats = dff.describe().reset_index()
        return descriptive_stats.to_dict('records')

    descriptive_stats = df_ODES_Dataset.describe().reset_index()
    return descriptive_stats.to_dict('records')

@app.callback(
    Output('activefilters', 'data'),
    Input("graph-parcoords", "restyleData")
)
def updateFilters(data):
    df_ODES_Dataset = pd.read_excel('datasets/ARGET_ATRP_ODEs_Dataset.xlsx')
    if 'index' in df_ODES_Dataset.columns:
        df_ODES_Dataset = df_ODES_Dataset.drop(columns=['index'])

    dims = df_ODES_Dataset.columns
    if data:
        key = list(data[0].keys())[0]
        col = dims[int(key.split('[')[1].split(']')[0])]
        newData = Patch()
        newData[col] = data[0][key]
        return newData
    return {}


@app.callback([Output('html-viewer', 'src', allow_duplicate=True),
               Output("macro-output", "children", allow_duplicate=True),
               Output('create-report-btn', 'children', allow_duplicate=True)],
              Input('create-report-btn', 'n_clicks'),
              prevent_initial_call=True)
def update_output(n_clicks):
    df_ODES_Dataset = pd.read_excel('datasets/ARGET_ATRP_ODEs_Dataset.xlsx')
    if 'index' in df_ODES_Dataset.columns:
        df_ODES_Dataset = df_ODES_Dataset.drop(columns=['index'])

    if df_ODES_Dataset is not None:
        data_analytics(df_ODES_Dataset)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        Html_Page = f"assets/relatorio_analise.html?update={timestamp}"
        return Html_Page, "", "Report Updated!"

    return "", "", "Update Report"


@app.callback([Output('simulation-btn', 'children', allow_duplicate=True),
               Output("progress", "value", allow_duplicate=True),
               Output("progress", "label", allow_duplicate=True)],
              Input('simulation-btn', 'n_clicks'),
              prevent_initial_call=True)
def simulate(n_clicks):
    t = None
    y = None
    SimulateODEs()
    with open('assets/status.txt', 'w') as file:
        file.write(str(100))
    return "Simulation Finished!", 100, 100


@app.callback(
    [Output("progress", "value", allow_duplicate=True),
     Output("progress", "label", allow_duplicate=True)],
    Input('simulation-btn', 'n_clicks'),
    Input("progress-interval", "n_intervals"),
    prevent_initial_call=True)
def update_progress(n_clicks, n):
    status = 'assets/status.txt'
    with open(status, 'r') as file:
        progress_value = file.read()
        progress_value = float(progress_value)

    progress = min(progress_value % 110, 100)
    # only add text after 20% progress to ensure text isn't squashed too much
    return progress, f"{progress} %" if progress >= 20 else ""


@app.callback(
    Output('save-doe-btn', 'children'),
    [Input('save-doe-btn', 'n_clicks')],
    [State('table', 'data'),
     State('numero_de_simulacoes', 'value')],
    prevent_initial_call=True)
def save_excel(n_clicks, rows, num_simulacoes):
    if n_clicks > 0:
        # Criando DataFrame dos dados da tabela
        df_to_save = pd.DataFrame(rows)

        # Caminho do arquivo onde o Excel será salvo
        filepath = 'datasets/ARGET_ATRP_Input.xlsx'

        # Usando ExcelWriter para salvar em abas diferentes
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            df_to_save.to_excel(writer, sheet_name='DOE', index=False)  # Salvando dados da tabela na aba 'DOE'

            # Salvando o número de simulações em outra aba
            df_infos = pd.DataFrame({'Number of Simulations': [num_simulacoes]})
            df_infos.to_excel(writer, sheet_name='infos', index=False)  # Salvando na aba 'infos'

        return 'DOE Configuration Saved!'
    return 'Save DOE Configuration!'


if __name__ == '__main__':
    app.run_server(host='127.0.0.5', port=8080, debug=False)



