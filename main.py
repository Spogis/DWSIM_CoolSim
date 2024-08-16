import re
import time
from datetime import datetime
import base64
from PIL import Image
from dash.exceptions import PreventUpdate

from doe.DOE import *
from apps.DataAnalytics import *
from apps.odes import *
from apps.run_DWSIM import *
from apps.mlp_validation import *
from apps.excel_aut import *

from layouts.layout_DOE import *
from layouts.layout_parallel_chart import *
from layouts.layout_report import *
from layouts.layout_simulate import *
from layouts.layout_about import *
from layouts.layout_mlp_setup import *
from layouts.layout_fast_mlp import *
from layouts.layout_advanced_mlp import *
from layouts.layout_mlp_evaluation import *
from layouts.layout_solve_once import *
from layouts.layout_validate import *
from layouts.layout_upload_results import *

from keras_files.KerasMLP import *
from keras_files.KerasPredict import *
from keras_files.KerasMLP_OPT import *


with open('assets/status.txt', 'w') as file:
    file.write(str(0.0))

MLP_Type = "Direct MLP"
input_columns = ['Evaporator Temperature', 'Condenser Temperature', 'Adiabatic Efficiency']
output_columns = ['Compressor Energy', 'Electric Current', 'Discharge Temperature', 'Refrigerant Mass Flow']
drop_options = initial_columns()


# Inicializa o app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "DWSIM CoolSim (IA)"

server = app.server

initial_df_data = pd.read_excel('datasets/DOE_Setup.xlsx', sheet_name='DOE')

app.layout = html.Div([
    html.Br(),
    html.Br(),
    html.Div([
        html.Img(src='assets/logo.png',
                 style={'width': '100%', 'height': 'auto', 'margin-left': 'auto',
                        'margin-right': 'auto', 'position': 'fixed', 'top': '0', 'left': '0', 'z-index': '1000'}),
    ], style={'text-align': 'center', 'margin-bottom': '10px'}),

    html.Div([
        html.Div([
            html.Br(),  # Adiciona um espaço entre o logo e as abas
            dcc.Tabs(id='tabs', value='Simulate_Once', children=[
                dcc.Tab(label='Solve DWSIM', value='Simulate_Once',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='DOE Setup', value='DOE',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='Run DOE', value='Simulate',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='Upload Results', value='Upload',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '20px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '20px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),

                dcc.Tab(label='EDA', value='Data_Analytics',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='Parallel Chart', value='Parallel_Chart',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '20px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '20px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),


                dcc.Tab(label='MLP Setup', value='MLP_Setup',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='Fast MLP', value='Fast_MLP_Training',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='Advanced MLP', value='Advanced_MLP_Training',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
                dcc.Tab(label='MLP Prediction', value='MLP_Evaluation',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '20px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '20px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),

                dcc.Tab(label='MLP Test', value='MLP_Validation',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '20px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '20px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),

                dcc.Tab(label='About', value='About',
                        style={'fontSize': '14px', 'width': '200px', 'padding': '10px', 'border': '1px solid #ccc',
                               'border-radius': '5px', 'margin-bottom': '5px', 'background-color': '#f9f9f9'},
                        selected_style={'fontSize': '14px', 'backgroundColor': '#007BFF', 'color': 'white',
                                        'width': '200px', 'padding': '10px', 'border': '1px solid #007BFF',
                                        'border-radius': '5px', 'margin-bottom': '5px',
                                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}),
            ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh', 'width': '220px',
                      'padding': '10px', 'position': 'fixed', 'margin-top': '50px', 'left': '0', 'z-index': '999'}),
        ], style={'display': 'flex'}),
        html.Div(id='tabs-content',
                 style={'flex': 1, 'padding': '20px', 'border': '1px solid #ccc', 'border-radius': '10px',
                        'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)', 'margin-left': '220px', 'margin-top': '50px',
                        'overflow-y': 'auto', 'height': 'calc(100vh - 100px)'})
    ], style={'display': 'flex'}),
    dcc.Store(id='store-data'),
])



@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def update_tab_content(selected_tab):
    if selected_tab == 'DOE':
        return layout_DOE()
    elif selected_tab == 'Simulate_Once':
        return layout_solve_once()
    elif selected_tab == 'Simulate':
        return layout_simulate()
    elif selected_tab == 'Upload':
        return layout_upload_results()
    elif selected_tab == 'Data_Analytics':
        return layout_report()
    elif selected_tab == 'Parallel_Chart':
        return parallel_chart()
    elif selected_tab == 'MLP_Setup':
        return layout_mlp_setup(input_columns, output_columns, drop_options, MLP_Type)
    elif selected_tab == 'Fast_MLP_Training':
        return layout_fast_mlp()
    elif selected_tab == 'Advanced_MLP_Training':
        return layout_advanced_mlp()
    elif selected_tab == 'MLP_Evaluation':
        return layout_mlp_evaluation(input_columns)
    elif selected_tab == 'MLP_Validation':
        return layout_validate()
    elif selected_tab == 'About':
        return layout_about()


@app.callback(Output('create-doe-btn', 'children', allow_duplicate=True),
              Input('create-doe-btn', 'n_clicks'),
              [State('table', 'data'),
               State('numero_de_simulacoes', 'value')],
              prevent_initial_call=True)
def create_doe(n_clicks, rows, num_simulacoes):
    df_to_save = pd.DataFrame(rows)
    filepath = 'datasets/DOE_Setup.xlsx'

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
    df_ODES_Dataset = pd.read_excel('datasets/ODEs_Dataset.xlsx')
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
        exportfile = 'datasets/Parallel_Filter_Stats.xlsx'
        descriptive_stats.to_excel(exportfile, index=False)
        return descriptive_stats.to_dict('records')

    descriptive_stats = df_ODES_Dataset.describe().reset_index()
    exportfile = 'datasets/Parallel_Filter_Stats.xlsx'
    descriptive_stats.to_excel(exportfile, index=False)
    return descriptive_stats.to_dict('records')

@app.callback(
    Output('activefilters', 'data'),
    Input("graph-parcoords", "restyleData")
)
def updateFilters(data):
    df_ODES_Dataset = pd.read_excel('datasets/ODEs_Dataset.xlsx')
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
    df_ODES_Dataset = pd.read_excel('datasets/ODEs_Dataset.xlsx')
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
               Output("progress", "label", allow_duplicate=True),
               Output("loading-output3", "children", allow_duplicate=True)],
              Input('simulation-btn', 'n_clicks'),
              prevent_initial_call=True)
def simulate(n_clicks):
    Simulate_DWSIM_DOE()
    with open('assets/status.txt', 'w') as file:
        file.write(str(100))
    return "Simulation Finished!", 100, 100, ""


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
        filepath = 'datasets/DOE_Setup.xlsx'

        # Usando ExcelWriter para salvar em abas diferentes
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            df_to_save.to_excel(writer, sheet_name='DOE', index=False)  # Salvando dados da tabela na aba 'DOE'

            # Salvando o número de simulações em outra aba
            df_infos = pd.DataFrame({'Number of Simulations': [num_simulacoes]})
            df_infos.to_excel(writer, sheet_name='infos', index=False)  # Salvando na aba 'infos'

        return 'DOE Configuration Saved!'
    return 'Save DOE Configuration!'


@app.callback(Output('column-input-selector', 'value', allow_duplicate=True),
              Input('column-input-selector', 'value'),
              prevent_initial_call=True)
def update_table(selected_columns):
    global input_columns
    input_columns = selected_columns
    return selected_columns


@app.callback(Output('column-output-selector', 'value', allow_duplicate=True),
              Input('column-output-selector', 'value'),
              prevent_initial_call=True)
def update_table(selected_columns):
    global output_columns
    output_columns = selected_columns
    return selected_columns


@app.callback([Output("loading-output1", "children", allow_duplicate=True),
               Output("button-output", "children", allow_duplicate=True),
               Output('r2-simple-mlp-textarea', 'value')],
              Input("run-MLP-button", "n_clicks"),
              prevent_initial_call=True)
def MLP(n_clicks):
    global input_columns, output_columns
    dataset = pd.read_excel('datasets/ODEs_Dataset.xlsx')
    if 'index' in dataset.columns:
        dataset = dataset.drop(columns=['index'])

    r2_str = RunMLP(dataset, input_columns, output_columns)

    # Caminho do diretório contendo as imagens
    directory_path = 'assets/images'

    # Lista para armazenar os componentes de imagem
    image_components = []

    # Lista de extensões de arquivo para considerar como imagens
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    # Itera sobre todos os arquivos no diretório
    for filename in os.listdir(directory_path):
        # Verifica se o arquivo é uma imagem
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            # Cria o caminho completo do arquivo
            file_path = os.path.join(directory_path, filename)
            # Cria um componente de imagem e adiciona à lista
            image_components.append(html.Img(src=file_path, style={'width': '50%', 'height': 'auto'}))

    loading_status = ""
    return loading_status, image_components, r2_str

@app.callback(Output('output-text', 'value'),
              Input('predict-button', 'n_clicks'),
              State('input-text', 'value'))
def update_output(n_clicks, input_value):
    try:
        input_list = re.split(r'\s*[,\s;]\s*', input_value)
        input_data = np.array([list(map(float, input_list))])
        ypred = PredictValues(input_data)

        predicted_str = ""
        for i in range(len(ypred)):
            valor_formatado = f"{ypred[i]:.3e}"
            predicted_str += f"{output_columns[i]}:  {valor_formatado}\n"

    except Exception as e:
        predicted_str = ""
    return predicted_str

@app.callback([Output("loading-output2", "children", allow_duplicate=True),
               Output("button-output-advanced", "children", allow_duplicate=True),
               Output('best-hps-textarea', 'value'),
               Output('model-summary-textarea', 'value'),
               Output('r2-opt-mlp-textarea', 'value')],
              Input("run-OPTMLP-button", "n_clicks"),
              prevent_initial_call=True)
def OPTMLP(n_clicks):
    global input_columns, output_columns
    dataset = pd.read_excel('datasets/ODEs_Dataset.xlsx')
    if 'index' in dataset.columns:
        dataset = dataset.drop(columns=['index'])

    best_hps_str, model_summary_str, r2_str  = RunOptimizedMLP(dataset, input_columns, output_columns)

    # Caminho do diretório contendo as imagens
    directory_path = 'assets/images'

    # Lista para armazenar os componentes de imagem
    image_components = []

    # Lista de extensões de arquivo para considerar como imagens
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    # Itera sobre todos os arquivos no diretório
    for filename in os.listdir(directory_path):
        # Verifica se o arquivo é uma imagem
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            # Cria o caminho completo do arquivo
            file_path = os.path.join(directory_path, filename)
            # Cria um componente de imagem e adiciona à lista
            image_components.append(html.Img(src=file_path, style={'width': '50%', 'height': 'auto'}))

    loading_status = ""
    return loading_status, image_components, best_hps_str, model_summary_str, r2_str


@app.callback(Output('compressor_energy_value', 'value'),
              Output('electric_current_value', 'value'),
              Output('discharge_temperature_value', 'value'),
              Output('refrigerant_mass_flow_value', 'value'),
              Output("loading-output4", "children", allow_duplicate=True),
              Output('output-dwsim-fig', 'figure', allow_duplicate=True),
              Output('output-dwsim-fig', 'style', allow_duplicate=True),
              Input('dwsim-once-btn', 'n_clicks'),
              State('evaporator_temperature_value', 'value'),
              State('condenser_temperature_value', 'value'),
              State('adiabatic_efficiency_value', 'value'),
              prevent_initial_call=True)
def simulate(n_clicks, evaporator_temperature_value, condenser_temperature_value, adiabatic_efficiency_value):
    if n_clicks > 0:
        energy, discharge_temperature, mass_flow = run_DWSIM(evaporator_temperature=evaporator_temperature_value,
                                                             condenser_temperature=condenser_temperature_value,
                                                             adiabatic_efficiency=adiabatic_efficiency_value,
                                                             picture='Yes')

        energy = energy * 1000
        discharge_temperature = discharge_temperature - 273.15
        mass_flow = mass_flow * 60
        electric_current = energy / 220

        energy = "{:1.2f}".format(energy)
        discharge_temperature = "{:1.2f}".format(discharge_temperature)
        mass_flow = "{:1.2f}".format(mass_flow)
        electric_current = "{:1.2f}".format(electric_current)

        # Caminho da imagem com identificador único
        image_filename = f'assets/pfd.png?t={time.time()}'

        # Criar uma nova figura a cada execução
        fig = go.Figure()

        # Constantes
        img_width = 1024
        img_height = 768
        scale_factor = 1.2

        # Adicionar um traço invisível para auxiliar na lógica de redimensionamento automático
        fig.add_trace(
            go.Scatter(
                x=[0, img_width * scale_factor],
                y=[0, img_height * scale_factor],
                mode="markers",
            )
        )

        # Configurar os eixos
        fig.update_xaxes(
            visible=False,
            range=[0, img_width * scale_factor]
        )

        fig.update_yaxes(
            visible=False,
            range=[0, img_height * scale_factor],
            scaleanchor="x"
        )

        # Adicionar a imagem
        fig.add_layout_image(
            dict(
                x=0,
                sizex=img_width * scale_factor,
                y=img_height * scale_factor,
                sizey=img_height * scale_factor,
                xref="x",
                yref="y",
                opacity=1.0,
                layer="below",
                sizing="stretch",
                source=image_filename)
        )

        # Configurar o layout
        fig.update_layout(
            width=img_width * scale_factor,
            height=img_height * scale_factor,
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        )

        # Retornar os valores e a figura
        return energy, electric_current, discharge_temperature, mass_flow, "", fig, {'display': 'block'}

    # Caso nenhum clique tenha ocorrido
    return None, None, None, None, "", None, {'display': 'none'}

@app.callback([Output('validation-btn', 'children', allow_duplicate=True),
               Output("progress2", "value", allow_duplicate=True),
               Output("progress2", "label", allow_duplicate=True),
               Output("loading-output5", "children", allow_duplicate=True),
               Output('Compressor_Energy_rscore', 'value'),
               Output('Electric_Current_rscore', 'value'),
               Output('Discharge_Temperature_rscore', 'value'),
               Output('Refrigerant_Mass_Flow_rscore', 'value'),
               Output('Evaporator_Temperature_rscore', 'value'),
               Output('Condenser_Temperature_rscore', 'value'),
               Output('Adiabatic_Efficiency_rscore', 'value')],
              Input('validation-btn', 'n_clicks'),
              State('Validation_Cases', 'value'),
              prevent_initial_call=True)
def simulate(n_clicks, ValidationCases):
    print(input_columns)

    if input_columns == ['Evaporator Temperature', 'Condenser Temperature', 'Adiabatic Efficiency']:
        MLP_Validation(ValidationCases)
        #Calculate R2 Scores
        df = pd.read_excel('datasets/MLP_Validation_Dataset.xlsx')

        energy_r2 = mean_absolute_percentage_error(df['MLP_Compressor_Energy'],
                                                   df['Compressor Energy'])*100
        energy_formatado = f"{energy_r2:.2f}%"

        electric_current_r2 = mean_absolute_percentage_error(df['MLP_Electric_Current'],
                                                             df['Electric Current'])*100
        electric_current_formatado = f"{electric_current_r2:.2f}%"

        discharge_temperature_r2 = mean_absolute_percentage_error(df['MLP_Discharge_Temperature'],
                                                                  df['Discharge Temperature'])*100
        discharge_temperature_formatado = f"{discharge_temperature_r2:.2f}%"

        mass_flow_r2 = mean_absolute_percentage_error(df['MLP_Refrigerant_Mass_Flow'],
                                                      df['Refrigerant Mass Flow']) * 100
        mass_flow_formatado = f"{mass_flow_r2:.2f}%"

        with open('assets/status2.txt', 'w') as file:
            file.write(str(100))

        return ("MLP Test Finished!", 100, 100, "", energy_formatado, electric_current_formatado,
                discharge_temperature_formatado, mass_flow_formatado, "N/A", "N/A", "N/A")

    if input_columns == ['Compressor Energy', 'Electric Current', 'Discharge Temperature', 'Refrigerant Mass Flow']:
        Inverse_MLP_Validation(ValidationCases)
        # Calculate R2 Scores
        df = pd.read_excel('datasets/MLP_Validation_Dataset.xlsx')

        Evaporator_Temperature_r2 = mean_absolute_percentage_error(df['MLP_Evaporator_Temperature'],
                                                  df['Evaporator Temperature'])*100
        Evaporator_Temperature_formatado = f"{Evaporator_Temperature_r2:.2f}%"

        Condenser_Temperature_r2 = mean_absolute_percentage_error(df['MLP_Condenser_Temperature'],
                                                df['Condenser Temperature'])*100
        Condenser_Temperature_formatado = f"{Condenser_Temperature_r2:.2f}%"

        Adiabatic_Efficiency_r2 = mean_absolute_percentage_error(df['MLP_Adiabatic_Efficiency'],
                                                  df['Adiabatic Efficiency'])*100
        Adiabatic_Efficiency_formatado = f"{Adiabatic_Efficiency_r2:.2f}%"

        with open('assets/status2.txt', 'w') as file:
            file.write(str(100))

        return ("MLP Test Finished!", 100, 100, "", "N/A", "N/A", "N/A", "N/A",
                Evaporator_Temperature_formatado, Condenser_Temperature_formatado, Adiabatic_Efficiency_formatado)


@app.callback(
    [Output("progress2", "value", allow_duplicate=True),
     Output("progress2", "label", allow_duplicate=True)],
    Input('validation-btn', 'n_clicks'),
    Input("progress-interval2", "n_intervals"),
    prevent_initial_call=True)
def update_progress(n_clicks, n):
    status = 'assets/status2.txt'
    with open(status, 'r') as file:
        progress_value = file.read()
        progress_value = float(progress_value)

    progress = min(progress_value % 110, 100)
    # only add text after 20% progress to ensure text isn't squashed too much
    return progress, f"{progress} %" if progress >= 20 else ""


@app.callback(Output('column-input-selector', 'value', allow_duplicate=True),
              Output('column-output-selector', 'value', allow_duplicate=True),
              Input('MLP-setup-selector','value'),
              prevent_initial_call=True)
def change_MLP_setup(MLP_setup_selector):
    global input_columns, output_columns, drop_options
    global MLP_Type

    if MLP_setup_selector == "Direct MLP":
        MLP_Type = "Direct MLP"
        input_columns = ['Evaporator Temperature', 'Condenser Temperature', 'Adiabatic Efficiency']
        output_columns = ['Compressor Energy', 'Electric Current', 'Discharge Temperature', 'Refrigerant Mass Flow']
        return input_columns, output_columns
    elif MLP_setup_selector == "Inverse MLP":
        MLP_Type = "Inverse MLP"
        input_columns = ['Compressor Energy', 'Electric Current', 'Discharge Temperature', 'Refrigerant Mass Flow']
        output_columns = ['Evaporator Temperature', 'Condenser Temperature', 'Adiabatic Efficiency']
        return input_columns, output_columns
    else:
        MLP_Type = "Custom MLP"
        return drop_options, drop_options


@app.callback([Output('table', 'data', allow_duplicate=True),
               Output('numero_de_simulacoes', 'value'),
               Output('save-doe-btn', 'children', allow_duplicate=True),
               Output('create-doe-btn', 'children', allow_duplicate=True)],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')],
              prevent_initial_call=True)
def update_output(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'xlsx' in filename:
            initial_df_data = pd.read_excel(io.BytesIO(decoded), sheet_name='DOE')
            simulation_cases = pd.read_excel(io.BytesIO(decoded), sheet_name='infos')
            cases = simulation_cases['Number of Simulations'].max()
            rows = initial_df_data.to_dict('records')
            return rows, cases, 'Save DOE Configuration!', 'Create DOE'
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


@app.callback([Output('table', 'data', allow_duplicate=True),
               Output('save-doe-btn', 'children', allow_duplicate=True),
               Output('create-doe-btn', 'children', allow_duplicate=True)],
              Input('adding-rows-btn', 'n_clicks'),
              [State('table', 'data')],
              prevent_initial_call=True
)

def add_row(n_clicks, rows):
    if n_clicks > 0:
        rows.append({col: ('' if col not in ['Variable Name', 'Variable Type', 'Trust Level']
                           else '0.95' if col == 'Trust Level'  # Definindo 95% para a coluna 'Trust Level'
                           else variable_types[0] if col == 'Variable Type'
                           else '')
                     for col in df.columns})
    return rows, 'Save DOE Configuration!', 'Create DOE'


@app.callback(Output('table', 'style_data_conditional', allow_duplicate=True),
              Output('save-doe-btn', 'children', allow_duplicate=True),
              Output('create-doe-btn', 'children', allow_duplicate=True),
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
    return conditions, 'Save DOE Configuration!', 'Create DOE'


@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-results', 'contents'),
    prevent_initial_call=True)
def save_uploaded_file(contents):
    global drop_options, MLP_Type
    global input_columns, output_columns
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')

    if 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type:
        decoded = base64.b64decode(content_string)
        # Salvando o arquivo diretamente no diretório especificado
        with open('datasets/ODEs_Dataset.xlsx', 'wb') as f:
            f.write(decoded)
        drop_options = initial_columns()
        MLP_Type = "Custom MLP"
        input_columns = drop_options
        output_columns = drop_options

        return html.Div([
            html.H6('File uploaded and saved successfully!')
        ])
    else:
        return html.Div([
            html.H6('Please upload a file in .xlsx format!')
        ])


@app.callback(
    Output("download-excel", "data"),
    Input("btn-stat-download", "n_clicks"),
    prevent_initial_call=True
)
def download_excel(n_clicks):
    # Caminho para o arquivo Excel existente
    path_to_excel = "datasets/Parallel_Filter_Stats.xlsx"
    return dcc.send_file(path_to_excel)


@app.callback(Output('graph-parcoords', 'figure'),
              Input('table_data_analysis_min_max', 'data'),
              State('activefilters', 'data'))
def update_figure(rows, stored_filters):
    filters = {col: [float(rows[1][col]), float(rows[0][col])] for col in rows[0] if col in rows[1]}
    df = pd.read_excel('datasets/ODEs_Dataset.xlsx')
    if 'index' in df.columns:
        df = df.drop(columns=['index'])
    fig = update_graph_parcoords_min_max(df, filters)
    return fig

#######################################################################################################################
# RUN EXCEL
#######################################################################################################################

@app.callback(Output('Altura', 'value'),
              Input('run-excel-btn', 'n_clicks'),
              State('Volume', 'value'))
def runexcel(n_clicks, Volume):
    if n_clicks > 0:
        valor = atualiza_planilha(Volume)
        return valor
    return None


if __name__ == '__main__':
    app.run_server(debug=False)
