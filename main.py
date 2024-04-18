import re
import time
from datetime import datetime

from doe.DOE import *
from apps.DataAnalytics import *
from apps.odes import *
from apps.optimization import *

from layouts.Layout_DOE import *
from layouts.layout_parallel_chart import *
from layouts.layout_report import *
from layouts.layout_simulate import *
from layouts.layout_about import *
from layouts.layout_mlp_setup import *
from layouts.layout_fast_mlp import *
from layouts.layout_advanced_mlp import *
from layouts.layout_mlp_evaluation import *
from layouts.layout_optimization import *
from layouts.layout_solve_once import *

from apps.ReactionConstants import *

from keras_files.KerasMLP import *
from keras_files.KerasPredict import *
from keras_files.KerasMLP_OPT import *

M = 5.82
MWm = 104.15
Hours = 40

with open('assets/status.txt', 'w') as file:
    file.write(str(0.0))

input_columns, output_columns, drop_options = initial_columns()

# Inicializa o app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.title = "ARGET ATRP"

server = app.server

initial_df_data = pd.read_excel('datasets/DOE_Setup.xlsx', sheet_name='DOE')

app.layout = html.Div([
    html.Br(),
    html.Br(),
    html.Div([
        html.Img(src='assets/logo.png', style={'height': '100px', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ], style={'text-align': 'center', 'margin-bottom': '10px'}),

    html.Div([
        dcc.Tabs(id='tabs', value='Simulate_Once', children=[
            dcc.Tab(label='Solve ODEs', value='Simulate_Once',
                    style={'fontSize': '16px'},
                    selected_style={'fontSize': '16px', 'backgroundColor': 'blue', 'color': 'white'}),
            dcc.Tab(label='DOE Setup', value='DOE',
                    style={'fontSize': '16px'},
                    selected_style={'fontSize': '16px', 'backgroundColor': 'blue', 'color': 'white'}),
            dcc.Tab(label='Run DOE', value='Simulate',
                    style={'fontSize': '16px'},
                    selected_style={'fontSize': '16px', 'backgroundColor': 'blue', 'color': 'white'}),
            dcc.Tab(label='EDA', value='Data_Analytics',
                    style={'fontSize': '16px'},
                    selected_style={'fontSize': '16px', 'backgroundColor': 'blue', 'color': 'white'}),
            dcc.Tab(label='Parallel Chart', value='Parallel_Chart',
                    style={'fontSize': '16px'},
                    selected_style={'fontSize': '16px', 'backgroundColor': 'blue', 'color': 'white'}),
            dcc.Tab(label='MLP Setup', value='MLP_Setup',
                    style={'fontSize': '16px'},
                    selected_style={'fontSize': '16px', 'backgroundColor': 'blue', 'color': 'white'}),
            dcc.Tab(label='Fast MLP', value='Fast_MLP_Training',
                    style={'fontSize': '16px'},
                    selected_style={'fontSize': '16px', 'backgroundColor': 'blue', 'color': 'white'}),
            dcc.Tab(label='Advanced MLP', value='Advanced_MLP_Training',
                    style={'fontSize': '16px'},
                    selected_style={'fontSize': '16px', 'backgroundColor': 'blue', 'color': 'white'}),
            dcc.Tab(label='MLP Evaluation', value='MLP_Evaluation',
                    style={'fontSize': '16px'},
                    selected_style={'fontSize': '16px', 'backgroundColor': 'blue', 'color': 'white'}),
            dcc.Tab(label='Optimization', value='Optimization',
                    style={'fontSize': '16px'},
                    selected_style={'fontSize': '16px', 'backgroundColor': 'blue', 'color': 'white'}),
            dcc.Tab(label='About', value='About',
                    style={'fontSize': '16px'},
                    selected_style={'fontSize': '16px', 'backgroundColor': 'blue', 'color': 'white'}),
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
    elif selected_tab == 'Simulate_Once':
        return layout_solve_once(M, MWm, Hours)
    elif selected_tab == 'Simulate':
        return layout_simulate(M, MWm, Hours)
    elif selected_tab == 'Data_Analytics':
        return layout_report()
    elif selected_tab == 'Parallel_Chart':
        return parallel_chart()
    elif selected_tab == 'MLP_Setup':
        return layout_mlp_setup(input_columns, output_columns, drop_options)
    elif selected_tab == 'Fast_MLP_Training':
        return layout_fast_mlp()
    elif selected_tab == 'Advanced_MLP_Training':
        return layout_advanced_mlp()
    elif selected_tab == 'MLP_Evaluation':
        return layout_mlp_evaluation(input_columns)
    elif selected_tab == 'Optimization':
        return layout_optimization(output_columns)
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
        return descriptive_stats.to_dict('records')

    descriptive_stats = df_ODES_Dataset.describe().reset_index()
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
    global Hours, MWm, M
    t = None
    y = None
    SimulateODEs(Hours, MWm, M)
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


@app.callback(Output('column-input-selector', 'value'),
              Input('column-input-selector', 'value'))
def update_table(selected_columns):
    global input_columns
    input_columns = selected_columns
    return selected_columns


@app.callback(Output('column-output-selector', 'value'),
              Input('column-output-selector', 'value'))
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


@app.callback(Output('reaction_time_value', 'value'),
              Input('reaction_time_value', 'value'))
def update_time(value):
    global Hours
    Hours = value
    return value


@app.callback(Output('styrene_monomer_value', 'value'),
              Input('styrene_monomer_value', 'value'))
def update_styrene_monomer_value(value):
    global M
    M = value
    return value


@app.callback(Output('monomer_molar_mass_value', 'value'),
              Input('monomer_molar_mass_value', 'value'))
def update_monomer_molar_mass_value(value):
    global MWm
    MWm = value
    return value


@app.callback(Output('optimize-text', 'value'),
              Output('optimization-time', 'value'),
              Output("loading-output4", "children", allow_duplicate=True),
              Input('optimize-button', 'n_clicks'),
              State('input-desired-values-text', 'value'),
              prevent_initial_call=True)
def run_opt(n_clicks, input_value):
    inicio = time.time()
    try:
        input_list = re.split(r'\s*[,\s;]\s*', input_value)
        desired_values = np.array(list(map(float, input_list)))
        ypred = optimize(desired_values, input_columns)

        predicted_str = ""
        for i in range(len(ypred)):
            valor_formatado = f"{ypred[i]:.3e}"
            predicted_str += f"{input_columns[i]}:  {valor_formatado}\n"

        termino = time.time()
        duracao = termino - inicio

        minutos = int(duracao // 60)
        segundos = int(duracao % 60)

        exec_time = f"Optimization Time = {minutos}:{segundos:02d}"

    except Exception as e:
        predicted_str = ""
        exec_time = "Optimization Error"

    return predicted_str, exec_time, ""


@app.callback(Output('graph1', 'figure'),
              Output('graph1', 'style'),
              Output('graph2', 'figure'),
              Output('graph2', 'style'),
              Output('graph3', 'figure'),
              Output('graph3', 'style'),
              Output('graph4', 'figure'),
              Output('graph4', 'style'),
              Output("loading-output4", "children", allow_duplicate=True),
              Input('simulation-once-btn', 'n_clicks'),
              State('reaction_time_value_2', 'value'),
              State('styrene_monomer_value_2', 'value'),
              State('monomer_molar_mass_value_2', 'value'),
              State('POX_M_value', 'value'),
              State('C_A_value', 'value'),
              State('POX_C_value', 'value'),
              prevent_initial_call=True)
def simulate(n_clicks, reaction_time_value_2, styrene_monomer_value_2, monomer_molar_mass_value_2, POX_M_value, C_A_value, POX_C_value):
    if n_clicks > 0:
        fig1, fig2, fig3, fig4 = SimulateODEs_Once(reaction_time_value_2, monomer_molar_mass_value_2, styrene_monomer_value_2,
                                 POX_M_value, C_A_value, POX_C_value)
        style = {'display': 'block'}
        return fig1, style, fig2, style, fig3, style, fig4, style, ""

    style = {'display': 'none'}
    return None, style, None, style, None, style, None, style, ""

if __name__ == '__main__':
    app.run_server(host='127.0.0.5', port=8080, debug=False)



