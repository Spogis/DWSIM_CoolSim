import dash
from dash import html, dcc
from dash import Input, Output
import dash_bootstrap_components as dbc

# Inicializa o app Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Kauê"
server = app.server

def Nome_Pagina(nome):
    print(nome)


app.layout = html.Div([
    html.Br(),
    html.Br(),
    html.Div([
        html.Img(src='assets/logo.png', style={'height': '100px', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ], style={'text-align': 'center', 'margin-bottom': '10px'}),

    html.Div([
        dcc.Tabs(id='tabs', value='DOE', children=[
            dcc.Tab(label='Generate DOE', value='DOE'),
            dcc.Tab(label='Solve ODEs', value='Simulate'),
        ], style={'align': 'center', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ]),
    html.Div(id='tabs-content'),
])


@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def atualiza_tab(valor_do_tab):
    if valor_do_tab == 'DOE':
        layout = html.Div([
            html.Br(),
            html.Br(),
            html.Div([
                html.H5("Você selecionou a aba Generate DOE"),
            ], style={'text-align': 'center', 'margin-bottom': '10px'}),
        ])
        Nome_Pagina(valor_do_tab)
        return layout
    if valor_do_tab == 'Simulate':
        Nome_Pagina(valor_do_tab)
        return html.H5("Você selecionou a aba Solve ODEs"),
    return ""

if __name__ == '__main__':
    app.run_server(debug=True)