import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, Patch
import pandas as pd
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objs as go
import plotly.express as px


def create_parallel_coordinates(df, constraints=None):
    fig = px.parallel_coordinates(df, color=df.columns[-1])
    if constraints:
        dimensions = []
        for var, (min_val, max_val) in constraints.items():
            dimensions.append({
                'label': var,
                'constraintrange': [float(min_val), float(max_val)]
            })
        fig.update_traces(dimensions=dimensions)
    return fig


def create_min_max_table(df):
    min_max_df = pd.DataFrame({
        'Variable': df.columns,
        'Min Value': df.min(),
        'Max Value': df.max()
    })
    return min_max_df


def create_descriptive_table(df):
    desc_df = df.describe().transpose().reset_index()
    desc_df.rename(columns={'index': 'Variable'}, inplace=True)
    return desc_df


def initialconstraints(df):
    initial_constraints = {
        var: (float(df[var].min()), float(df[var].max()))
        for var in df.columns
    }
    return initial_constraints


def parallel_chart():
    df = pd.read_excel('datasets/ODEs_Dataset.xlsx')
    if 'index' in df.columns:
        df = df.drop(columns=['index'])

    desc_df = create_descriptive_table(df)
    exportfile = 'datasets/Parallel_Filter_Stats.xlsx'
    desc_df.to_excel(exportfile, index=False)

    initial_figure = create_parallel_coordinates(df, initialconstraints(df))

    layout = html.Div([
        dcc.Graph(id='parallel-coordinates', figure=initial_figure),

        html.Br(),
        html.Div([
            html.H2('Parallel Chart Filters'),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                  'marginBottom': '10px'}),

        html.Div([
            dcc.Upload(
                id='upload-filters',
                children=html.Div(['Drag or ', html.A('Select a File with Chart Filters!')]),
                style={
                    'width': '400px', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '3px', 'borderStyle': 'dashed', 'borderRadius': '10px',
                    'textAlign': 'center'
                },
                multiple=False
            ),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'marginBottom': '10px'}),

        dash_table.DataTable(
            id='table_data_analysis_min_max',
            columns=[
                {'name': 'Variable', 'id': 'Variable', 'editable': False},
                {'name': 'Min Value', 'id': 'Min Value', 'editable': True},
                {'name': 'Max Value', 'id': 'Max Value', 'editable': True}
            ],
            data=create_min_max_table(df).to_dict('records'),
            editable=True,
            style_table={'width': '75%', 'margin': 'auto'},
            style_header={'fontWeight': 'bold'},
            style_cell={'textAlign': 'center'}
        ),

        html.Br(),
        html.Div([
            html.Button('Download Descriptive Statistics', id='btn-stat-download', n_clicks=0,
                        style={'backgroundColor': 'orange', 'color': 'white', 'fontWeight': 'bold',
                               'fontSize': '20px',
                               'marginRight': '50px'}),
            html.Button('RESET Filters', id='reset-button', n_clicks=0,
                        style={'backgroundColor': 'red', 'color': 'white', 'fontWeight': 'bold',
                               'fontSize': '20px',
                               'marginRight': '50px'}),
            html.Button('SAVE Filters', id='save-button', n_clicks=0,
                        style={'backgroundColor': 'green', 'color': 'white', 'fontWeight': 'bold',
                               'fontSize': '20px'}),
            dcc.Download(id="download-excel")
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center',
                  'marginBottom': '10px'}),

        html.Br(),
        dash_table.DataTable(
            id='table_data_descriptive',
            columns=[
                {'name': col, 'id': col, 'type': 'numeric',
                 'format': Format(precision=4, scheme=Scheme.exponent)}
                for col in create_descriptive_table(df).columns
            ],
            data=create_descriptive_table(df).to_dict('records'),
            style_table={'width': '75%', 'margin': 'auto'},
            style_header={'fontWeight': 'bold'},
            style_cell={'textAlign': 'center'}
        ),

        dcc.Store(id='activefilters', data={})
    ])

    return layout


