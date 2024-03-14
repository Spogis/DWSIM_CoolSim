import dash
from dash import html, dcc, dash_table
import base64

pdf_about_path = 'assets/ARGET ATRP.pdf'

def encode_pdf_to_base64(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        encoded_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    return f"data:application/pdf;base64,{encoded_pdf}"

def layout_about():
    encoded_pdf = encode_pdf_to_base64(pdf_about_path)
    layout = html.Div([
        html.Iframe(id='pdf-viewer', src=encoded_pdf, style={'width': '100%', 'height': '600px', 'margin': 'auto', 'text-align': 'center', 'display': 'flex', 'justify-content': 'center'})
    ])
    return layout
