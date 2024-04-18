import pandas as pd
import numpy as np
import io
from scipy.integrate import solve_ivp
import plotly.graph_objs as go

from apps.ReactionConstants import *


def ARGET_ODES(t, y):
    #Momentos de ordem zero para as cadeias:
    R0 = y[0] # Vivas
    Q0 = y[1] # Dormentes
    D0 = y[2] # Mortas

    #Momentos de ordem um para as cadeias:
    R1 = y[3] # Vivas
    Q1 = y[4] # Dormentes
    D1 = y[5] # Mortas

    #Momentos de ordem dois para as cadeias:
    R2 = y[6] # Vivas
    Q2 = y[7] # Dormentes
    D2 = y[8] # Mortas

    #Monomero
    M = y[9]

    #Iniciador
    P0X = y[10]

    #Radicais de iniciação
    P0 = y[11]

    #Espécie ativadora
    C = y[12]

    #Espécie desativadora
    CX = y[13]

    #Agente redutor
    A = y[14]

    #Agente redutor na forma oxidada
    AX = y[15]


    #Momentos de ordem zero para as cadeias:
    dR0dt = kp*M*P0 +kact*C*Q0 -kdact*CX*R0 -kt*(P0 + R0)*R0 -ktr*R0
    dQ0dt = -kact*C*Q0 +kdact*CX*R0
    dD0dt = (ktc/2)*R0*R0 +ktd*R0*R0 +kt*P0*R0 +ktr*R0

    dR1dt = kp*M*(P0 + R0) +kact*C*Q1 -kdact*CX*R1 -kt*(P0 + R0)*R1 -ktr*R1
    dQ1dt = -kact*C*Q1 +kdact*CX*R1
    dD1dt = kt*(P0 + R0)*R1 +ktr*R1

    dR2dt = kp*M*(P0 + R0 +2*R1) +kact*C*Q2 -kdact*CX*R2 -kt*(P0 + R0)*R2 -ktr*R2
    dQ2dt = -kact*C*Q2 +kdact*CX*R2
    dD2dt =kt*(P0 + R0)*R2 +ktc*R1*R1 +ktr*R2

    #Monômero
    dMdt = -kp*M*(R0 + P0)

    #Iniciador
    dP0Xdt = -kact0*P0X*C +kdact0*P0*CX

    #Radicais de iniciação
    dP0dt = -kp*M*P0 +kact0*P0X*C -kdact0*P0*CX -kt*(R0 + P0)*P0 -ktr*P0

    #Espécie ativadora
    dCdt = -kact0*P0X*C +kdact0*P0*CX -kact*C*Q0 +kdact*CX*R0 +kr*A*CX

    #Espécie desativadora
    dCXdt = -dCdt

    #Agente redutor
    dAdt = -kr*A*CX

    #Agente redutor na forma oxidada
    dAXdt = -dAdt

    return [dR0dt, dQ0dt, dD0dt,
            dR1dt, dQ1dt, dD1dt,
            dR2dt, dQ2dt, dD2dt,
            dMdt,
            dP0Xdt, dP0dt,
            dCdt, dCXdt,
            dAdt, dAXdt]


def SolveODEs(initial_conditions, reation_time):
    global t, y
    total_time = reation_time*3600
    t_eval = np.linspace(0, total_time, num=int(total_time*10), endpoint=True)

    sol = solve_ivp(ARGET_ODES,
                [0, total_time],
                initial_conditions,
                method='LSODA', #Radau / LSODA
                t_eval=t_eval)
    t = np.transpose(sol.t)
    y = np.transpose(sol.y)



def MoreUsableDataset(MWm):
    global t, y

    df_time = pd.DataFrame({'Time':t})

    temp_variable_names = ['R0', 'Q0', 'D0',
                           'R1', 'Q1', 'D1',
                           'R2', 'Q2', 'D2',
                           'M',
                           'P0X', 'P0',
                           'C', 'CX',
                           'A', 'AX']

    df_num_sol = pd.DataFrame(y, columns=temp_variable_names)
    results = pd.concat([df_time,df_num_sol], axis=1)

    results['X'] = (results['M'].iloc[0]-results['M'])/results['M'].iloc[0]
    results['DPn'] = (results['R1']+results['Q1']+results['D1'])/(results['R0']+results['Q0']+results['D0'])
    results['DPw'] = (results['R2']+results['Q2']+results['D2'])/(results['R1']+results['Q1']+results['D1'])

    results['Mn'] = MWm * results['DPn']
    results['PDI'] = results['DPw']/results['DPn']
    results['Time'] = results['Time']/3600

    return results


def SimulateODEs(reation_time, MWm, M):
    df = pd.read_excel('datasets/DOE_LHC.xlsx')
    numberofsimulations = len(df)

    exportdataset = pd.DataFrame(columns=['POX/C',
                                          'C/A',
                                          'POX/M',
                                          'X',
                                          'PDI',
                                          'Mn'])

    for i in range(numberofsimulations):
        Temp_P0XC = df.iloc[i]['POX/C']
        Temp_CA = df.iloc[i]['C/A']
        Temp_POXM = df.iloc[i]['POX/M']

        POX = Temp_POXM * M
        C = POX / Temp_P0XC
        A = C / Temp_CA

        Initial_Conditions = [0, 0, 0,
                              0, 0, 0,
                              0, 0, 0,
                              M,
                              POX, 0,
                              C, 0,
                              A, 0]

        SolveODEs(Initial_Conditions, reation_time)
        results = MoreUsableDataset(MWm)

        designdata = np.array([POX / C, C / A, POX / M,
                               results['X'].iloc[-1],
                               results['PDI'].iloc[-1],
                               results['Mn'].iloc[-1]])

        exportdataset.loc[len(exportdataset)] = designdata

        if i % int(numberofsimulations / 20) == 0:
            status = round((i / numberofsimulations) * 100, 0)
            with open('assets/status.txt', 'w') as file:
                file.write(str(status))

    exportfile = 'datasets/ODEs_Dataset.xlsx'
    exportdataset.to_excel(exportfile, index=False)


def SimulateODEs_Once(reation_time, MWm, M, POXM, CA, P0XC):
    global t, y
    POX = POXM * M
    C = POX / P0XC
    A = C / CA

    Initial_Conditions = [0, 0, 0,
                          0, 0, 0,
                          0, 0, 0,
                          M,
                          POX, 0,
                          C, 0,
                          A, 0]

    SolveODEs(Initial_Conditions, reation_time)
    ode_results = MoreUsableDataset(MWm)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ode_results['Time'], y=ode_results['X'], mode='lines'))
    fig1.update_layout(
        title={'text': 'Conversion x Time',
               'y': 0.9,
               'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top'},
        title_font=dict(size=24, family='Arial'),
        xaxis_title='Time [h]',
        yaxis_title='Conversion',
        xaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        yaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        template='plotly_white',
        margin=dict(l=60, r=60, t=60, b=60)
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ode_results['X'],
        y=[Mn * 10 ** 4 for Mn in ode_results['Mn']],
        mode='lines',
        line=dict(color='royalblue', width=2),
    ))
    fig2.update_layout(
        title={'text': 'Number-average molecular weight x Conversion',
               'y': 0.9,
               'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top'},
        title_font=dict(size=24, family='Arial'),
        xaxis_title='Conversion',
        yaxis_title='Mn*10^-4 [g/mol]',
        xaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        yaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        template='plotly_white',
        margin=dict(l=60, r=60, t=60, b=60)
    )

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=ode_results['X'], y=ode_results['PDI'], mode='lines'))
    fig3.update_layout(
        title={'text': 'Polydispersity Index (PDI) x Conversion',
               'y': 0.9,
               'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top'},
        title_font=dict(size=24, family='Arial'),
        xaxis_title='Conversion',
        yaxis_title='Polydispersity Index (PDI)',
        xaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        yaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        template='plotly_white',
        margin=dict(l=60, r=60, t=60, b=60)
    )

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=ode_results['X'],
        y=[C * 10 ** 6 for C in ode_results['C']],
        mode='lines',
        line=dict(color='royalblue', width=2),
    ))
    fig4.update_layout(
        title={'text': 'Active catalyst concentration x Conversion',
               'y': 0.9,
               'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top'},
        title_font=dict(size=24, family='Arial'),
        xaxis_title='Conversion',
        yaxis_title='C*10^6 [mol.L−1]',
        xaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        yaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        template='plotly_white',
        margin=dict(l=60, r=60, t=60, b=60)
    )

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=ode_results['X'],
        y=[CX * 10 ** 5 for CX in ode_results['CX']],
        mode='lines',
        line=dict(color='royalblue', width=2),
    ))
    fig5.update_layout(
        title={'text': 'Inactive catalyst concentration x Conversion',
               'y': 0.9,
               'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top'},
        title_font=dict(size=24, family='Arial'),
        xaxis_title='Conversion',
        yaxis_title='CX*10^5 [mol.L−1]',
        xaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        yaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        template='plotly_white',
        margin=dict(l=60, r=60, t=60, b=60)
    )

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=ode_results['X'],
        y=[a * 10 ** 4 for a in ode_results['A']],
        mode='lines',
        line=dict(color='royalblue', width=2),
    ))
    fig6.update_layout(
        title={'text': 'Reducing agent in reduced form x Conversion',
               'y': 0.9,
               'x': 0.5,
               'xanchor': 'center',
               'yanchor': 'top'},
        title_font=dict(size=24, family='Arial'),
        xaxis_title='Conversion',
        yaxis_title='A*10^4 [mol.L−1]',
        xaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        yaxis=dict(
            title_font=dict(size=18, family='Arial'),
            tickfont=dict(size=14, family='Arial'),
        ),
        template='plotly_white',
        margin=dict(l=60, r=60, t=60, b=60)
    )

    final_X = ode_results['X'].iloc[-1]
    final_PDI = ode_results['PDI'].iloc[-1]
    final_Mn = ode_results['Mn'].iloc[-1]

    return fig1, fig2, fig3, fig4, fig5, fig6, final_X, final_PDI, final_Mn
