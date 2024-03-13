import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp

from asset.ReactionConstants import *


def model(t, y):
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


def SolveODEs(Initial_Conditions):
    global t, y
    Hours = 40
    Total_Time = 40*3600
    t_eval = np.linspace(0, Total_Time, num=int(Total_Time*10), endpoint=True)

    sol = solve_ivp(model,
                [0, Total_Time],
                Initial_Conditions,
                method='LSODA', #Radau / LSODA
                t_eval=t_eval)
    t = np.transpose(sol.t)
    y = np.transpose(sol.y)



def MoreUsableDataset():
    global t, y

    df_time = pd.DataFrame({'Time':t})

    Temp_Variable_Names = ['R0', 'Q0', 'D0',
                       'R1', 'Q1', 'D1',
                       'R2', 'Q2', 'D2',
                       'M',
                       'P0X', 'P0',
                       'C', 'CX',
                       'A', 'AX']

    df_num_sol = pd.DataFrame(y , columns = Temp_Variable_Names)
    Results = pd.concat([df_time,df_num_sol], axis=1)

    Results['X'] = (Results['M'].iloc[0]-Results['M'])/Results['M'].iloc[0]
    Results['DPn'] = (Results['R1']+Results['Q1']+Results['D1'])/(Results['R0']+Results['Q0']+Results['D0'])
    Results['DPw'] = (Results['R2']+Results['Q2']+Results['D2'])/(Results['R1']+Results['Q1']+Results['D1'])

    Results['Mn'] = MWm * Results['DPn']
    Results['PDI'] = Results['DPw']/Results['DPn']
    Results['Time'] = Results['Time']/3600

    return Results
