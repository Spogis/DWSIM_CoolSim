# remove the following two lines to run on linux
import pythoncom
pythoncom.CoInitialize()

import clr
import os

from System.IO import Directory, Path, File
from System import String, Environment

from apps.print_flowsheet import *

dwsimpath = "C:\\Users\\nicol\\AppData\\Local\\DWSIM\\"


clr.AddReference(dwsimpath + "CapeOpen.dll")
clr.AddReference(dwsimpath + "DWSIM.Automation.dll")
clr.AddReference(dwsimpath + "DWSIM.Interfaces.dll")
clr.AddReference(dwsimpath + "DWSIM.GlobalSettings.dll")
clr.AddReference(dwsimpath + "DWSIM.SharedClasses.dll")
clr.AddReference(dwsimpath + "DWSIM.Thermodynamics.dll")
clr.AddReference(dwsimpath + "DWSIM.UnitOperations.dll")
clr.AddReference(dwsimpath + "DWSIM.Inspector.dll")
clr.AddReference(dwsimpath + "System.Buffers.dll")
clr.AddReference(dwsimpath + "DWSIM.Thermodynamics.ThermoC.dll")

from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType
from DWSIM.Thermodynamics import Streams, PropertyPackages
from DWSIM.UnitOperations import UnitOperations
from DWSIM.Automation import Automation3
from DWSIM.GlobalSettings import Settings
from System import Array


def run_DWSIM(evaporator_temperature, condenser_temperature, adiabatic_efficiency, picture='none'):
    diretorio_atual = os.getcwd()
    Directory.SetCurrentDirectory(diretorio_atual)

    #Automation manager
    manager = Automation3()

    FlowsheetFile = "DWSIM/DWSIM_FILE.dwxmz"
    myflowsheet = manager.LoadFlowsheet(FlowsheetFile)

    # Set Evaporator Temperature / Compressor Inlet Temperature
    obj = myflowsheet.GetFlowsheetSimulationObject('Entrada do Compressor')
    feed = obj.GetAsObject()
    feed.SetTemperature(evaporator_temperature + 273.15) #Kelvin
    feed.Calculate()

    # Set Expansion Valve Outlet Pressure
    Pressure = feed.GetPressure()
    obj = myflowsheet.GetFlowsheetSimulationObject('VALVE-1')
    valve = obj.GetAsObject()
    valve.OutletPressure = Pressure

    # Set Condenser Temperature
    obj = myflowsheet.GetFlowsheetSimulationObject('Auxiliar')
    aux_feed = obj.GetAsObject()
    temperature_aux = condenser_temperature
    aux_feed.SetTemperature(temperature_aux + 273.15) #Kelvin
    aux_feed.Calculate()

    # Set Compressor Outlet Pressure and Adiabatic Efficiency
    obj = myflowsheet.GetFlowsheetSimulationObject('Compressor')
    compressor = obj.GetAsObject()
    #compressor.SetCalculationMode(0)
    compressor.AdiabaticEfficiency = adiabatic_efficiency
    aux_pressure = aux_feed.GetPressure()
    compressor.POut = aux_pressure


    # request a calculation
    errors = manager.CalculateFlowsheet4(myflowsheet)
    errors = manager.CalculateFlowsheet4(myflowsheet)

    # Get Compressor Energy
    obj = myflowsheet.GetFlowsheetSimulationObject('Energia do Compressor')
    energ = obj.GetAsObject()
    energia = energ.GetEnergyFlow()

    # Get Compressor Outlet Temperature
    obj = myflowsheet.GetFlowsheetSimulationObject('Saida do Compressor')
    temp = obj.GetAsObject()
    discharge_temperature = temp.GetTemperature()

    # Get Refrigerant Mass Flow Rate
    mass_flow = feed.GetMassFlow()

    # Save FlowSheet
    # manager.SaveFlowsheet(myflowsheet, FlowsheetFile, True)

    if picture == 'Yes':
        print_flowsheet(myflowsheet)

    return energia, discharge_temperature, mass_flow

