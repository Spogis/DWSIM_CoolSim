# add compounds

cnames = ["Water", "Ethanol","Acetone"]

myflowsheet.AddCompound("Water")
myflowsheet.AddCompound("Ethanol")
myflowsheet.AddCompound("Acetone")

# create and connect objects

feed  = myflowsheet.AddFlowsheetObject("Material Stream", "Feed")
dist = myflowsheet.AddFlowsheetObject("Material Stream", "Distillate")
bottoms = myflowsheet.AddFlowsheetObject("Material Stream", "Bottoms")
column = myflowsheet.AddFlowsheetObject("Distillation Column", "Column")

feed = feed.GetAsObject()
dist = dist.GetAsObject()
bottoms = bottoms.GetAsObject()
column = column.GetAsObject()

# change number of stages - default is 10

column.SetNumberOfStages(12)

# connect streams to column

column.ConnectFeed(feed, 6)
column.ConnectDistillate(dist)
column.ConnectBottoms(bottoms)

myflowsheet.NaturalLayout()

feed.SetOverallComposition(Array[float]([0.4, 0.4, 0.2]))
feed.SetTemperature(350.0) # K
feed.SetPressure(101325.0) # Pa
feed.SetMolarFlow(300.0) # mol/s

# allowed specs:

# Heat_Duty = 0
# Product_Molar_Flow_Rate = 1
# Component_Molar_Flow_Rate = 2
# Product_Mass_Flow_Rate = 3
# Component_Mass_Flow_Rate = 4
# Component_Fraction = 5
# Component_Recovery = 6
# Stream_Ratio = 7
# Temperature = 8

column.SetCondenserSpec("Reflux Ratio", 3.0, "")
column.SetReboilerSpec("Product_Molar_Flow_Rate", 200.0, "mol/s")

# property package

nrtl = myflowsheet.CreateAndAddPropertyPackage("NRTL")

# request a calculation

errors = manager.CalculateFlowsheet4(myflowsheet)

# get condenser and reboiler duties

cduty = column.CondenserDuty
rduty = column.ReboilerDuty

print("Condenser Duty: " + str(cduty) + " kW")
print("Reboiler Duty: " + str(rduty) + " kW")

dtemp = dist.GetTemperature()
dflow = dist.GetMolarFlow()
btemp = bottoms.GetTemperature()
bflow = bottoms.GetMolarFlow()

print()
print("Distillate Temperature: " + str(dtemp) + " K")
print("Bottoms Temperature: " + str(btemp) + " K")

print()
print("Distillate Molar Flow: " + str(dflow) + " mol/s")
print("Bottoms Molar Flow: " + str(bflow) + " mol/s")

# product compositions

print()

distcomp = dist.GetOverallComposition()
print("Distillate Molar Composition:")
for i in range(0, 3):
	print(cnames[i] + ": " + str(distcomp[i]))
	i+=1

print()

bcomp = bottoms.GetOverallComposition()
print("Bottoms Molar Composition:")
for i in range(0, 3):
	print(cnames[i] + ": " + str(bcomp[i]))
	i+=1

# save file

fileNameToSave = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "python_column_sample.dwxmz")

manager.SaveFlowsheet(myflowsheet, fileNameToSave, True)

# save the pfd to an image and display it

clr.AddReference(dwsimpath + "SkiaSharp.dll")
clr.AddReference("System.Drawing")

from SkiaSharp import SKBitmap, SKImage, SKCanvas, SKEncodedImageFormat
from System.IO import MemoryStream
from System.Drawing import Image
from System.Drawing.Imaging import ImageFormat

PFDSurface = myflowsheet.GetSurface()

imgwidth = 1024
imgheight = 768

bmp = SKBitmap(imgwidth, imgheight)
canvas = SKCanvas(bmp)
PFDSurface.Center(imgwidth, imgheight)
PFDSurface.ZoomAll(imgwidth, imgheight)
PFDSurface.UpdateCanvas(canvas)
d = SKImage.FromBitmap(bmp).Encode(SKEncodedImageFormat.Png, 100)
str = MemoryStream()
d.SaveTo(str)
image = Image.FromStream(str)
imgPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "pfd.png")
image.Save(imgPath, ImageFormat.Png)
str.Dispose()
canvas.Dispose()
bmp.Dispose()

from PIL import Image

im = Image.open(imgPath)
im.show()