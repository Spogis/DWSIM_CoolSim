# save the pfd to an image and display it
import clr
import os

dwsimpath = "C:\\Users\\nicol\\AppData\\Local\\DWSIM\\"

clr.AddReference(dwsimpath + "SkiaSharp.dll")
clr.AddReference("System.Drawing")


def print_flowsheet(myflowsheet):
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

    imgPath = "assets/pfd.png"
    image.Save(imgPath, ImageFormat.Png)
    str.Dispose()
    canvas.Dispose()
    bmp.Dispose()

    from PIL import Image
    # im = Image.open(imgPath)
    # im.show()