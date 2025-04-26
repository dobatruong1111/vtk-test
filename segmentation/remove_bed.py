import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import math
from typing import Optional
import SimpleITK as sitk

def vtk2sitk(vtkImage: vtk.vtkImageData) -> sitk.Image:
    # Takes a VTK image, returns a SimpleITK image
    dimentions = vtkImage.GetDimensions()
    spacing = vtkImage.GetSpacing()

    shape = dimentions[::-1]
    nparray = vtk_to_numpy(vtkImage.GetPointData().GetScalars()).reshape(shape)

    image = sitk.GetImageFromArray(nparray)
    image.SetSpacing(spacing)
    return image

def getLabelmap(binaryImageData: vtk.vtkImageData, minimumSize: int) -> Optional[vtk.vtkImageData]:
    from vtkITK import vtkITKIslandMath

    islandMath = vtkITKIslandMath()
    islandMath.SetInputData(binaryImageData)
    islandMath.SetFullyConnected(False)
    islandMath.SetMinimumSize(minimumSize)
    islandMath.Update()
    islandCount = islandMath.GetNumberOfIslands()
    print(f"Island count: {islandCount}")
    if islandCount < 2:
        return
    islandImage = islandMath.GetOutput()
    # print(islandImage.GetScalarRange())
    return islandImage

def splitSegments(imageData: vtk.vtkImageData, imageThreshold=-50, minimumSize=1000, maxNumberOfSegments=1) -> Optional[vtk.vtkImageData]:
    # Create a mask using global thresholding
    scalarRange = imageData.GetScalarRange()
    thresh = vtk.vtkImageThreshold()
    thresh.SetInputData(imageData)
    thresh.ThresholdBetween(imageThreshold, scalarRange[1])
    thresh.SetInValue(1)
    thresh.SetOutValue(0)
    thresh.SetOutputScalarType(vtk.VTK_UNSIGNED_INT)
    thresh.Update()
    binaryImageData = thresh.GetOutput()
    scalarRange = binaryImageData.GetScalarRange()

    sitkImage = vtk2sitk(binaryImageData)

    ccfilter = sitk.ConnectedComponentImageFilter()
    ccfilter.SetFullyConnected(True)
    output = ccfilter.Execute(sitkImage)

    relabel = sitk.RelabelComponentImageFilter()
    relabel.SetMinimumObjectSize(minimumSize)
    output = relabel.Execute(output)

    labelcount = relabel.GetNumberOfObjects()
    print(f"Label count: {labelcount}")
    if labelcount < 2:
        return

    array = sitk.GetArrayFromImage(output)
    binaryImageData.GetPointData().SetScalars(numpy_to_vtk(array.flat))
    labelmap = binaryImageData
    
    dimensions = labelmap.GetDimensions()
    if dimensions[0] <= 0 or dimensions[1] <= 0 or dimensions[2] <= 0:
        print("Labelmap is empty, there are no label values. Running vtkImageAccumulate would cause a crash")
        return
    
    scalarRange = labelmap.GetScalarRange()
    lowLabel = math.floor(scalarRange[0])
    highLabel = math.ceil(scalarRange[1])

    # Generalized histograms
    imageAccumulate = vtk.vtkImageAccumulate()
    imageAccumulate.SetInputData(labelmap)
    imageAccumulate.IgnoreZeroOn()
    imageAccumulate.SetComponentExtent(lowLabel, highLabel, 0, 0, 0, 0)
    imageAccumulate.SetComponentOrigin(0, 0, 0)
    imageAccumulate.SetComponentSpacing(1, 1, 1)
    imageAccumulate.Update()

    labels = vtk.vtkIntArray()
    for label in range(lowLabel, highLabel + 1):
        if label == 0:
            continue
        frequency = imageAccumulate.GetOutput().GetPointData().GetScalars().GetTuple1(label - lowLabel)
        if frequency == 0:
            continue
        labels.InsertNextValue(label)

    for i in range(labels.GetNumberOfTuples()):
        if maxNumberOfSegments > 0 and i >= maxNumberOfSegments:
            break

        label = int(labels.GetTuple1(i))
        
        threshold = vtk.vtkImageThreshold()
        threshold.SetInputData(labelmap)
        threshold.ThresholdBetween(label, label)
        threshold.SetInValue(0)
        threshold.SetOutValue(1)
        threshold.Update()

        castIn = vtk.vtkImageCast()
        castIn.SetInputData(threshold.GetOutput())
        castIn.SetOutputScalarTypeToUnsignedChar()
        castIn.Update()

        return castIn.GetOutput()

def applyMask(imageData: vtk.vtkImageData, mask: vtk.vtkImageData, fillValue=-1000) -> None:
    dimensions = imageData.GetDimensions()
    shape = dimensions[::-1]
    array = vtk_to_numpy(imageData.GetPointData().GetScalars()).reshape(shape)
    maskArray = vtk_to_numpy(mask.GetPointData().GetScalars()).reshape(shape)
    
    array[maskArray > 0] = fillValue
    imageData.GetPointData().SetScalars(numpy_to_vtk(array.flat))

def showVolume(dirpath: str) -> None:
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(1000, 500)
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    style = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(style)
    renderWindow.SetInteractor(renderWindowInteractor)

    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dirpath)
    reader.Update()
    imageData = reader.GetOutput()

    mask = splitSegments(imageData)
    if mask is not None:
        applyMask(imageData, mask)

    mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
    mapper.SetInputData(imageData)
    mapper.AutoAdjustSampleDistancesOff()
    mapper.LockSampleDistanceToInputSpacingOn()

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()

    volumeProperty.SetAmbient(0.15)
    volumeProperty.SetDiffuse(0.9)
    volumeProperty.SetSpecular(0.3)
    volumeProperty.SetSpecularPower(15)

    gradientOpacity = vtk.vtkPiecewiseFunction()
    gradientOpacity.AddPoint(0, 1)
    gradientOpacity.AddPoint(255, 1)
    volumeProperty.SetGradientOpacity(gradientOpacity)

    scalarColorTransferFunction = vtk.vtkColorTransferFunction()
    # Slicer preset (CT-AAA)
    scalarColorTransferFunction.AddRGBPoint(-3024, 0, 0, 0)
    scalarColorTransferFunction.AddRGBPoint(143.556, 0.615686, 0.356863, 0.184314)
    scalarColorTransferFunction.AddRGBPoint(166.222, 0.882353, 0.603922, 0.290196)
    scalarColorTransferFunction.AddRGBPoint(214.389, 1, 1, 1)
    scalarColorTransferFunction.AddRGBPoint(419.736, 1, 0.937033, 0.954531)
    scalarColorTransferFunction.AddRGBPoint(3071, 0.827451, 0.658824, 1)
    volumeProperty.SetColor(scalarColorTransferFunction)

    scalarOpacity = vtk.vtkPiecewiseFunction()
    # Slicer preset (CT-AAA)
    scalarOpacity.AddPoint(-3024, 0)
    scalarOpacity.AddPoint(143.556, 0)
    scalarOpacity.AddPoint(166.222, 0.686275)
    scalarOpacity.AddPoint(214.389, 0.696078)
    scalarOpacity.AddPoint(419.736, 0.833333)
    scalarOpacity.AddPoint(3071, 0.803922)
    volumeProperty.SetScalarOpacity(scalarOpacity)

    gradientOpacity = vtk.vtkPiecewiseFunction()
    gradientOpacity.AddPoint(0, 1)
    gradientOpacity.AddPoint(255, 1)
    volumeProperty.SetGradientOpacity(gradientOpacity)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(volumeProperty)

    renderer.AddVolume(volume)
    renderer.ResetCamera()

    renderWindowInteractor.Start()

if __name__ == "__main__":
    dirpath = "/home/dbtruong/workingspace/dicom/220277460 Nguyen Thanh Dat"
    showVolume(dirpath)
