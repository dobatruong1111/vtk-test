import vtk

class PanInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self) -> None:
        super().__init__()
        self.isPan = False
        self.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self.leftButtonPressEventHandle)
        self.AddObserver(vtk.vtkCommand.MouseMoveEvent, self.mouseMoveEventHandle)
        self.AddObserver(vtk.vtkCommand.LeftButtonReleaseEvent, self.leftButtonReleaseEventHandle)

    def leftButtonPressEventHandle(self, obj: vtk.vtkInteractorStyleTrackballCamera, event: str) -> None:
        self.StartPan()
        if not self.isPan:
            self.isPan = True
        self.OnLeftButtonDown()

    def mouseMoveEventHandle(self, obj: vtk.vtkInteractorStyleTrackballCamera, event: str) -> None:
        if self.isPan:
            self.Pan()
        else:
            self.OnMouseMove()

    def leftButtonReleaseEventHandle(self, obj: vtk.vtkInteractorStyleTrackballCamera, event: str) -> None:
        if self.isPan:
            self.isPan = False
        self.EndPan()
        self.OnLeftButtonUp()

def main(path: str) -> None:
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(1000, 500)
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    style = PanInteractorStyle()
    renderWindowInteractor.SetInteractorStyle(style)
    renderWindow.SetInteractor(renderWindowInteractor)

    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(path)
    reader.Update()
    imageData = reader.GetOutput()

    mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
    mapper.SetInputData(imageData)
    mapper.AutoAdjustSampleDistancesOff()
    mapper.LockSampleDistanceToInputSpacingOn()

    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()

    volumeProperty.SetAmbient(0.1)
    volumeProperty.SetDiffuse(0.9)
    volumeProperty.SetSpecular(0.2)
    volumeProperty.SetSpecularPower(10)

    scalarColorTransferFunction = vtk.vtkColorTransferFunction()
    scalarColorTransferFunction.AddRGBPoint(-3024, 0, 0, 0)
    scalarColorTransferFunction.AddRGBPoint(143.556, 0.615686, 0.356863, 0.184314)
    scalarColorTransferFunction.AddRGBPoint(166.222, 0.882353, 0.603922, 0.290196)
    scalarColorTransferFunction.AddRGBPoint(214.389, 1, 1, 1)
    scalarColorTransferFunction.AddRGBPoint(419.736, 1, 0.937033, 0.954531)
    scalarColorTransferFunction.AddRGBPoint(3071, 0.827451, 0.658824, 1)
    volumeProperty.SetColor(scalarColorTransferFunction)

    scalarOpacity = vtk.vtkPiecewiseFunction()
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
    path = "./data/220277460 Nguyen Thanh Dat"
    main(path)
