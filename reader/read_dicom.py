import vtk

STANDARD = [
    {
        "name": 'air',
        "range": [-1000],
        "color": [[0, 0, 0]] # black
    },
    {
        "name": 'lung',
        "range": [-600, -400],
        "color": [[194 / 255, 105 / 255, 82 / 255]]
    },
    {
        "name": 'fat',
        "range": [-100, -60],
        "color": [[194 / 255, 166 / 255, 115 / 255]]
    },
    {
        "name": 'soft tissue',
        "range": [40, 80],
        "color": [[102 / 255, 0, 0], [153 / 255, 0, 0]] # red
    },
    {
        "name": 'bone',
        "range": [400, 1000],
        "color": [[255 / 255, 217 / 255, 163 / 255]] # ~ white
    }
]

def to_rgb_points(colormap):
    rgb_points = []
    for item in colormap:
        crange = item["range"]
        color = item["color"]
        for idx, r in enumerate(crange):
            if len(color) == len(crange):
                rgb_points.append([r] + color[idx])
            else:
                rgb_points.append([r] + color[0])
    return rgb_points

def main() -> None:
    path = "/home/dbtruong/workingspace/dicom/220277460 Nguyen Thanh Dat"

    rgb_points = to_rgb_points(STANDARD)
    colors = vtk.vtkNamedColors()
    reader = vtk.vtkDICOMImageReader()
    mapper = vtk.vtkSmartVolumeMapper()
    volume = vtk.vtkVolume()
    volumeProperty = vtk.vtkVolumeProperty()
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    scalarOpacity = vtk.vtkPiecewiseFunction()
    color = vtk.vtkColorTransferFunction()
    renderWindowIn = vtk.vtkRenderWindowInteractor()

    # Outline
    # Description: drawing a bounding box out volume object
    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(reader.GetOutputPort())
    outlineMapper = vtk.vtkPolyDataMapper()
    outlineMapper.SetInputConnection(outline.GetOutputPort())
    outlineActor = vtk.vtkActor()
    outlineActor.SetMapper(outlineMapper)
    outlineActor.GetProperty().SetColor(0, 0, 0)
    
    reader.SetDirectoryName(path)
    reader.Update()
    imageData = reader.GetOutput() # vtkImageData
    
    # This option will use hardware accelerated rendering exclusively
    # This is a good option if you know there is hardware acceleration
    mapper.SetRequestedRenderModeToGPU()
    mapper.SetInputData(imageData)

    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()
    # Lighting of volume
    volumeProperty.SetAmbient(0.1)
    volumeProperty.SetDiffuse(0.9)
    volumeProperty.SetSpecular(0.2)
    # Color map thought a transfer function
    for rgb_point in rgb_points:
        color.AddRGBPoint(rgb_point[0], rgb_point[1], rgb_point[2], rgb_point[3])
    volumeProperty.SetColor(color)

    # Bone preset
    # scalarOpacity.AddPoint(184.129411764706, 0)
    # scalarOpacity.AddPoint(2271.070588235294, 1)
    # Angio preset
    # scalarOpacity.AddPoint(125.42352941176478, 0)
    # scalarOpacity.AddPoint(1785, 1)
    # Muscle preset
    scalarOpacity.AddPoint(-63.16470588235279, 0)
    scalarOpacity.AddPoint(559.1764705882356, 1)
    # Mip preset
    # scalarOpacity.AddPoint(-1661.5882352941176, 0)
    # scalarOpacity.AddPoint(2449.5490196078435, 1)
    volumeProperty.SetScalarOpacity(scalarOpacity)

    volume.SetMapper(mapper)
    volume.SetProperty(volumeProperty)

    renderer.SetBackground(colors.GetColor3d("White"))
    renderer.AddVolume(volume)
    renderer.AddActor(outlineActor)
    
    renderWindow.SetWindowName("3D Dicom")
    renderWindow.SetSize(500, 500)
    renderWindow.AddRenderer(renderer)
    
    renderWindowIn.SetRenderWindow(renderWindow)
    style = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowIn.SetInteractorStyle(style)

    renderWindowIn.Initialize()
    renderWindowIn.Start()

if __name__ == "__main__":
    main()
    