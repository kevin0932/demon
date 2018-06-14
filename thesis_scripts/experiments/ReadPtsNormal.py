import Rhino
import scriptcontext
import rhinoscriptsyntax as rs
import System

def Import_npts():
    filter = "PointNormal (*.npts)|*.npts|All Files (*.*)|*.*||"
    filename = rs.OpenFileName("Open points with normals file", filter)
    if not filename: return

    file = open(filename, "r")
    contents = file.readlines()
    file.close()

    cloud = Rhino.Geometry.PointCloud()
    npoint = Rhino.Geometry.Point3d.Unset
    normal = Rhino.Geometry.Vector3d.Unset

    rs.StatusBarProgressMeterShow(" Processing", 0, len(contents), True, True)
    rs.Prompt("Processing, please wait")

    for i, line in enumerate(contents):
        items = line.strip("()\n").split(" ")
        if not items[0].startswith("nan"):
            npoint.X = float(items[0])
            npoint.Y = float(items[1])
            npoint.Z = float(items[2])
            normal.X = float(items[3])
            normal.Y = float(items[4])
            normal.Z = float(items[5])
            cloud.Add(npoint, normal)
            if (i%50000) == 0: rs.StatusBarProgressMeterUpdate(i)

    rs.Prompt()
    rs.StatusBarProgressMeterHide()

    if cloud.IsValid:
        scriptcontext.doc.Objects.AddPointCloud(cloud)
        scriptcontext.doc.Views.Redraw()
        print("One poincloud with normals build from", file.name)

    cloud.Dispose()

if (__name__ == "__main__"):
    Import_npts()
