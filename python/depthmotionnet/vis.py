#
#  DeMoN - Depth Motion Network
#  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import pyximport; pyximport.install()
import numpy as np
from .helpers import angleaxis_to_rotation_matrix


def compute_point_cloud_from_depthmap( depth, K, R, t, normals=None, colors=None ):
    """Creates a point cloud numpy array and optional normals and colors arrays

    depth: numpy.ndarray
        2d array with depth values

    K: numpy.ndarray
        3x3 matrix with internal camera parameters

    R: numpy.ndarray
        3x3 rotation matrix

    t: numpy.ndarray
        3d translation vector

    normals: numpy.ndarray
        optional array with normal vectors

    colors: numpy.ndarray
        optional RGB image with the same dimensions as the depth map.
        The shape is (3,h,w) with type uint8

    """
    from .vis_cython import compute_point_cloud_from_depthmap as _compute_point_cloud_from_depthmap
    return _compute_point_cloud_from_depthmap(depth, K, R, t, normals, colors)

def compute_point_cloud_from_depthmap_scaled( depth, K, R, t, normals=None, colors=None, scale=1 ):
    """Creates a point cloud numpy array and optional normals and colors arrays

    depth: numpy.ndarray
        2d array with depth values

    K: numpy.ndarray
        3x3 matrix with internal camera parameters

    R: numpy.ndarray
        3x3 rotation matrix

    t: numpy.ndarray
        3d translation vector

    normals: numpy.ndarray
        optional array with normal vectors

    colors: numpy.ndarray
        optional RGB image with the same dimensions as the depth map.
        The shape is (3,h,w) with type uint8

    """
    from .vis_cython import compute_point_cloud_from_depthmap_scaled as _compute_point_cloud_from_depthmap_scaled
    return _compute_point_cloud_from_depthmap_scaled(depth, K, R, t, normals, colors, scale)


def create_camera_polydata(R, t, only_polys=False):
    """Creates a vtkPolyData object with a camera mesh"""
    import vtk
    cam_points = np.array([
        [0, 0, 0],
        [-1,-1, 1.5],
        [ 1,-1, 1.5],
        [ 1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0,1.2,1.5],
        [ 1,-0.5,1.5],
        [ 1, 0.5,1.5],
        [ 1.2, 0, 1.5]]
    )
    cam_points = (0.25*cam_points - t).dot(R)

    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(cam_points.shape[0])
    Colors = vtk.vtkUnsignedCharArray()#
    Colors.SetNumberOfComponents(3)#
    Colors.SetName("Colors")#
    Colors.SetNumberOfTuples(cam_points.shape[0])#
    for i in range(cam_points.shape[0]):
        vpoints.SetPoint(i, cam_points[i])
        # Colors.InsertNextTuple3(255,255,255)#
        Colors.SetTuple3(i , 255, 255, 255)
    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)
    vpoly.GetPointData().SetScalars(Colors)#

    poly_cells = vtk.vtkCellArray()

    if not only_polys:
        line_cells = vtk.vtkCellArray()

        line_cells.InsertNextCell( 5 );
        line_cells.InsertCellPoint( 1 );
        line_cells.InsertCellPoint( 2 );
        line_cells.InsertCellPoint( 3 );
        line_cells.InsertCellPoint( 4 );
        line_cells.InsertCellPoint( 1 );

        line_cells.InsertNextCell( 3 );
        line_cells.InsertCellPoint( 1 );
        line_cells.InsertCellPoint( 0 );
        line_cells.InsertCellPoint( 2 );

        line_cells.InsertNextCell( 3 );
        line_cells.InsertCellPoint( 3 );
        line_cells.InsertCellPoint( 0 );
        line_cells.InsertCellPoint( 4 );

        # x-axis indicator
        line_cells.InsertNextCell( 3 );
        line_cells.InsertCellPoint( 8 );
        line_cells.InsertCellPoint( 10 );
        line_cells.InsertCellPoint( 9 );
        vpoly.SetLines(line_cells)
    else:
        # left
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 1 );
        poly_cells.InsertCellPoint( 4 );

        # right
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 3 );
        poly_cells.InsertCellPoint( 2 );

        # top
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 4 );
        poly_cells.InsertCellPoint( 3 );

        # bottom
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 2 );
        poly_cells.InsertCellPoint( 1 );

        # x-axis indicator
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 8 );
        poly_cells.InsertCellPoint( 10 );
        poly_cells.InsertCellPoint( 9 );

    # up vector (y-axis)
    poly_cells.InsertNextCell( 3 );
    poly_cells.InsertCellPoint( 5 );
    poly_cells.InsertCellPoint( 6 );
    poly_cells.InsertCellPoint( 7 );

    vpoly.SetPolys(poly_cells)

    return vpoly


def create_camera_actor(R, t):
    """Creates a vtkActor object with a camera mesh"""
    import vtk
    vpoly = create_camera_polydata(R, t)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vpoly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()
    actor.GetProperty().SetLineWidth(2)

    return actor


def create_pointcloud_polydata(points, colors=None):
    """Creates a vtkPolyData object with the point cloud from numpy arrays

    points: numpy.ndarray
        pointcloud with shape (n,3)

    colors: numpy.ndarray
        uint8 array with colors for each point. shape is (n,3)

    Returns vtkPolyData object
    """
    import vtk
    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i, points[i])
    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)

    if not colors is None:
        vcolors = vtk.vtkUnsignedCharArray()
        vcolors.SetNumberOfComponents(3)
        vcolors.SetName("Colors")
        vcolors.SetNumberOfTuples(points.shape[0])
        for i in range(points.shape[0]):
            vcolors.SetTuple3(i ,colors[i,0],colors[i,1], colors[i,2])
        vpoly.GetPointData().SetScalars(vcolors)

    vcells = vtk.vtkCellArray()

    for i in range(points.shape[0]):
        vcells.InsertNextCell(1)
        vcells.InsertCellPoint(i)

    vpoly.SetVerts(vcells)

    return vpoly



def create_pointcloud_actor(points, colors=None):
    """Creates a vtkActor with the point cloud from numpy arrays

    points: numpy.ndarray
        pointcloud with shape (n,3)

    colors: numpy.ndarray
        uint8 array with colors for each point. shape is (n,3)

    Returns vtkActor object
    """
    import vtk
    vpoly = create_pointcloud_polydata(points, colors)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vpoly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(3)

    return actor


# def visualize_prediction( inverse_depth, intrinsics=None, normals=None, rotation=None, translation=None, image=None ):
def visualize_prediction( inverse_depth, intrinsics=None, normals=None, R1=None, t1=None, rotation=None, translation=None, image=None, scale=1 ):
    """Visualizes the network predictions

    inverse_depth: numpy.ndarray
        2d array with the inverse depth values with shape (h,w)

    intrinsics: numpy.ndarray
        4 element vector with the normalized intrinsic parameters with shape
        (4,)

    normals: numpy.ndarray
        normal map with shape (3,h,w)

    rotation: numpy.ndarray
        rotation in axis angle format with 3 elements with shape (3,)

    translation: numpy.ndarray
        translation vector with shape (3,)

    image: numpy.ndarray
        Image with shape (3,h,w) in the range [-0.5,0.5].
    """
    import vtk
    depth = (1/inverse_depth).squeeze()

    w = depth.shape[-1]
    h = depth.shape[-2]
    # w = depth.shape[-1]*12
    # h = depth.shape[-2]*12

    if intrinsics is None:
        intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]) # sun3d intrinsics

    K = np.eye(3)
    K[0,0] = intrinsics[0]*w
    K[1,1] = intrinsics[1]*h
    K[0,2] = intrinsics[2]*w
    K[1,2] = intrinsics[3]*h
    # K[0,0] = 0.89115971*3072
    # K[1,1] = 1.18821287*2304
    # K[0,2] = 3072/2
    # K[1,2] = 2304/2
    # K[0,0] = 0.89115971*256
    # K[1,1] = 1.18821287*192
    # K[0,2] = 256/2
    # K[1,2] = 192/2

    if not R1 is None and not t1 is None:
        R1 = R1
        t1 = t1
    else:
        R1 = np.eye(3)
        t1 = np.zeros((3,))

    # R2, t2 are not used in my test
    if not rotation is None and not translation is None:
        R2 = angleaxis_to_rotation_matrix(rotation.squeeze())
        t2 = translation.squeeze()
    else:
        R2 = np.eye(3)
        t2 = np.zeros((3,))

    if not normals is None:
        n = normals.squeeze()
    else:
        n = None

    if not image is None:
        img = ((image+0.5)*255).astype(np.uint8)
    else:
        img = None

    # pointcloud = compute_point_cloud_from_depthmap(depth, K, R1, t1, n, img)
    pointcloud = compute_point_cloud_from_depthmap_scaled(depth, K, R1, t1, n, img, scale)
    # pointcloud = compute_point_cloud_from_depthmap_scaled(depth, K, R1, t1, n, img, 1/scale)
    # pointcloud = compute_point_cloud_from_depthmap_scaled(depth, K, R1, t1, n, img, 1)
    # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    # if normals!=None:
    #     print("pointcloud['normals'].shape = ", pointcloud['normals'].shape)

    # create a mat for later filtering purpose
    if normals!=None:
        tmpDataMat = np.concatenate((pointcloud['points'],pointcloud['colors'],pointcloud['normals']), axis=1)
    else:
        tmpDataMat = np.concatenate((pointcloud['points'],pointcloud['colors']), axis=1)

    # # filter the points by depth, for visualization purposes!
    # Zthres = 60
    # tmpDataMat = tmpDataMat[tmpDataMat[:,2]<Zthres]
    # # print("point cloud is filtered beyond z = ", Zthres)
    # pointcloud['points'] = tmpDataMat[:,0:3]
    # pointcloud['colors'] = tmpDataMat[:,3:6]
    # if normals!=None:
    #     pointcloud['normals'] = tmpDataMat[:,6:9]
    # # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    # # if normals!=None:
    # #     print("pointcloud['normals'].shape = ", pointcloud['normals'].shape)

    # filter the points by RGB color, for visualization purposes (here just remove white points assuming that those are from sky)!
    #tmpDataMat = tmpDataMat[tmpDataMat[:,3]>254 & tmpDataMat[:,4]>254 & tmpDataMat[:,5]>254]
    # mask = (tmpDataMat[:,3]<255) | (tmpDataMat[:,4]<255) |(tmpDataMat[:,5]<255)
    mask = (tmpDataMat[:,3]<254) | (tmpDataMat[:,4]<254) |(tmpDataMat[:,5]<254)
    tmpDataMat = tmpDataMat[mask,:]
    # print("point cloud is filtered by its white color")
    pointcloud['points'] = tmpDataMat[:,0:3]
    pointcloud['colors'] = tmpDataMat[:,3:6]
    if normals!=None:
        pointcloud['normals'] = tmpDataMat[:,6:9]
    # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    # if normals!=None:
    #     print("pointcloud['normals'].shape = ", pointcloud['normals'].shape)

    # # further filter the outlier points by depth, for visualization purposes!
    # #tmpDataMat = tmpDataMat[tmpDataMat[:,3]>254 & tmpDataMat[:,4]>254 & tmpDataMat[:,5]>254]
    # mask = (tmpDataMat[:,2]<(np.mean(tmpDataMat[:,2])+3*np.std(tmpDataMat[:,2])))
    # tmpDataMat = tmpDataMat[mask,:]
    # pointcloud['points'] = tmpDataMat[:,0:3]
    # pointcloud['colors'] = tmpDataMat[:,3:6]
    # if normals!=None:
    #     pointcloud['normals'] = tmpDataMat[:,6:9]
    # # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    # # if normals!=None:
    # #     print("pointcloud['normals'].shape = ", pointcloud['normals'].shape)

    return pointcloud

    # renderer = vtk.vtkRenderer()
    # renderer.SetBackground(0, 0, 0)
    #
    # pointcloud_actor = create_pointcloud_actor(
    #     points=pointcloud['points'],
    #     colors=pointcloud['colors'] if 'colors' in pointcloud else None,
    #     )
    # renderer.AddActor(pointcloud_actor)
    #
    # cam1_actor = create_camera_actor(R1,t1)
    # renderer.AddActor(cam1_actor)
    #
    # cam2_actor = create_camera_actor(R2,t2)
    # renderer.AddActor(cam2_actor)
    #
    # axes = vtk.vtkAxesActor()
    # axes.GetXAxisCaptionActor2D().SetHeight(0.05)
    # axes.GetYAxisCaptionActor2D().SetHeight(0.05)
    # axes.GetZAxisCaptionActor2D().SetHeight(0.05)
    # axes.SetCylinderRadius(0.03)
    # axes.SetShaftTypeToCylinder()
    # renderer.AddActor(axes)
    #
    # renwin = vtk.vtkRenderWindow()
    # renwin.SetWindowName("Point Cloud Viewer")
    # renwin.SetSize(800,600)
    # renwin.AddRenderer(renderer)
    #
    #
    # # An interactor
    # interactor = vtk.vtkRenderWindowInteractor()
    # interstyle = vtk.vtkInteractorStyleTrackballCamera()
    # interactor.SetInteractorStyle(interstyle)
    # interactor.SetRenderWindow(renwin)
    #
    # # Start
    # interactor.Initialize()
    # interactor.Start()
    #
    # return pointcloud

def generate_point_cloud_from_depth_python(depth, K, R1, t1, img, scale=1):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.
    """
    rows, cols = depth.shape

    if not R1 is None and not t1 is None:
        R1 = R1
        t1 = t1
    else:
        R1 = np.eye(3)
        t1 = np.zeros((3,))

    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    # valid = (depth > 0) & (depth < 255)
    # z = np.where(valid, depth / 256.0, np.nan)
    # Zthres = 80

    # depth = depth * scale
    # valid = (depth > 0) #& (depth < Zthres)
    # z = np.where(valid, depth, np.nan)
    # x = np.where(valid, z * (c - K[0,2]) / K[0,0], 0)
    # y = np.where(valid, z * (r - K[1,2]) / K[1,1], 0)

    index = 0
    points = np.empty((rows*cols,3))
    for y in range(rows):
        for x in range(cols):
            d = depth[y,x]
            tmp = np.zeros(3)
            if np.isfinite(d) and d > 0:
                tmp[0] = d*((x+rows) - K[0,2])/K[0,0] - t1[0]
                tmp[1] = d*((y+rows) - K[1,2])/K[1,1] - t1[1]
                tmp[2] = d - t1[2]
                points[index,0] = R1[0,0]*tmp[0] + R1[1,0]*tmp[1] + R1[2,0]*tmp[2]
                points[index,1] = R1[0,1]*tmp[0] + R1[1,1]*tmp[1] + R1[2,1]*tmp[2]
                points[index,2] = R1[0,2]*tmp[0] + R1[1,2]*tmp[1] + R1[2,2]*tmp[2]
                index += 1

    pointcloud = {}
    # pointcloud['points'] = np.dstack((x, y, z))
    pointcloud['points'] = points
    pointcloud['colors'] = img.transpose([1,2,0])
    # # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    # pointcloud['points'] = np.reshape(pointcloud['points'],[pointcloud['points'].shape[0]*pointcloud['points'].shape[1], pointcloud['points'].shape[2]])
    pointcloud['colors'] = np.reshape(pointcloud['colors'],[pointcloud['colors'].shape[0]*pointcloud['colors'].shape[1], pointcloud['colors'].shape[2]])
    # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)

    tmpDataMat = np.concatenate((pointcloud['points'],pointcloud['colors']), axis=1)

    # # # filter the points by depth, for visualization purposes!
    # Zthres = 80
    # tmpDataMat = tmpDataMat[tmpDataMat[:,2]<Zthres]
    # # # print("point cloud is filtered beyond z = ", Zthres)
    # pointcloud['points'] = tmpDataMat[:,0:3]
    # pointcloud['colors'] = tmpDataMat[:,3:6]
    # # # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # # # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    #
    # filter the points by RGB color, for visualization purposes (here just remove white points assuming that those are from sky)!
    #tmpDataMat = tmpDataMat[tmpDataMat[:,3]>254 & tmpDataMat[:,4]>254 & tmpDataMat[:,5]>254]
    mask = (tmpDataMat[:,3]<255) | (tmpDataMat[:,4]<255) |(tmpDataMat[:,5]<255)
    tmpDataMat = tmpDataMat[mask,:]
    # print("point cloud is filtered by its white color")
    pointcloud['points'] = tmpDataMat[:,0:3]
    pointcloud['colors'] = tmpDataMat[:,3:6]
    # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    #
    # # further filter the outlier points by depth, for visualization purposes!
    # #tmpDataMat = tmpDataMat[tmpDataMat[:,3]>254 & tmpDataMat[:,4]>254 & tmpDataMat[:,5]>254]
    # mask = (tmpDataMat[:,2]<(np.mean(tmpDataMat[:,2])+3*np.std(tmpDataMat[:,2])))
    # tmpDataMat = tmpDataMat[mask,:]
    # pointcloud['points'] = tmpDataMat[:,0:3]
    # pointcloud['colors'] = tmpDataMat[:,3:6]
    # # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    #
    # # # # add scale
    # # pointcloud['points'] = pointcloud['points'] * scale

    # # # put it back to the global frame
    # TheiaExtrinsics = np.eye(4)
    # TheiaExtrinsics[0:3,0:3] = R1
    # TheiaExtrinsics[0:3,3] = t1 * scale
    # #
    # points_N_4 = np.ones((pointcloud['points'].shape[0],pointcloud['points'].shape[1]+1))
    # points_N_4[:,0:3] = pointcloud['points']
    # points_global_4_N = np.dot(TheiaExtrinsics.T, points_N_4.T)
    # print("pointcloud['points'] = ", pointcloud['points'])
    # pointcloud['points'] = points_global_4_N[0:3,:].T
    # print("pointcloud['points'] = ", pointcloud['points'])
    return pointcloud

def make_pointcloud_prediction_in_global_coordinate( inverse_depth, intrinsics=None, normals=None, R1=None, t1=None, rotation=None, translation=None, image=None, scale=1 ):
    """Visualizes the network predictions

    inverse_depth: numpy.ndarray
        2d array with the inverse depth values with shape (h,w)

    intrinsics: numpy.ndarray
        4 element vector with the normalized intrinsic parameters with shape
        (4,)

    normals: numpy.ndarray
        normal map with shape (3,h,w)

    rotation: numpy.ndarray
        rotation in axis angle format with 3 elements with shape (3,)

    translation: numpy.ndarray
        translation vector with shape (3,)

    image: numpy.ndarray
        Image with shape (3,h,w) in the range [-0.5,0.5].
    """
    import vtk
    depth = (1/inverse_depth).squeeze()

    w = depth.shape[-1]
    h = depth.shape[-2]

    if intrinsics is None:
        intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]) # sun3d intrinsics

    K = np.eye(3)
    K[0,0] = intrinsics[0]*w
    K[1,1] = intrinsics[1]*h
    K[0,2] = intrinsics[2]*w
    K[1,2] = intrinsics[3]*h

    if not R1 is None and not t1 is None:
        R1 = R1
        t1 = t1
    else:
        R1 = np.eye(3)
        t1 = np.zeros((3,))

    if not rotation is None and not translation is None:
        R2 = angleaxis_to_rotation_matrix(rotation.squeeze())
        t2 = translation.squeeze()
    else:
        R2 = np.eye(3)
        t2 = np.zeros((3,))

    if not normals is None:
        n = normals.squeeze()
    else:
        n = None

    if not image is None:
        img = ((image+0.5)*255).astype(np.uint8)
    else:
        img = None

    print("img.shape = ", img.shape)

    # pointcloud = compute_point_cloud_from_depthmap(depth, K, R1, t1, n, img)
    pointcloud = generate_point_cloud_from_depth_python(depth, K, R1, t1, img, scale)
    # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    # if normals!=None:
    #     print("pointcloud['normals'].shape = ", pointcloud['normals'].shape)

    # if normals!=None:
    #     tmpDataMat = np.concatenate((pointcloud['points'],pointcloud['colors'],pointcloud['normals']), axis=1)
    # else:
    #     tmpDataMat = np.concatenate((pointcloud['points'],pointcloud['colors']), axis=1)
    # # filter the points by depth, for visualization purposes!
    # Zthres = 80
    # tmpDataMat = tmpDataMat[tmpDataMat[:,2]<Zthres]
    # # print("point cloud is filtered beyond z = ", Zthres)
    # pointcloud['points'] = tmpDataMat[:,0:3]
    # pointcloud['colors'] = tmpDataMat[:,3:6]
    # if normals!=None:
    #     pointcloud['normals'] = tmpDataMat[:,6:9]
    # # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    # # if normals!=None:
    # #     print("pointcloud['normals'].shape = ", pointcloud['normals'].shape)
    #
    # # filter the points by RGB color, for visualization purposes (here just remove white points assuming that those are from sky)!
    # #tmpDataMat = tmpDataMat[tmpDataMat[:,3]>254 & tmpDataMat[:,4]>254 & tmpDataMat[:,5]>254]
    # mask = (tmpDataMat[:,3]<255) | (tmpDataMat[:,4]<255) |(tmpDataMat[:,5]<255)
    # tmpDataMat = tmpDataMat[mask,:]
    # # print("point cloud is filtered by its white color")
    # pointcloud['points'] = tmpDataMat[:,0:3]
    # pointcloud['colors'] = tmpDataMat[:,3:6]
    # if normals!=None:
    #     pointcloud['normals'] = tmpDataMat[:,6:9]
    # # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    # # if normals!=None:
    # #     print("pointcloud['normals'].shape = ", pointcloud['normals'].shape)
    #
    # # further filter the outlier points by depth, for visualization purposes!
    # #tmpDataMat = tmpDataMat[tmpDataMat[:,3]>254 & tmpDataMat[:,4]>254 & tmpDataMat[:,5]>254]
    # mask = (tmpDataMat[:,2]<(np.mean(tmpDataMat[:,2])+3*np.std(tmpDataMat[:,2])))
    # tmpDataMat = tmpDataMat[mask,:]
    # pointcloud['points'] = tmpDataMat[:,0:3]
    # pointcloud['colors'] = tmpDataMat[:,3:6]
    # if normals!=None:
    #     pointcloud['normals'] = tmpDataMat[:,6:9]
    # # print("pointcloud['points'].shape = ", pointcloud['points'].shape)
    # # print("pointcloud['colors'].shape = ", pointcloud['colors'].shape)
    # # if normals!=None:
    # #     print("pointcloud['normals'].shape = ", pointcloud['normals'].shape)


    return pointcloud

def export_prediction_to_ply( output_prefix, inverse_depth, intrinsics=None, normals=None, rotation=None, translation=None, image=None ):
    """Exports the network predictions to ply files meant for external visualization

    inverse_depth: numpy.ndarray
        2d array with the inverse depth values with shape (h,w)

    intrinsics: numpy.ndarray
        4 element vector with the normalized intrinsic parameters with shape
        (4,)

    normals: numpy.ndarray
        normal map with shape (3,h,w)

    rotation: numpy.ndarray
        rotation in axis angle format with 3 elements with shape (3,)

    translation: numpy.ndarray
        translation vector with shape (3,)

    image: numpy.ndarray
        Image with shape (3,h,w) in the range [-0.5,0.5].
    """
    import vtk
    depth = (1/inverse_depth).squeeze()

    w = depth.shape[-1]
    h = depth.shape[-2]

    if intrinsics is None:
        intrinsics = np.array([0.89115971, 1.18821287, 0.5, 0.5]) # sun3d intrinsics

    K = np.eye(3)
    K[0,0] = intrinsics[0]*w
    K[1,1] = intrinsics[1]*h
    K[0,2] = intrinsics[2]*w
    K[1,2] = intrinsics[3]*h

    R1 = np.eye(3)
    t1 = np.zeros((3,))

    if not rotation is None and not translation is None:
        R2 = angleaxis_to_rotation_matrix(rotation.squeeze())
        t2 = translation.squeeze()
    else:
        R2 = np.eye(3)
        t2 = np.zeros((3,))

    if not normals is None:
        n = normals.squeeze()
    else:
        n = None

    if not image is None:
        img = ((image+0.5)*255).astype(np.uint8)
    else:
        img = None

    pointcloud = compute_point_cloud_from_depthmap(depth, K, R1, t1, n, img)

    pointcloud_polydata = create_pointcloud_polydata(
        points=pointcloud['points'],
        colors=pointcloud['colors'] if 'colors' in pointcloud else None,
        )
    plywriter = vtk.vtkPLYWriter()
    plywriter.SetFileName(output_prefix + 'points.ply')
    plywriter.SetInputData(pointcloud_polydata)
    plywriter.SetArrayName('Colors')
    plywriter.Write()

    cam1_polydata = create_camera_polydata(R1,t1, True)
    plywriter = vtk.vtkPLYWriter()
    plywriter.SetFileName(output_prefix + 'cam1.ply')
    plywriter.SetInputData(cam1_polydata)
    plywriter.Write()

    cam2_polydata = create_camera_polydata(R2,t2, True)
    plywriter = vtk.vtkPLYWriter()
    plywriter.SetFileName(output_prefix + 'cam2.ply')
    plywriter.SetInputData(cam2_polydata)
    plywriter.Write()



def transform_pointcloud_points(points, T):
    """Transforms the pointcloud with T

    points: numpy.ndarray
        pointcloud with shape (n,3)

    T: numpy.ndarray
        The 4x4 transformation

    Returns the transformed points
    """
    tmp = np.empty((points.shape[0],points.shape[1]+1),dtype=points.dtype)
    tmp[:,0:3] = points
    tmp[:,3] = 1
    return T.dot(tmp.transpose())[0:3].transpose()
