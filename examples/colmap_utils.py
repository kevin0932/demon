"""Functions for reading COLMAP 2.0 data"""

import numpy as np
import re
import os
from collections import namedtuple
from minieigen import Quaternion, Matrix3
from PIL import Image as PILImage
from depthmotionnet.dataset_tools.view import View

Camera = namedtuple('Camera',['model','width','height','params'])
Image = namedtuple('Image',['cam_id','name','q','t'])

def quaternion_to_rotation_matrix(q):
    """Converts quaternion to rotation matrix

    q: tuple with 4 elements

    Returns a 3x3 numpy array
    """
    q = Quaternion(*q)
    R = q.toRotationMatrix()
    return np.array([list(R.row(0)), list(R.row(1)), list(R.row(2))],dtype=np.float32)



def read_binary_file(filename):
    """Reads a binary file e.g. depth map or normal map

    filename: str
        path to the binary file

    Returns the data as numpy.ndarray
    """
    with open(filename, 'rb') as f:

        # read header
        count_ampersand = 0
        header = bytes()
        for i in range(100):
            c = f.read(1)
            header += c
            if c == b'&':
                count_ampersand += 1
            if count_ampersand == 3:
                break
        if count_ampersand != 3:
            raise RuntimeError('cannot read header')

        # parse header
        match = re.match(b'(\d+)&(\d+)&(\d+)&', header)
        if match:
            channels = int(match.group(3))
            height = int(match.group(2))
            width = int(match.group(1))
        else:
            raise RuntimeError('cannot read header')

        # read data
        data = f.read()
        return np.frombuffer(data, dtype=np.float32).reshape((channels,height,width))



def read_cameras_txt(filename):
    """Simple reader for the cameras.txt file

    filename: str
        path to the cameras.txt

    Returns a dictionary will all cameras
    """
    result = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                pass
            else:
                items = line.split(' ')
                params = [ float(x) for x in items[4:] ]
                camera = Camera(
                    model = items[1],
                    width = int(items[2]),
                    height = int(items[3]),
                    params = params,
                )
                result[int(items[0])] = camera
    return result




def read_images_txt(filename):
    """Simple reader for the images.txt file

    filename: str
        path to the cameras.txt

    Returns a dictionary will all cameras
    """
    result = {}
    with open(filename, 'r') as f:
        line = f.readline()
        while line.startswith('#'):
            line = f.readline()

        line1 = line
        line2 = f.readline()

        while line1:
            items = line1.split(' ')
            image = Image(
                cam_id = int(items[8]),
                name = items[9].strip(),
                q = tuple([float(x) for x in items[1:5]]),
                t = tuple([float(x) for x in items[5:8]])
            )

            result[int(items[0])] = image

            line1 = f.readline()
            line2 = f.readline()

    return result



def read_points3d_txt(filename):
    """Simple reader for the points3D.txt file

    filename: str
        path to the cameras.txt

    Returns a numpy array for positions and for rgb colors
    """
    with open(filename, 'r') as f:
        header1 = f.readline()
        header2 = f.readline()
        header3 = f.readline()
        match = re.search('Number of points: (\d+)', header3)
        if match is None:
            raise RuntimeError('cannot parse header')

        num_points = int(match.group(1))
        points = np.empty((num_points,3), dtype=np.float32)
        colors = np.empty((num_points,3), dtype=np.uint8)
        for i in range(num_points):
            items = f.readline().split(' ')
            points[i] = [float(x) for x in items[1:4]]
            colors[i] = [int(x) for x in items[4:7]]

    return points, colors



def create_views(cameras, images, image_dir, depth_dir):
    """Create a list of Views from colmap cameras and images

    cameras: dict of Camera tuples
        cameras read with read_cameras_txt()

    images: dict of Image tuples
        images read with read_images_txt()

    image_dir: str
        The director with the undistorted images

    Returns a list of View tuples
    """
    result = []
    for image_id, image in images.items():
        camera = cameras[image.cam_id]

        if camera.model != 'PINHOLE':
        # if camera.model != 'SIMPLE_RADIAL':
        # if camera.model != 'OPENCV':    # gerrard-hall
            raise RuntimeError('Wrong camera model "'+camera.model+'"')

        K = np.zeros((3,3),np.float64)

        # compatible with camera model of 'PINHOLE' (when you use colmap command-line interface and export text result in the 'sparse' folder under the directory of 'dense')
        K[0,0] = camera.params[0]
        K[1,1] = camera.params[1]
        K[0,2] = camera.params[2]
        K[1,2] = camera.params[3]

        # # # compatible with camera model of 'SIMPLE_RADIAL' (when you export text result from sparse folder by colmap autoreconstruction)
        # # K[0,0] = camera.params[0]
        # # K[1,1] = camera.params[0]
        # K[0,0] = camera.params[0]/1.984  # redmond   # add a scale for focal length if image_undistorter in colmap output a different image size from origimal image size
        # K[1,1] = camera.params[0]/1.984  # redmond   # add a scale for focal length if image_undistorter in colmap output a different image size from origimal image size
        # K[0,2] = camera.params[1] # pc should also be scaled in the image size was changed during image_undistortion
        # K[1,2] = camera.params[2] # pc should also be scaled in the image size was changed during image_undistortion
        # # camera.params[3] is the radial distortion parameter, so it is skipped!

        # # # compatible with camera model of 'OPENCV' (when you export text result from sparse folder by colmap autoreconstruction)
        # # K[0,0] = camera.params[0]
        # # K[1,1] = camera.params[0]
        # K[0,0] = camera.params[0]# factor added in the cameras.txt 2.808  # gerrard-hall   # add a scale for focal length if image_undistorter in colmap output a different image size from origimal image size
        # K[1,1] = camera.params[0]# factor added in the cameras.txt 1.984  # gerrard-hall   # add a scale for focal length if image_undistorter in colmap output a different image size from origimal image size
        # K[0,2] = camera.params[1]   # pc should also be scaled in the image size was changed during image_undistortion
        # K[1,2] = camera.params[2]   # pc should also be scaled in the image size was changed during image_undistortion

        K[2,2] = 1

        R = quaternion_to_rotation_matrix(image.q).astype(np.float64)
        t = np.array(image.t, dtype=np.float32).astype(np.float64)

        if image_dir:
            pilimg = PILImage.open(os.path.join(image_dir,image.name))
        else:
            pilimg = None

        if depth_dir:
            depthfilename = os.path.join(depth_dir,image.name+'.geometric.bin')
            if os.path.isfile(depthfilename):
                depth = read_binary_file(depthfilename).squeeze()
            else:
                depthfilename = os.path.join(depth_dir,image.name+'.photometric.bin')
                if os.path.isfile(depthfilename):
                    depth = read_binary_file(depthfilename).squeeze()
                else:
                    raise RuntimeError('Cannot find depth map file')
            depth_metric='camera_z'
        else:
            depth=None
            depth_metric=None

        result.append( View(R=R, t=t, K=K, image=pilimg, depth=depth, depth_metric=depth_metric) )
    return result
