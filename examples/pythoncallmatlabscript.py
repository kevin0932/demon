import numpy as np
import time
import matlab
import matlab.engine

pointcloud = np.random.randint(500, size=(49152, 3))

data = pointcloud
print('pass begin')
st = time.time()
data_matlab = matlab.double(data.tolist())
print ('pass numpy to matlab finished in {:.2f} sec'.format(time.time() - st))

st = time.time()
eng = matlab.engine.start_matlab()
#matlab_normals = eng.findPointNormals(data_matlab,9.)
matlab_normals = eng.findPointNormals(data_matlab,49.)
print ('call matlab script to estimate normals and it is finished in {:.2f} sec'.format(time.time() - st))

#print(matlab_normals)
print(type(matlab_normals))
print((matlab_normals.size))

np_normals = np.array(matlab_normals._data.tolist())
np_normals = np_normals.reshape(matlab_normals.size)
print(type(np_normals))
print((np_normals.dtype))
print(np_normals.shape)
print(np_normals[0,:])