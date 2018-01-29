import os
import numpy as np
import math

from matplotlib import pyplot as plt
import sys

# # import plotly.plotly as py
# import plotly
# import plotly.graph_objs as go
# # from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# plotly.tools.set_credentials_file(username='kevin0932', api_key='nJMfpcwX271AYMzx31YC')

import numpy as np
from sklearn import linear_model, datasets

def read_relative_poses_text(path='/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_hotel_beijing~beijing_hotel_2/demon_prediction/scale_record_DeMoN_Theia_Colmap_GT_correctionGT_correctionColmap_bak.txt'):
    scaleDATA = []
    dummy_image_pair_id = 1
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                scaleDeMoN = np.float64(elems[0])
                scaleTheia = np.float64(elems[1])
                scaleColmap = np.float64(elems[2])
                scaleGT = np.float64(elems[3])
                correctionScaleGT = np.float64(elems[4])
                correctionScaleColmap = np.float64(elems[5])
                absOrientationErrorInDeg = np.float64(elems[6])
                absOrientationError = np.float64(elems[7])
                view_overlap_ratio = np.float64(elems[8])
                GTbaselineLength = np.float64(elems[9])
                scaleDATA.append(np.array([scaleDeMoN, scaleTheia, scaleColmap, scaleGT, correctionScaleGT, correctionScaleColmap, absOrientationErrorInDeg, absOrientationError, view_overlap_ratio, GTbaselineLength]))
    return np.array(scaleDATA)

scaleArray = read_relative_poses_text('/home/kevin/anaconda_tensorflow_demon_ws/demon/datasets/traindata/SUN3D_Train_mit_w85_lounge1~wg_lounge1_1/demon_prediction/images_demon/dense/1/scale_record_DeMoN_Theia_Colmap_GT_correctionGT_correctionColmap.txt')
print("scaleArray.shape = ", scaleArray.shape)
print(np.isfinite(scaleArray))
print(np.any(np.isnan(scaleArray)))
print(np.any(np.isfinite(scaleArray)))
scaleArray = scaleArray[np.isfinite(scaleArray[:,0]),:]
print("scaleArray.shape = ", scaleArray.shape)
scaleArray = scaleArray[np.isfinite(scaleArray[:,1]),:]
print("scaleArray.shape = ", scaleArray.shape)
scaleArray = scaleArray[np.isfinite(scaleArray[:,2]),:]
print("scaleArray.shape = ", scaleArray.shape)
scaleArray = scaleArray[np.isfinite(scaleArray[:,3]),:]
print("scaleArray.shape = ", scaleArray.shape)
scaleArray = scaleArray[np.isfinite(scaleArray[:,4]),:]
print("scaleArray.shape = ", scaleArray.shape)
scaleArray = scaleArray[np.isfinite(scaleArray[:,5]),:]
print("scaleArray.shape = ", scaleArray.shape)
# based on scaleGT and correctionScaleGT
scaleCompared = scaleArray[:,3:5]
# # based on scaleColmap and correctionScaleColmap
# scaleCompared = np.vstack((scaleArray[:,2],scaleArray[:,5])).T
X = scaleCompared[:,0]
if X.ndim ==1:
    X = np.reshape(X,[X.shape[0],1])
y = scaleCompared[:,1]

# Fit line using all data
model = linear_model.LinearRegression()
model.fit(X, y)

# Robustly fit linear model with RANSAC algorithm
model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
model_ransac.fit(X, y)
inlier_mask = model_ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(0, 1.2*np.max(X))
line_y = model.predict(line_X[:, np.newaxis])
line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])


print("Estimated coefficients (normal, RANSAC):")
print(model.coef_, model_ransac.estimator_.coef_)

inliers = X[inlier_mask]
print("inliers.shape = ", inliers.shape)

lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.', label='Outliers')
plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw, label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()

print("inlier_mask.shape = ", inlier_mask.shape)
scaleArray = np.concatenate((scaleArray,np.reshape(inlier_mask, [inlier_mask.shape[0],1])), axis=1)
print("scaleArray.shape = ", scaleArray.shape)


plt.scatter(scaleArray[inlier_mask,8], scaleArray[inlier_mask,9], color='yellowgreen', marker='.', label='Inliers')
plt.scatter(scaleArray[outlier_mask,8], scaleArray[outlier_mask,9], color='gold', marker='.', label='Outliers')
plt.legend(loc='lower right')
plt.xlabel("view_overlap_ratio")
plt.ylabel("GTBaseline")
plt.show()

# print(np.arange(np.sum(inlier_mask)).shape)
plt.scatter(np.arange(scaleArray.shape[0])[inlier_mask], scaleArray[inlier_mask,6], color='yellowgreen', marker='.', label='Inliers')
plt.scatter(np.arange(scaleArray.shape[0])[outlier_mask], scaleArray[outlier_mask,6], color='gold', marker='.', label='Outliers')
plt.legend(loc='lower right')
plt.ylabel("Absolute Orientation Error in Degrees")
plt.show()

# def data_to_plotly(x):
#     k = []
#
#     for i in range(0, len(x)):
#         k.append(x[i][0])
#
#     return k
#
#
# lw = 2
#
# p1 = go.Scatter(x=data_to_plotly(X[inlier_mask]), y=y[inlier_mask],
#                 mode='markers',
#                 marker=dict(color='yellowgreen', size=6),
#                 name='Inliers')
# p2 = go.Scatter(x=data_to_plotly(X[outlier_mask]), y=y[outlier_mask],
#                 mode='markers',
#                 marker=dict(color='gold', size=6),
#                 name='Outliers')
#
# p3 = go.Scatter(x=line_X, y=line_y,
#                 mode='lines',
#                 line=dict(color='navy', width=lw,),
#                 name='Linear regressor')
# p4 = go.Scatter(x=line_X, y=line_y_ransac,
#                 mode='lines',
#                 line=dict(color='cornflowerblue', width=lw),
#                 name='RANSAC regressor')
# data = [p1, p2, p3, p4]
# # layout = go.Layout(xaxis=dict(zeroline=False, showgrid=False), yaxis=dict(zeroline=False, showgrid=False))
# layout = go.Layout(xaxis=dict(showgrid=True), yaxis=dict(showgrid=True, scaleanchor="x", scaleratio=1))
#
# fig = go.Figure(data=data, layout=layout)
#
#
# plotly.plotly.iplot(fig)
# # iplot(fig)
