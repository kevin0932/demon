import os
import numpy as np
import math

from matplotlib import pyplot as plt
import sys

# import plotly.plotly as py
import plotly
import plotly.graph_objs as go
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
plotly.tools.set_credentials_file(username='kevin0932', api_key='nJMfpcwX271AYMzx31YC')

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
                scaleDATA.append(np.array([scaleDeMoN, scaleTheia, scaleColmap, scaleGT, correctionScaleGT, correctionScaleColmap]))
    return np.array(scaleDATA)

scaleArray = read_relative_poses_text()
print("scaleArray.shape = ", scaleArray.shape)

# scaleCompared = scaleArray[:,3:5]
scaleCompared = np.vstack((scaleArray[:,2],scaleArray[:,5])).T
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


def data_to_plotly(x):
    k = []

    for i in range(0, len(x)):
        k.append(x[i][0])

    return k


lw = 2

p1 = go.Scatter(x=data_to_plotly(X[inlier_mask]), y=y[inlier_mask],
                mode='markers',
                marker=dict(color='yellowgreen', size=6),
                name='Inliers')
p2 = go.Scatter(x=data_to_plotly(X[outlier_mask]), y=y[outlier_mask],
                mode='markers',
                marker=dict(color='gold', size=6),
                name='Outliers')

p3 = go.Scatter(x=line_X, y=line_y,
                mode='lines',
                line=dict(color='navy', width=lw,),
                name='Linear regressor')
p4 = go.Scatter(x=line_X, y=line_y_ransac,
                mode='lines',
                line=dict(color='cornflowerblue', width=lw),
                name='RANSAC regressor')
data = [p1, p2, p3, p4]
# layout = go.Layout(xaxis=dict(zeroline=False, showgrid=False), yaxis=dict(zeroline=False, showgrid=False))
layout = go.Layout(xaxis=dict(showgrid=True), yaxis=dict(showgrid=True, scaleanchor="x", scaleratio=1))

fig = go.Figure(data=data, layout=layout)


plotly.plotly.iplot(fig)
# iplot(fig)

inliers = X[inlier_mask]
print("inliers.shape = ", inliers.shape)
