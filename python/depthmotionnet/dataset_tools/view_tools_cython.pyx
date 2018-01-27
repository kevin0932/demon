import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isfinite
from libc.math cimport sqrt



@cython.boundscheck(False)
cdef _compute_visible_points_mask(
        np.ndarray[np.float32_t, ndim=2] depth,
        np.ndarray[np.float32_t, ndim=2] K1,
        np.ndarray[np.float32_t, ndim=2] R1,
        np.ndarray[np.float32_t, ndim=1] t1,
        np.ndarray[np.float32_t, ndim=2] P2,
        int width2,
        int height2,
        int borderx,
        int bordery):

    cdef np.float32_t point3d[3]
    cdef np.float32_t point4d[4]
    point4d[3] = 1.0
    cdef np.float32_t point_proj[3]
    cdef int x, y
    cdef np.float32_t px, py
    cdef np.float32_t d
    cdef np.ndarray[np.float32_t,ndim=2] RT = R1.transpose()

    cdef np.ndarray[np.uint8_t,ndim=2] mask = np.zeros((depth.shape[0],depth.shape[1]), dtype=np.uint8)
    cdef np.ndarray[np.uint8_t,ndim=3] matched_pixels = np.zeros((depth.shape[0],depth.shape[1], 2), dtype=np.uint8)

    for y in range(depth.shape[0]):
        for x in range(depth.shape[1]):

            d = depth[y,x]
            if np.isfinite(d) and d > 0.0:
                px = x + 0.5
                py = y + 0.5

                point3d[0] = d*(px - K1[0,2])/K1[0,0]
                point3d[1] = d*(py - K1[1,2])/K1[1,1]
                point3d[2] = d
                point3d[0] -= t1[0]
                point3d[1] -= t1[1]
                point3d[2] -= t1[2]
                point4d[0] = RT[0,0]*point3d[0] + RT[0,1]*point3d[1] + RT[0,2]*point3d[2]
                point4d[1] = RT[1,0]*point3d[0] + RT[1,1]*point3d[1] + RT[1,2]*point3d[2]
                point4d[2] = RT[2,0]*point3d[0] + RT[2,1]*point3d[1] + RT[2,2]*point3d[2]

                point_proj[0] = P2[0,0]*point4d[0] + P2[0,1]*point4d[1] + P2[0,2]*point4d[2] + P2[0,3]*point4d[3]
                point_proj[1] = P2[1,0]*point4d[0] + P2[1,1]*point4d[1] + P2[1,2]*point4d[2] + P2[1,3]*point4d[3]
                point_proj[2] = P2[2,0]*point4d[0] + P2[2,1]*point4d[1] + P2[2,2]*point4d[2] + P2[2,3]*point4d[3]
                if point_proj[2] > 0.0:
                    point_proj[0] /= point_proj[2]
                    point_proj[1] /= point_proj[2]
                    if point_proj[0] > borderx and point_proj[1] > bordery and point_proj[0] < width2-borderx and point_proj[1] < height2-bordery:
                        mask[y,x] = 1
                        matched_pixels[y,x,0] = int(point_proj[0])
                        matched_pixels[y,x,1] = int(point_proj[1])
                else:
                    matched_pixels[y,x,0] = np.nan
                    matched_pixels[y,x,1] = np.nan
    return mask, matched_pixels



def compute_visible_points_mask( view1, view2, borderx=0, bordery=0 ):
    """Computes a mask of the pixels in view1 that are visible in view2

    view1: View namedtuple
        First view

    view2: View namedtuple
        Second view

    borderx: int
        border in x direction. Points in the border are considered invalid

    bordery: int
        border in y direction. Points in the border are considered invalid

    Returns a mask of valid points
    """
    assert view1.depth_metric == 'camera_z', "Depth metric must be 'camera_z'"

    P2 = np.empty((3,4), dtype=np.float32)
    P2[:,0:3] = view2.R
    P2[:,3:4] = view2.t.reshape((3,1))
    P2 = view2.K.dot(P2)

    return _compute_visible_points_mask(
            view1.depth,
            view1.K.astype(np.float32),
            view1.R.astype(np.float32),
            view1.t.astype(np.float32),
            P2.astype(np.float32),
            view2.depth.shape[1],
            view2.depth.shape[0],
            borderx,
            bordery)




@cython.boundscheck(False)
cdef _compute_depth_ratios(
        np.ndarray[np.float32_t, ndim=2] depth1,
        np.ndarray[np.float32_t, ndim=2] depth2,
        np.ndarray[np.float32_t, ndim=2] K1,
        np.ndarray[np.float32_t, ndim=2] R1,
        np.ndarray[np.float32_t, ndim=1] t1,
        np.ndarray[np.float32_t, ndim=2] P2 ):
    cdef np.float32_t point3d[3]
    cdef np.float32_t point4d[4]
    point4d[3] = 1.0
    cdef np.float32_t point_proj[3]
    cdef int x, y, x2, y2
    cdef np.float32_t px, py
    cdef np.float32_t d, d2
    cdef np.ndarray[np.float32_t,ndim=2] RT = R1.transpose()

    cdef np.ndarray[np.float32_t,ndim=2] result = np.full((depth1.shape[0],depth1.shape[1]), np.nan, dtype=np.float32)

    for y in range(depth1.shape[0]):
        for x in range(depth1.shape[1]):

            d = depth1[y,x]
            if np.isfinite(d) and d > 0.0:
                px = x + 0.5
                py = y + 0.5

                point3d[0] = d*(px - K1[0,2])/K1[0,0]
                point3d[1] = d*(py - K1[1,2])/K1[1,1]
                point3d[2] = d
                point3d[0] -= t1[0]
                point3d[1] -= t1[1]
                point3d[2] -= t1[2]
                point4d[0] = RT[0,0]*point3d[0] + RT[0,1]*point3d[1] + RT[0,2]*point3d[2]
                point4d[1] = RT[1,0]*point3d[0] + RT[1,1]*point3d[1] + RT[1,2]*point3d[2]
                point4d[2] = RT[2,0]*point3d[0] + RT[2,1]*point3d[1] + RT[2,2]*point3d[2]

                point_proj[0] = P2[0,0]*point4d[0] + P2[0,1]*point4d[1] + P2[0,2]*point4d[2] + P2[0,3]*point4d[3]
                point_proj[1] = P2[1,0]*point4d[0] + P2[1,1]*point4d[1] + P2[1,2]*point4d[2] + P2[1,3]*point4d[3]
                point_proj[2] = P2[2,0]*point4d[0] + P2[2,1]*point4d[1] + P2[2,2]*point4d[2] + P2[2,3]*point4d[3]
                if point_proj[2] > 0.0:
                    point_proj[0] /= point_proj[2]
                    point_proj[1] /= point_proj[2]
                    if point_proj[0] > 0 and point_proj[1] > 0 and point_proj[0] < depth2.shape[1] and point_proj[1] < depth2.shape[0]:
                        # lookup the depth value
                        x2 = max(0,min(depth2.shape[1],int(round(point_proj[0]))))
                        y2 = max(0,min(depth2.shape[0],int(round(point_proj[1]))))
                        d2 = depth2[y2,x2]
                        if d2 > 0.0 and isfinite(d2):
                            s = point_proj[2]/d2
                            result[y,x] = s

    return result




def compute_depth_ratios( view1, view2 ):
    """Projects each point defined in view1 to view2 and computes the ratio of
    the depth value of the projected point and the stored depth value in view2.


    view1: View namedtuple
        First view

    view2: View namedtuple
        Second view

    Returns the scale value for view2 relative to view1
    """
    assert view1.depth_metric == 'camera_z', "Depth metric must be 'camera_z'"
    assert view2.depth_metric == 'camera_z', "Depth metric must be 'camera_z'"

    P2 = np.empty((3,4), dtype=np.float32)
    P2[:,0:3] = view2.R
    P2[:,3:4] = view2.t.reshape((3,1))
    P2 = view2.K.dot(P2)

    return _compute_depth_ratios(
            view1.depth,
            view2.depth,
            view1.K.astype(np.float32),
            view1.R.astype(np.float32),
            view1.t.astype(np.float32),
            P2.astype(np.float32) )



@cython.boundscheck(False)
cdef _compute_flow(
        np.ndarray[np.float32_t, ndim=2] depth1,
        np.ndarray[np.float32_t, ndim=2] K1,
        np.ndarray[np.float32_t, ndim=2] R1,
        np.ndarray[np.float32_t, ndim=1] t1,
        np.ndarray[np.float32_t, ndim=2] P2 ):
    cdef np.float32_t point3d[3]
    cdef np.float32_t point4d[4]
    point4d[3] = 1.0
    cdef np.float32_t point_proj[3]
    cdef int x, y, x2, y2
    cdef np.float32_t px, py
    cdef np.float32_t d, d2
    cdef np.ndarray[np.float32_t,ndim=2] RT = R1.transpose()

    cdef np.ndarray[np.float32_t,ndim=3] result = np.full((2,depth1.shape[0],depth1.shape[1]), np.nan, dtype=np.float32)

    for y in range(depth1.shape[0]):
        for x in range(depth1.shape[1]):

            d = depth1[y,x]
            if np.isfinite(d) and d > 0.0:
                px = x + 0.5
                py = y + 0.5

                point3d[0] = d*(px - K1[0,2])/K1[0,0]
                point3d[1] = d*(py - K1[1,2])/K1[1,1]
                point3d[2] = d
                point3d[0] -= t1[0]
                point3d[1] -= t1[1]
                point3d[2] -= t1[2]
                point4d[0] = RT[0,0]*point3d[0] + RT[0,1]*point3d[1] + RT[0,2]*point3d[2]
                point4d[1] = RT[1,0]*point3d[0] + RT[1,1]*point3d[1] + RT[1,2]*point3d[2]
                point4d[2] = RT[2,0]*point3d[0] + RT[2,1]*point3d[1] + RT[2,2]*point3d[2]

                point_proj[0] = P2[0,0]*point4d[0] + P2[0,1]*point4d[1] + P2[0,2]*point4d[2] + P2[0,3]*point4d[3]
                point_proj[1] = P2[1,0]*point4d[0] + P2[1,1]*point4d[1] + P2[1,2]*point4d[2] + P2[1,3]*point4d[3]
                point_proj[2] = P2[2,0]*point4d[0] + P2[2,1]*point4d[1] + P2[2,2]*point4d[2] + P2[2,3]*point4d[3]

                point_proj[0] /= point_proj[2]
                point_proj[1] /= point_proj[2]
                result[0,y,x] = point_proj[0]-px
                result[1,y,x] = point_proj[1]-py

    return result



###################################################################################################################
@cython.boundscheck(False)
cdef _igl_pointcloud_filtering_in_multiviews(
        np.ndarray[np.float32_t, ndim=2] K1,
        np.ndarray[np.float32_t, ndim=2] R1,
        np.ndarray[np.float32_t, ndim=1] t1,
        np.ndarray[np.float32_t, ndim=2] points_from_view1_in_global_frame, # [numPts,3]
        np.ndarray[np.float32_t, ndim=1] weights1,  # [numPts]
        #np.ndarray[np.float32_t, ndim=2] colors1, # here color corresponding to points [numPts,3]
        #np.ndarray[np.float32_t, ndim=2] scaled_depth1, # [h,w]

        np.ndarray[np.float32_t, ndim=3] K2s,   # [3,3,numViews]
        np.ndarray[np.float32_t, ndim=3] R2s,   # [3,3,numViews]
        np.ndarray[np.float32_t, ndim=2] t2s,   # [3,numViews]
        np.ndarray[np.float32_t, ndim=3] scaled_depth2s,  # [h,w,numViews]
        np.ndarray[np.float32_t, ndim=3] weights2s,   # [h,w,numViews]
        np.ndarray[np.float32_t, ndim=4] colors2s,  # here color image collections [h,w,3,numViews]
        np.float32_t sigma = 0.01,
        np.float32_t tp = 0.2,
        int borderx = 0,
        int bordery = 0):

    cdef np.float32_t point3d_j[3]
    cdef np.float32_t point4d_i[4]
    cdef np.float32_t point4d_j[4]
    point4d_i[3] = 1.0
    point4d_j[3] = 1.0
    cdef np.float32_t point_proj_i[3]
    cdef int x, y
    cdef np.float32_t px, py
    cdef np.float32_t d
    cdef np.float32_t matched_zj_3d_pt_in_view2, zi_3d_pt_in_view2, d_diff
    cdef np.ndarray[np.float32_t,ndim=2] RT = R1.transpose()

    cdef int width2 = scaled_depth2s.shape[1]
    cdef int height2 = scaled_depth2s.shape[0]

    cdef np.ndarray[np.uint8_t,ndim=1] mask = np.zeros((points_from_view1_in_global_frame.shape[0]), dtype=np.uint8)
    cdef np.ndarray[np.uint8_t,ndim=1] matched_pixel = np.zeros((2), dtype=np.uint8)
    cdef np.ndarray[np.float32_t,ndim=2] P2 = np.zeros((3,4), dtype=np.float32)

    cdef np.float32_t td = 0.25*sigma
    cdef np.float32_t tv = 0.075 * scaled_depth2s.shape[2]
    cdef np.float32_t div_const = (2/(255*sqrt(3)))
    cdef np.float32_t d_pt
    cdef np.float32_t w_pt
    cdef np.float32_t interpolated_w
    cdef int v_pt = 0
    cdef np.ndarray[np.float32_t,ndim=1] s = np.zeros((3), dtype=np.float32)
    cdef np.float32_t s2

    #cdef np.ndarray[np.float32_t,ndim=1] cam_vi = np.zeros((3), dtype=np.float32_t)
    #cdef np.ndarray[np.float32_t,ndim=1] cam_vj = np.zeros((3), dtype=np.float32_t)
    cdef np.float32_t cam_vi[3]
    cdef np.float32_t cam_vj[3]


    for ptIdx in range(points_from_view1_in_global_frame.shape[0]):
        d_pt = 0
        w_pt = 0
        v_pt = 0
        s[0] = 0
        s[1] = 0
        s[2] = 0
        s2 = 0

        #cur_3D_point_position = points_from_view1_in_global_frame[ptIdx,:]
        #cur_3D_point_color[0] = colors1[ptIdx,0]
        #cur_3D_point_color[1] = colors1[ptIdx,1]
        #cur_3D_point_color[2] = colors1[ptIdx,2]

        cam_vi[0] = - R1[0,0]*t1[0] - R1[1,0]*t1[1] - R1[2,0]*t1[2]
        cam_vi[1] = - R1[0,1]*t1[0] - R1[1,1]*t1[1] - R1[2,1]*t1[2]
        cam_vi[2] = - R1[0,2]*t1[0] - R1[1,2]*t1[1] - R1[2,2]*t1[2]
        cam_vj[0] = 0
        cam_vj[1] = 0
        cam_vj[2] = 0

        for vid in range(scaled_depth2s.shape[2]):
            P2[0,0] = K2s[0,0,vid]*R2s[0,0,vid] + K2s[0,1,vid]*R2s[1,0,vid] + K2s[0,2,vid]*R2s[2,0,vid]
            P2[0,1] = K2s[0,0,vid]*R2s[0,1,vid] + K2s[0,1,vid]*R2s[1,1,vid] + K2s[0,2,vid]*R2s[2,1,vid]
            P2[0,2] = K2s[0,0,vid]*R2s[0,2,vid] + K2s[0,1,vid]*R2s[1,2,vid] + K2s[0,2,vid]*R2s[2,2,vid]
            P2[0,3] = K2s[0,0,vid]*t2s[0,vid] + K2s[0,1,vid]*t2s[1,vid] + K2s[0,2,vid]*t2s[2,vid]

            P2[1,0] = K2s[1,0,vid]*R2s[0,0,vid] + K2s[1,1,vid]*R2s[1,0,vid] + K2s[1,2,vid]*R2s[2,0,vid]
            P2[1,1] = K2s[1,0,vid]*R2s[0,1,vid] + K2s[1,1,vid]*R2s[1,1,vid] + K2s[1,2,vid]*R2s[2,1,vid]
            P2[1,2] = K2s[1,0,vid]*R2s[0,2,vid] + K2s[1,1,vid]*R2s[1,2,vid] + K2s[1,2,vid]*R2s[2,2,vid]
            P2[1,3] = K2s[1,0,vid]*t2s[0,vid] + K2s[1,1,vid]*t2s[1,vid] + K2s[1,2,vid]*t2s[2,vid]

            P2[2,0] = K2s[2,0,vid]*R2s[0,0,vid] + K2s[2,1,vid]*R2s[1,0,vid] + K2s[2,2,vid]*R2s[2,0,vid]
            P2[2,1] = K2s[2,0,vid]*R2s[0,1,vid] + K2s[2,1,vid]*R2s[1,1,vid] + K2s[2,2,vid]*R2s[2,1,vid]
            P2[2,2] = K2s[2,0,vid]*R2s[0,2,vid] + K2s[2,1,vid]*R2s[1,2,vid] + K2s[2,2,vid]*R2s[2,2,vid]
            P2[2,3] = K2s[2,0,vid]*t2s[0,vid] + K2s[2,1,vid]*t2s[1,vid] + K2s[2,2,vid]*t2s[2,vid]

            cam_vj[0] = - R2s[0,0,vid]*t2s[0,vid] - R2s[1,0,vid]*t2s[1,vid] - R2s[2,0,vid]*t2s[2,vid]
            cam_vj[1] = - R2s[0,1,vid]*t2s[0,vid] - R2s[1,1,vid]*t2s[1,vid] - R2s[2,1,vid]*t2s[2,vid]
            cam_vj[2] = - R2s[0,2,vid]*t2s[0,vid] - R2s[1,2,vid]*t2s[1,vid] - R2s[2,2,vid]*t2s[2,vid]

            if cam_vi[0]*cam_vj[0] + cam_vi[1]*cam_vj[1] + cam_vi[2]*cam_vj[2] < 0:
                continue

            point4d_i[0] = points_from_view1_in_global_frame[ptIdx,0]
            point4d_i[1] = points_from_view1_in_global_frame[ptIdx,1]
            point4d_i[2] = points_from_view1_in_global_frame[ptIdx,2]

            point_proj_i[0] = P2[0,0]*point4d_i[0] + P2[0,1]*point4d_i[1] + P2[0,2]*point4d_i[2] + P2[0,3]*point4d_i[3]
            point_proj_i[1] = P2[1,0]*point4d_i[0] + P2[1,1]*point4d_i[1] + P2[1,2]*point4d_i[2] + P2[1,3]*point4d_i[3]
            point_proj_i[2] = P2[2,0]*point4d_i[0] + P2[2,1]*point4d_i[1] + P2[2,2]*point4d_i[2] + P2[2,3]*point4d_i[3]
            if point_proj_i[2] > 0.0:
                point_proj_i[0] /= point_proj_i[2]
                point_proj_i[1] /= point_proj_i[2]
                if point_proj_i[0] > borderx and point_proj_i[1] > bordery and point_proj_i[0] < width2-borderx and point_proj_i[1] < height2-bordery:
                    matched_pixel[0] = int(point_proj_i[0])
                    matched_pixel[1] = int(point_proj_i[1])

                    interpolated_w = weights2s[matched_pixel[1],matched_pixel[0],vid]

                    matched_dj_in_view2 = scaled_depth2s[matched_pixel[1],matched_pixel[0],vid]
                    if np.isfinite(matched_dj_in_view2) and matched_dj_in_view2 > 0.0:
                        px = matched_pixel[0] + 0.5
                        py = matched_pixel[1] + 0.5

                        point3d_j[0] = matched_dj_in_view2*(px - K2s[0,2,vid])/K2s[0,0,vid]
                        point3d_j[1] = matched_dj_in_view2*(py - K2s[1,2,vid])/K2s[1,1,vid]
                        point3d_j[2] = matched_dj_in_view2

                        zi_3d_pt_in_view2 = R2s[2,0,vid]*point4d_i[0] + R2s[2,1,vid]*point4d_i[1] + R2s[2,2,vid]*point4d_i[2] + t2s[2,vid]*point4d_i[3]
                        matched_zj_3d_pt_in_view2 = point4d_j[2]
                        d_diff = matched_zj_3d_pt_in_view2 - zi_3d_pt_in_view2

                        if d_diff < -sigma:
                            continue
                        if d_diff > sigma:
                            d_diff = sigma

                        d_pt = ( w_pt*d_pt + interpolated_w*d_diff/sigma ) / (w_pt + interpolated_w)
                        w_pt = w_pt + interpolated_w

                        if d_diff != sigma: #update photoconsistency only for range surfaces close to p
                            s[0] = s[0] + colors2s[matched_pixel[1],matched_pixel[0],0,vid]
                            s[1] = s[1] + colors2s[matched_pixel[1],matched_pixel[0],1,vid]
                            s[2] = s[2] + colors2s[matched_pixel[1],matched_pixel[0],2,vid]
                            s2 = s2 + s[0]*s[0] + s[1]*s[1] + s[2]*s[2]
                            v_pt = v_pt + 1
        if v_pt > 0:
            tmpVal = (s2 - (s[0]*s[0]+s[1]*s[1]+s[2]*s[2])/v_pt)
            #tmpVal = 0
            if tmpVal >= 0:
                p_pt = sqrt( tmpVal / v_pt ) * div_const
                if  d_pt > -td and d_pt < 0 and p_pt < tp and v_pt > tv:
                    mask[ptIdx] = 1

    return mask


def igl_pointcloud_filtering_in_multiviews( K1, R1, t1, points_from_view1_in_global_frame, weights1, # colors1, # scaled_depth1,
              K2s, R2s, t2s, scaled_depth2s, weights2s, colors2s, sigma, tp,
              borderx=0, bordery=0 ):

    assert points_from_view1_in_global_frame.shape[1] == 3, "point cloud 1 does not have 3 channels"
    assert scaled_depth2s.shape[0] == 192, "wrong height for depth 2s"
    assert scaled_depth2s.shape[1] == 256, "wrong width for depth 2s"
    #assert scaled_depth1.shape[0] == 192, "wrong height for depth 2s"
    #assert scaled_depth1.shape[1] == 256, "wrong width for depth 2s"

    return _igl_pointcloud_filtering_in_multiviews(
            K1.astype(np.float32),
            R1.astype(np.float32),
            t1.astype(np.float32),
            points_from_view1_in_global_frame.astype(np.float32),
            weights1.astype(np.float32),
            #colors1.astype(np.float32),
            #scaled_depth1.astype(np.float32),
            K2s.astype(np.float32),
            R2s.astype(np.float32),
            t2s.astype(np.float32),
            scaled_depth2s.astype(np.float32),
            weights2s.astype(np.float32),
            colors2s.astype(np.float32),
            sigma, tp,
            borderx,
            bordery)
