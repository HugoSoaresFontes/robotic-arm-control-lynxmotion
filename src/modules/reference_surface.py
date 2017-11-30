"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import logging
import os
from collections import deque
from platform import system
from time import time

import cv2
import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import GL_LINES
from pyglui import ui
from pyglui.cygl.utils import RGBA, draw_polyline_norm, draw_polyline, draw_points_norm, draw_points, Named_Texture
from pyglui.pyfontstash import fontstash
from pyglui.ui import get_opensans_font_path

from PyQt5.QtWidgets import QFileDialog, QWidget, QApplication

from gl_utils import adjust_gl_view, clear_gl_screen, basic_gl_setup, cvmat_to_glmat, \
    make_coord_system_norm_based
from methods import normalize, denormalize
from plugin import Plugin_List
from plugins.robot_control import Robot_Control
from plugins.tic_tac_toe_position_screen import Tic_Tac_Toe_Position_Screen
from video_capture import Surface_Source, Surface_Manager

import platform

logger = logging.getLogger(__name__)
marker_corners_norm = np.array(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)


def m_verts_to_screen(verts):
    # verts need to be sorted counter-clockwise stating at bottom left
    return cv2.getPerspectiveTransform(marker_corners_norm, verts)


def m_verts_from_screen(verts):
    # verts need to be sorted counter-clockwise stating at bottom left
    return cv2.getPerspectiveTransform(verts, marker_corners_norm)


class Global_Container(object):
    pass


class Reference_Surface(object):
    """docstring for Reference Surface

    The surface coodinate system is 0-1.
    Origin is the bottom left corner, (1,1) is the top right

    The first scalar in the pos vector is width we call this 'u'.
    The second is height we call this 'v'.
    The surface is thus defined by 4 vertecies:
        Our convention is this order: (0,0),(1,0),(1,1),(0,1)

    The surface is supported by a set of n>=1 Markers:
        Each marker has an id, you can not not have markers with the same id twice.
        Each marker has 4 verts (order is the same as the surface verts)
        Each maker vertex has a uv coord that places it on the surface

    When we find the surface in locate() we use the correspondence
    of uv and screen coords of all 4 verts of all detected markers to get the
    surface to screen homography.

    This allows us to get homographies for partially visible surfaces,
    all we need are 2 visible markers. (We could get away with just
    one marker but in pracise this is to noisy.)
    The more markers we find the more accurate the homography.

    """

    def __init__(self, g_pool, name="Superfície sem nome", saved_definition=None):
        self.g_pool = g_pool
        self.name = name
        self.markers = {}
        self.detected_markers = 0
        self.defined = False
        self.build_up_status = 0
        self.required_build_up = 90.
        self.detected = False
        self.m_to_screen = None
        self.m_from_screen = None
        self.camera_pose_3d = None
        self.use_distortion = True

        self.uid = str(time())
        self.real_world_size = {'x': 1., 'y': 1.}

        self.heatmap = np.ones(0)
        self.heatmap_detail = .2
        self.heatmap_texture = Named_Texture()
        self.gaze_history = deque()
        self.gaze_history_length = 1.0  # unit: seconds

        # window and gui vars
        self._window = None
        self.fullscreen = False
        self.window_should_open = False
        self.window_should_close = False

        self.gaze_on_srf = []  # points on surface for realtime feedback display

        self.glfont = fontstash.Context()
        self.glfont.add_font('opensans', get_opensans_font_path())
        self.glfont.set_size(22)
        self.glfont.set_color_float((0.2, 0.5, 0.9, 1.0))

        self.old_corners_robust = None
        if saved_definition is not None:
            self.load_from_dict(saved_definition)

        # UI Platform tweaks
        if system() == 'Linux':
            self.window_position_default = (0, 0)
        elif system() == 'Windows':
            self.window_position_default = (8, 31)
        else:
            self.window_position_default = (0, 0)

    def save_to_dict(self):
        """
        save all markers and name of this surface to a dict.
        """
        markers = dict([(m_id, m.uv_coords.tolist()) for m_id, m in self.markers.items()])
        return {'name': self.name, 'uid': self.uid, 'markers': markers,
                'real_world_size': self.real_world_size, 'gaze_history_length': self.gaze_history_length}

    def load_from_dict(self, d):
        """
        load all markers of this surface to a dict.
        """
        self.name = d['name']
        self.uid = d['uid']
        self.gaze_history_length = d.get('gaze_history_length', self.gaze_history_length)
        self.real_world_size = d.get('real_world_size', {'x': 1., 'y': 1.})

        marker_dict = d['markers']
        for m_id, uv_coords in marker_dict.items():
            self.markers[m_id] = Support_Marker(m_id)
            self.markers[m_id].load_uv_coords(np.asarray(uv_coords))

        # flag this surface as fully defined
        self.defined = True
        self.build_up_status = self.required_build_up

    def build_correspondance(self, visible_markers, min_marker_perimeter, min_id_confidence):
        """
        - use all visible markers
        - fit a convex quadrangle around it
        - use quadrangle verts to establish perpective transform
        - map all markers into surface space
        - build up list of found markers and their uv coords
        """
        usable_markers = [m for m in visible_markers if m['perimeter'] >= min_marker_perimeter]
        all_verts = [m['verts'] for m in usable_markers]

        if not all_verts:
            return

        all_verts = np.array(all_verts, dtype=np.float32)
        all_verts.shape = (-1, 1, 2)  # [vert,vert,vert,vert,vert...] with vert = [[r,c]]
        # all_verts_undistorted_normalized centered in img center flipped in y and range [-1,1]
        all_verts_undistorted_normalized = self.g_pool.capture.intrinsics.undistortPoints(all_verts,
                                                                                          use_distortion=self.use_distortion)
        hull = cv2.convexHull(all_verts_undistorted_normalized.astype(np.float32), clockwise=False)

        # simplify until we have excatly 4 verts
        if hull.shape[0] > 4:
            new_hull = cv2.approxPolyDP(hull, epsilon=1, closed=True)
            if new_hull.shape[0] >= 4:
                hull = new_hull
        # if hull.shape[0]>4:
        #     curvature = abs(GetAnglesPolyline(hull,closed=True))
        #     most_acute_4_threshold = sorted(curvature)[3]
        #     hull = hull[curvature<=most_acute_4_threshold]


        # all_verts_undistorted_normalized space is flipped in y.
        # we need to change the order of the hull vertecies
        hull = hull[[1, 0, 3, 2], :, :]

        # now we need to roll the hull verts until we have the right orientation:
        # all_verts_undistorted_normalized space has its origin at the image center.
        # adding 1 to the coordinates puts the origin at the top left.
        distance_to_top_left = np.sqrt((hull[:, :, 0] + 1) ** 2 + (hull[:, :, 1] + 1) ** 2)
        bot_left_idx = np.argmin(distance_to_top_left) + 1
        hull = np.roll(hull, -bot_left_idx, axis=0)

        # based on these 4 verts we calculate the transformations into a 0,0 1,1 square space
        m_from_undistored_norm_space = m_verts_from_screen(hull)
        self.detected = True
        # map the markers vertices into the surface space (one can think of these as texture coordinates u,v)
        marker_uv_coords = cv2.perspectiveTransform(all_verts_undistorted_normalized, m_from_undistored_norm_space)
        marker_uv_coords.shape = (-1, 4, 1, 2)  # [marker,marker...] marker = [ [[r,c]],[[r,c]] ]

        # build up a dict of discovered markers. Each with a history of uv coordinates
        for m, uv in zip(usable_markers, marker_uv_coords):
            try:
                self.markers[m['id']].add_uv_coords(uv)
            except KeyError:
                self.markers[m['id']] = Support_Marker(m['id'])
                self.markers[m['id']].add_uv_coords(uv)

        # average collection of uv correspondences accros detected markers
        self.build_up_status = sum([len(m.collected_uv_coords) for m in self.markers.values()]) / float(
            len(self.markers))

        if self.build_up_status >= self.required_build_up:
            self.finalize_correnspondance()

    def finalize_correnspondance(self):
        """
        - prune markers that have been visible in less than x percent of frames.
        - of those markers select a good subset of uv coords and compute mean.
        - this mean value will be used from now on to estable surface transform
        """
        persistent_markers = {}
        for k, m in self.markers.items():
            if len(m.collected_uv_coords) > self.required_build_up * .5:
                persistent_markers[k] = m
        self.markers = persistent_markers
        for m in self.markers.values():
            m.compute_robust_mean()

        self.defined = True
        if hasattr(self, 'on_finish_define'):
            self.on_finish_define()
            del self.on_finish_define

    def update_gaze_history(self):
        self.gaze_history.extend(self.gaze_on_srf)
        try:  # use newest gaze point to determine age threshold
            age_threshold = self.gaze_history[-1]['timestamp'] - self.gaze_history_length
            while self.gaze_history[1]['timestamp'] < age_threshold:
                self.gaze_history.popleft()  # remove outdated gaze points
        except IndexError:
            pass

    def generate_heatmap(self):
        data = [gp['norm_pos'] for gp in self.gaze_history if gp['confidence'] > 0.6]
        self._generate_heatmap(data)

    def _generate_heatmap(self, data):
        if not data:
            return

        grid = int(self.real_world_size['y']), int(self.real_world_size['x'])

        xvals, yvals = zip(*((x, 1. - y) for x, y in data))
        hist, *edges = np.histogram2d(yvals, xvals, bins=grid,
                                      range=[[0, 1.], [0, 1.]], normed=False)
        filter_h = int(self.heatmap_detail * grid[0]) // 2 * 2 + 1
        filter_w = int(self.heatmap_detail * grid[1]) // 2 * 2 + 1
        hist = cv2.GaussianBlur(hist, (filter_h, filter_w), 0)

        hist_max = hist.max()
        hist *= (255. / hist_max) if hist_max else 0.
        hist = hist.astype(np.uint8)
        c_map = cv2.applyColorMap(hist, cv2.COLORMAP_JET)
        # reuse allocated memory if possible
        if self.heatmap.shape != (*grid, 4):
            self.heatmap = np.ones((*grid, 4), dtype=np.uint8)
            self.heatmap[:, :, 3] = 125
        self.heatmap[:, :, :3] = c_map
        self.heatmap_texture.update_from_ndarray(self.heatmap)

    def gl_display_heatmap(self):
        if self.detected:
            m = cvmat_to_glmat(self.m_to_screen)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, 1, 0, 1, -1, 1)  # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            # apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            self.heatmap_texture.draw()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

    def locate(self, visible_markers, min_marker_perimeter, min_id_confidence, locate_3d=False, ):
        """
        - find overlapping set of surface markers and visible_markers
        - compute homography (and inverse) based on this subset
        """

        if not self.defined:
            self.build_correspondance(visible_markers, min_marker_perimeter, min_id_confidence)

        res = self._get_location(visible_markers, min_marker_perimeter, min_id_confidence, locate_3d)
        self.detected = res['detected']
        self.detected_markers = res['detected_markers']
        self.m_to_screen = res['m_to_screen']
        self.m_from_undistored_norm_space = res['m_from_undistored_norm_space']
        self.m_to_undistored_norm_space = res['m_to_undistored_norm_space']
        self.m_from_screen = res['m_from_screen']
        self.camera_pose_3d = res['camera_pose_3d']

    def _get_location(self, visible_markers, min_marker_perimeter, min_id_confidence, locate_3d=False):

        filtered_markers = [m for m in visible_markers if
                            m['perimeter'] >= min_marker_perimeter and m['id_confidence'] > min_id_confidence]
        marker_by_id = {}
        # if an id shows twice we use the bigger marker (usually this is a screen camera echo artifact.)
        for m in filtered_markers:
            if m["id"] in marker_by_id and m["perimeter"] < marker_by_id[m['id']]['perimeter']:
                pass
            else:
                marker_by_id[m["id"]] = m

        visible_ids = set(marker_by_id.keys())
        requested_ids = set(self.markers.keys())
        overlap = visible_ids & requested_ids
        # need at least two markers per surface when the surface is more that 1 marker.
        if overlap and len(overlap) >= min(2, len(requested_ids)):
            detected = True
            xy = np.array([marker_by_id[i]['verts'] for i in overlap])
            uv = np.array([self.markers[i].uv_coords for i in overlap])
            uv.shape = (-1, 1, 2)

            # our camera lens creates distortions we want to get a good 2d estimate despite that so we:
            # compute the homography transform from marker into the undistored normalized image space
            # (the line below is the same as what you find in methods.undistort_unproject_pts, except that we ommit the z corrd as it is always one.)
            xy_undistorted_normalized = self.g_pool.capture.intrinsics.undistortPoints(xy.reshape(-1, 2),
                                                                                       use_distortion=self.use_distortion)

            m_to_undistored_norm_space, mask = cv2.findHomography(uv, xy_undistorted_normalized, method=cv2.RANSAC,
                                                                  ransacReprojThreshold=100)
            if not mask.all():
                detected = False
            m_from_undistored_norm_space, mask = cv2.findHomography(xy_undistorted_normalized, uv)
            # project the corners of the surface to undistored space
            corners_undistored_space = cv2.perspectiveTransform(marker_corners_norm.reshape(-1, 1, 2),
                                                                m_to_undistored_norm_space)
            # project and distort these points  and normalize them
            corners_redistorted = self.g_pool.capture.intrinsics.projectPoints(
                cv2.convertPointsToHomogeneous(corners_undistored_space), use_distortion=self.use_distortion)
            corners_nulldistorted = self.g_pool.capture.intrinsics.projectPoints(
                cv2.convertPointsToHomogeneous(corners_undistored_space), use_distortion=self.use_distortion)

            # normalize to pupil norm space
            corners_redistorted.shape = -1, 2
            corners_redistorted /= self.g_pool.capture.intrinsics.resolution
            corners_redistorted[:, -1] = 1 - corners_redistorted[:, -1]

            # normalize to pupil norm space
            corners_nulldistorted.shape = -1, 2
            corners_nulldistorted /= self.g_pool.capture.intrinsics.resolution
            corners_nulldistorted[:, -1] = 1 - corners_nulldistorted[:, -1]

            # maps for extreme lens distortions will behave irratically beyond the image bounds
            # since our surfaces often extend beyond the screen we need to interpolate
            # between a distored projection and undistored one.

            # def ratio(val):
            #     centered_val = abs(.5 - val)
            #     # signed distance to img cennter .5 is imag bound
            #     # we look to interpolate between .7 and .9
            #     inter = max()

            corners_robust = []
            for nulldist, redist in zip(corners_nulldistorted, corners_redistorted):
                if -.4 < nulldist[0] < 1.4 and -.4 < nulldist[1] < 1.4:
                    corners_robust.append(redist)
                else:
                    corners_robust.append(nulldist)

            corners_robust = np.array(corners_robust)

            if self.old_corners_robust is not None and np.mean(np.abs(corners_robust - self.old_corners_robust)) < 0.02:
                smooth_corners_robust = self.old_corners_robust
                smooth_corners_robust += .5 * (corners_robust - self.old_corners_robust)

                corners_robust = smooth_corners_robust
                self.old_corners_robust = smooth_corners_robust
            else:
                self.old_corners_robust = corners_robust

            # compute a perspective thransform from from the marker norm space to the apparent image.
            # The surface corners will be at the right points
            # However the space between the corners may be distored due to distortions of the lens,
            m_to_screen = m_verts_to_screen(corners_robust)
            m_from_screen = m_verts_from_screen(corners_robust)

            camera_pose_3d = None
            if locate_3d:
                # 3d marker support pose estimation:
                # scale normalized object points to world space units (think m,cm,mm)
                uv.shape = -1, 2
                uv *= [self.real_world_size['x'], self.real_world_size['y']]
                # convert object points to lie on z==0 plane in 3d space
                uv3d = np.zeros((uv.shape[0], uv.shape[1] + 1))
                uv3d[:, :-1] = uv
                xy.shape = -1, 1, 2
                # compute pose of object relative to camera center
                is3dPoseAvailable, rot3d_cam_to_object, translate3d_cam_to_object = self.g_pool.capture.intrinsics.solvePnP(
                    uv3d, xy)
                print("{} \t {} \t {}".format(translate3d_cam_to_object[0], translate3d_cam_to_object[1],
                                              translate3d_cam_to_object[2]))

                if is3dPoseAvailable:
                    # not verifed, potentially usefull info: http://stackoverflow.com/questions/17423302/opencv-solvepnp-tvec-units-and-axes-directions

                    ###marker posed estimation from virtually projected points.
                    # object_pts = np.array([[[0,0],[0,1],[1,1],[1,0]]],dtype=np.float32)
                    # projected_pts = cv2.perspectiveTransform(object_pts,self.m_to_screen)
                    # projected_pts.shape = -1,2
                    # projected_pts *= img_size
                    # projected_pts.shape = -1, 1, 2
                    # # scale object points to world space units (think m,cm,mm)
                    # object_pts.shape = -1,2
                    # object_pts *= self.real_world_size
                    # # convert object points to lie on z==0 plane in 3d space
                    # object_pts_3d = np.zeros((4,3))
                    # object_pts_3d[:,:-1] = object_pts
                    # self.is3dPoseAvailable, rot3d_cam_to_object, translate3d_cam_to_object = cv2.solvePnP(object_pts_3d, projected_pts, K, dist_coef,flags=cv2.CV_EPNP)


                    # transformation from Camera Optical Center:
                    #   first: translate from Camera center to object origin.
                    #   second: rotate x,y,z
                    #   coordinate system is x,y,z where z goes out from the camera into the viewed volume.
                    # print rot3d_cam_to_object[0],rot3d_cam_to_object[1],rot3d_cam_to_object[2], translate3d_cam_to_object[0],translate3d_cam_to_object[1],translate3d_cam_to_object[2]

                    # turn translation vectors into 3x3 rot mat.
                    rot3d_cam_to_object_mat, _ = cv2.Rodrigues(rot3d_cam_to_object)

                    # to get the transformation from object to camera we need to reverse rotation and translation
                    translate3d_object_to_cam = - translate3d_cam_to_object
                    # rotation matrix inverse == transpose
                    rot3d_object_to_cam_mat = rot3d_cam_to_object_mat.T

                    # we assume that the volume of the object grows out of the marker surface and not into it. We thus have to flip the z-Axis:
                    flip_z_axix_hm = np.eye(4, dtype=np.float32)
                    flip_z_axix_hm[2, 2] = -1
                    # create a homogenous tranformation matrix from the rotation mat
                    rot3d_object_to_cam_hm = np.eye(4, dtype=np.float32)
                    rot3d_object_to_cam_hm[:-1, :-1] = rot3d_object_to_cam_mat
                    # create a homogenous tranformation matrix from the translation vect
                    translate3d_object_to_cam_hm = np.eye(4, dtype=np.float32)
                    translate3d_object_to_cam_hm[:-1, -1] = translate3d_object_to_cam.reshape(3)

                    # combine all tranformations into transformation matrix that decribes the move from object origin and orientation to camera origin and orientation
                    tranform3d_object_to_cam = np.matrix(flip_z_axix_hm) * np.matrix(
                        rot3d_object_to_cam_hm) * np.matrix(translate3d_object_to_cam_hm)
                    camera_pose_3d = tranform3d_object_to_cam
            if detected == False:
                camera_pose_3d = None
                m_from_screen = None
                m_to_screen = None
                m_from_undistored_norm_space = None
                m_to_undistored_norm_space = None

        else:
            detected = False
            camera_pose_3d = None
            m_from_screen = None
            m_to_screen = None
            m_from_undistored_norm_space = None
            m_to_undistored_norm_space = None

        return {'detected': detected, 'detected_markers': len(overlap),
                'm_from_undistored_norm_space': m_from_undistored_norm_space,
                'm_to_undistored_norm_space': m_to_undistored_norm_space, 'm_from_screen': m_from_screen,
                'm_to_screen': m_to_screen, 'camera_pose_3d': camera_pose_3d}

    def img_to_ref_surface(self, pos):
        # convenience lines to allow 'simple' vectors (x,y) to be used
        shape = pos.shape
        pos.shape = (-1, 1, 2)
        new_pos = cv2.perspectiveTransform(pos, self.m_from_screen)
        new_pos.shape = shape
        return new_pos

    def ref_surface_to_img(self, pos):
        # convenience lines to allow 'simple' vectors (x,y) to be used
        shape = pos.shape
        pos.shape = (-1, 1, 2)
        new_pos = cv2.perspectiveTransform(pos, self.m_to_screen)
        new_pos.shape = shape
        return new_pos

    @staticmethod
    def map_datum_to_surface(d, m_from_screen):
        pos = np.array([d['norm_pos']]).reshape(1, 1, 2)
        mapped_pos = cv2.perspectiveTransform(pos, m_from_screen)
        mapped_pos.shape = (2)
        on_srf = bool((0 <= mapped_pos[0] <= 1) and (0 <= mapped_pos[1] <= 1))
        return {'topic': d['topic'] + "_on_surface", 'norm_pos': (mapped_pos[0], mapped_pos[1]),
                'confidence': d['confidence'], 'on_srf': on_srf, 'base_data': d, 'timestamp': d['timestamp']}

    def map_data_to_surface(self, data, m_from_screen):
        return [self.map_datum_to_surface(d, m_from_screen) for d in data]

    def move_vertex(self, vert_idx, new_pos):
        """
        this fn is used to manipulate the surface boundary (coordinate system)
        new_pos is in uv-space coords
        if we move one vertex of the surface we need to find
        the tranformation from old quadrangle to new quardangle
        and apply that transformation to our marker uv-coords
        """
        before = marker_corners_norm
        after = before.copy()
        after[vert_idx] = new_pos
        transform = cv2.getPerspectiveTransform(after, before)
        for m in self.markers.values():
            m.uv_coords = cv2.perspectiveTransform(m.uv_coords, transform)

    def add_marker(self, marker, visible_markers, min_marker_perimeter, min_id_confidence):
        '''
        add marker to surface.
        '''
        res = self._get_location(visible_markers, min_marker_perimeter, min_id_confidence, locate_3d=False)
        if res['detected']:
            support_marker = Support_Marker(marker['id'])
            marker_verts = np.array(marker['verts'])
            marker_verts.shape = (-1, 1, 2)
            if self.use_distortion:
                marker_verts_undistorted_normalized = self.g_pool.capture.intrinsics.undistortPoints(marker_verts)
            else:
                marker_verts_undistorted_normalized = self.g_pool.capture.intrinsics.undistortPoints(marker_verts)
            marker_uv_coords = cv2.perspectiveTransform(marker_verts_undistorted_normalized,
                                                        res['m_from_undistored_norm_space'])
            support_marker.load_uv_coords(marker_uv_coords)
            self.markers[marker['id']] = support_marker

    def remove_marker(self, marker):
        if len(self.markers) == 1:
            logger.warning("Need at least one marker per surface. Will not remove this last marker.")
            return
        self.markers.pop(marker['id'])

    def marker_status(self):
        return "{}   {}/{}".format(self.name, self.detected_markers, len(self.markers))

    def get_mode_toggle(self, pos, img_shape):
        if self.detected and self.defined:
            x, y = pos
            frame = np.array([[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]], dtype=np.float32)
            frame = cv2.perspectiveTransform(frame, self.m_to_screen)
            text_anchor = frame.reshape((5, -1))[2]
            text_anchor[1] = 1 - text_anchor[1]
            text_anchor *= img_shape[1], img_shape[0]
            text_anchor = text_anchor[0], text_anchor[1] - 75
            surface_edit_anchor = text_anchor[0], text_anchor[1] + 25
            marker_edit_anchor = text_anchor[0], text_anchor[1] + 50
            if np.sqrt((x - surface_edit_anchor[0]) ** 2 + (y - surface_edit_anchor[1]) ** 2) < 15:
                return 'surface_mode'
            elif np.sqrt((x - marker_edit_anchor[0]) ** 2 + (y - marker_edit_anchor[1]) ** 2) < 15:
                return 'marker_mode'
            else:
                return None
        else:
            return None

    def save_frame(self):
        frame = self.frame

        if self.detected and frame is not None:

            if not getattr(self, '_app', None):
                self._app = QApplication([])

            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            file, _ = QFileDialog.getSaveFileName(QWidget(), "Salvar frame", "",
                                                      "JPEG Files (*.jpeg)", options=options)
            if file:
                if '.jpeg' not in file:
                    file += '.jpeg'

                cv2.imwrite(file, self.frame)
        else:
            logger.error("Superfície não detectada")

    def gl_draw_frame(self, img_size, color=(1.0, 0.2, 0.6, 1.0), highlight=False, surface_mode=False,
                      marker_mode=False):
        """
        draw surface and markers
        """
        if self.detected:
            r, g, b, a = color
            frame = np.array([[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]], dtype=np.float32)
            hat = np.array([[[.3, .7], [.7, .7], [.5, .9], [.3, .7]]], dtype=np.float32)
            hat = cv2.perspectiveTransform(hat, self.m_to_screen)
            frame = cv2.perspectiveTransform(frame, self.m_to_screen)
            alpha = min(1, self.build_up_status / self.required_build_up)
            if highlight:
                draw_polyline_norm(frame.reshape((5, 2)), 1, RGBA(r, g, b, a * .1), line_type=GL_POLYGON)
            draw_polyline_norm(frame.reshape((5, 2)), 1, RGBA(r, g, b, a * alpha))
            draw_polyline_norm(hat.reshape((4, 2)), 1, RGBA(r, g, b, a * alpha))
            text_anchor = frame.reshape((5, -1))[2]
            text_anchor[1] = 1 - text_anchor[1]
            text_anchor *= img_size[1], img_size[0]
            text_anchor = text_anchor[0], text_anchor[1] - 75
            surface_edit_anchor = text_anchor[0], text_anchor[1] + 25
            marker_edit_anchor = text_anchor[0], text_anchor[1] + 50
            if self.defined:
                if marker_mode:
                    draw_points([marker_edit_anchor], color=RGBA(0, .8, .7))
                else:
                    draw_points([marker_edit_anchor])
                if surface_mode:
                    draw_points([surface_edit_anchor], color=RGBA(0, .8, .7))
                else:
                    draw_points([surface_edit_anchor])

                self.glfont.set_blur(3.9)
                self.glfont.set_color_float((0, 0, 0, .8))
                self.glfont.draw_text(text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status())
                self.glfont.draw_text(surface_edit_anchor[0] + 15, surface_edit_anchor[1] + 6, 'editar superfície')
                self.glfont.draw_text(marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6,
                                      'adicionar/remover marcadores')
                self.glfont.set_blur(0.0)
                self.glfont.set_color_float((0.1, 8., 8., .9))
                self.glfont.draw_text(text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status())
                self.glfont.draw_text(surface_edit_anchor[0] + 15, surface_edit_anchor[1] + 6, 'editar superfície')
                self.glfont.draw_text(marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6,
                                      'adicionar/remover marcadores')
            else:
                progress = (self.build_up_status / float(self.required_build_up)) * 100
                progress_text = '%.0f%%' % progress
                self.glfont.set_blur(3.9)
                self.glfont.set_color_float((0, 0, 0, .8))
                self.glfont.draw_text(text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status())
                self.glfont.draw_text(surface_edit_anchor[0] + 15, surface_edit_anchor[1] + 6,
                                      'procurando marcadores...')
                self.glfont.draw_text(marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6, progress_text)
                self.glfont.set_blur(0.0)
                self.glfont.set_color_float((0.1, 8., 8., .9))
                self.glfont.draw_text(text_anchor[0] + 15, text_anchor[1] + 6, self.marker_status())
                self.glfont.draw_text(surface_edit_anchor[0] + 15, surface_edit_anchor[1] + 6,
                                      'procurando marcadores...')
                self.glfont.draw_text(marker_edit_anchor[0] + 15, marker_edit_anchor[1] + 6, progress_text)

    def gl_draw_corners(self):
        """
        draw surface and markers
        """
        if self.detected:
            frame = cv2.perspectiveTransform(marker_corners_norm.reshape(-1, 1, 2), self.m_to_screen)
            draw_points_norm(frame.reshape((4, 2)), 20, RGBA(1.0, 0.2, 0.6, .5))

    @property
    def frame(self):
        if self.g_pool.capture._recent_frame and self.detected:
            w, h = self.g_pool.capture._recent_frame.width, self.g_pool.capture._recent_frame.height
            frame = self.g_pool.capture._recent_frame.gray

            pontos = cv2.perspectiveTransform(marker_corners_norm.reshape(-1, 1, 2), self.m_to_screen).reshape(4, 2)
            pontos = np.array([[p[0] * w, h - p[1] * h] for p in pontos], dtype=np.float32)

            m = cv2.getPerspectiveTransform(pontos,
                                            np.array(((0, 0), (500, 0), (500, 500), (0, 500)), dtype=np.float32))
            frame = cv2.warpPerspective(self.g_pool.capture._recent_frame.bgr, m, (500, 500))
            return frame

        return None

    def gl_display_in_window_(self):
        if self._window:
            active_window = glfw.get_current_context()
            glfw.make_context_current(self._window)

            glClearColor(0.2, 0.2, 0.2, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            events = {}

            for p in self.g_pool_inner.plugins:
                p.recent_events(events)

            self.g_pool_inner.plugins.clean()

            for p in self.g_pool_inner.plugins:
                p.gl_display()

            glfw.make_context_current(self._window)
            # render visual feedback from loaded plugins

            glViewport(0, 0, *glfw.get_window_size(self._window))
            unused_elements = self.g_pool_inner.gui.update()

            for button, action, mods in unused_elements.buttons:
                pos = glfw.get_cursor_pos(self._window)
                pos = normalize(pos, self.camera_render_size)
                # Position in img pixels
                pos = denormalize(pos, self.g_pool_inner.capture.frame_size)

                for p in self.g_pool_inner.plugins:
                    p.on_click(pos, button, action)

            glfw.swap_buffers(self._window)
            glfw.make_context_current(active_window)

    # #### fns to draw surface in seperate window
    def gl_display_in_window(self, world_tex):
        """
        here we map a selected surface onto a seperate window.
        """
        if self._window and self.detected:
            active_window = glfw.get_current_context()
            glfw.make_context_current(self._window)
            clear_gl_screen()

            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_from_screen)

            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, 1, 0, 1, -1, 1)  # gl coord convention

            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            # apply m  to our quad - this will stretch the quad such that the ref suface will span the window extends
            glLoadMatrixf(m)

            world_tex.draw()

            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()

            # now lets get recent pupil positions on this surface:
            for gp in self.gaze_on_srf:
                draw_points_norm([gp['norm_pos']], color=RGBA(0.0, 0.8, 0.5, 0.8), size=80)

            glfw.swap_buffers(self._window)
            glfw.make_context_current(active_window)

    #### fns to draw surface in separate window
    def gl_display_in_window_3d(self, world_tex):
        """
        here we map a selected surface onto a seperate window.
        """
        K, img_size = self.g_pool.capture.intrinsics.K, self.g_pool.capture.intrinsics.resolution

        if self._window and self.camera_pose_3d is not None:
            active_window = glfw.get_current_context()
            glfw.make_context_current(self._window)
            glClearColor(.8, .8, .8, 1.)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearDepth(1.0)
            glDepthFunc(GL_LESS)
            glEnable(GL_DEPTH_TEST)
            # self.trackball.push()

            glMatrixMode(GL_MODELVIEW)

            draw_coordinate_system(l=self.real_world_size['x'])
            glPushMatrix()
            glScalef(self.real_world_size['x'], self.real_world_size['y'], 1)
            draw_polyline([[0, 0], [0, 1], [1, 1], [1, 0]], color=RGBA(.5, .3, .1, .5), thickness=3)
            glPopMatrix()
            # Draw the world window as projected onto the plane using the homography mapping
            glPushMatrix()
            glScalef(self.real_world_size['x'], self.real_world_size['y'], 1)
            # cv uses 3x3 gl uses 4x4 tranformation matricies
            m = cvmat_to_glmat(self.m_from_screen)
            glMultMatrixf(m)
            glTranslatef(0, 0, -.01)
            world_tex.draw()
            draw_polyline([[0, 0], [0, 1], [1, 1], [1, 0]], color=RGBA(.5, .3, .6, .5), thickness=3)
            glPopMatrix()

            # Draw the camera frustum and origin using the 3d tranformation obtained from solvepnp
            glPushMatrix()
            glMultMatrixf(self.camera_pose_3d.T.flatten())
            draw_frustum(img_size, K, 150)
            glLineWidth(1)
            draw_frustum(img_size, K, .1)
            draw_coordinate_system(l=5)
            glPopMatrix()

            # self.trackball.pop()

            glfw.swap_buffers(self._window)
            glfw.make_context_current(active_window)

    def open_window(self):
        if not self._window:
            # UI Platform tweaks
            if platform.system() == 'Linux':
                self.scroll_factor = 10.0
                window_position_default = (30, 30)
            elif platform.system() == 'Windows':
                self.scroll_factor = 10.0
                window_position_default = (8, 31)
            else:
                self.scroll_factor = 1.0
                window_position_default = (0, 0)

            if self.fullscreen:
                monitor = glfw.get_monitors()[self.monitor_idx]
                mode = glfw.get_video_mode(monitor)
                height, width = mode[0], mode[1]
            else:
                monitor = None
                height, width = 640, int(640. / (
                self.real_world_size['x'] / self.real_world_size['y']))  # open with same aspect ratio as surface

            self._window = glfw.create_window(height, width, "Superfície de referência: " + self.name, monitor=monitor,
                                              share=glfw.get_current_context())

            if not self.fullscreen:
                glfw.set_window_pos(self._window, self.window_position_default[0], self.window_position_default[1])

            # self.trackball = Trackball()
            self.input = {'down': False, 'mouse': (0, 0)}

            self.icon_bar_width = 400
            window_size = None
            camera_render_size = None
            self.hdpi_factor = 1.0

            g_pool = Global_Container()

            g_pool.app = 'capture'
            g_pool.process = 'surface'

            g_pool.user_dir = self.g_pool.user_dir
            g_pool.capture = None
            g_pool.surface_tracker = self.g_pool.surface_tracker
            g_pool.get_timestamp = self.g_pool.get_timestamp
            g_pool.get_now = self.g_pool.get_now

            default_frame_settings = {
                'frame_rate': 30,
                'frame_size': (700, 700),
                'preferred_names': [self.name]
            }

            default_plugins = [("Surface_Source", default_frame_settings),
                               ("Robot_Control", {}),
                               ("Tic_Tac_Toe_Position_Screen", {})]

            plugins = [Surface_Source, Surface_Manager, Robot_Control, Tic_Tac_Toe_Position_Screen]

            g_pool.plugin_by_name = {p.__name__: p for p in plugins}

            g_pool.image_tex = Named_Texture()
            g_pool.main_window = self._window

            g_pool.capture = None
            g_pool.timebase = self.g_pool.timebase

            g_pool.gui = ui.UI()
            g_pool.gui_user_scale = 1.0
            g_pool.gui.scale = 1.0
            g_pool.menubar = ui.Scrolling_Menu("Configurações", pos=(-400, 0), size=(-50, 0),
                                               header_pos='headline')
            g_pool.iconbar = ui.Scrolling_Menu("Icons", pos=(-50, 0), size=(0, 0), header_pos='hidden')
            g_pool.quickbar = ui.Stretching_Menu('Quick Bar', (0, 100), (120, -100))
            g_pool.gui.append(g_pool.menubar)
            g_pool.gui.append(g_pool.iconbar)
            g_pool.gui.append(g_pool.quickbar)

            def set_scale(new_scale):
                g_pool.gui_user_scale = new_scale
                window_size = self.camera_render_size[0] + \
                              int(self.icon_bar_width * g_pool.gui_user_scale * self.hdpi_factor), \
                              glfw.get_framebuffer_size(self._window)[1]
                glfw.set_window_size(self._window, *window_size)

            def toggle_general_settings(collapsed):
                # g_pool.menubar.collapsed = collapsed
                for m in g_pool.menubar.elements:
                    m.collapsed = True
                general_settings.collapsed = False

            def set_window_size():
                f_width, f_height = g_pool.capture.frame_size
                f_width += int(self.icon_bar_width * g_pool.gui.scale)
                glfw.set_window_size(self._window, f_width, f_height)

            general_settings = ui.Growing_Menu('Geral', header_pos='headline')

            general_settings.append(ui.Button('Resetar tamanho da janela', set_window_size))
            general_settings.append(
                ui.Selector('gui_user_scale', g_pool, setter=set_scale, selection=[.6, .8, 1., 1.2, 1.4],
                            label='Tamanho da interface'))

            g_pool.menubar.append(general_settings)
            icon = ui.Icon('collapsed', general_settings, label=chr(0xe8b8), on_val=False, off_val=True,
                           setter=toggle_general_settings, label_font='pupil_icons')
            icon.tooltip = 'Configurações gerais'
            g_pool.iconbar.append(icon)

            g_pool.plugins = Plugin_List(g_pool, default_plugins)

            self.g_pool_inner = g_pool

            # Register callbacks
            glfw.set_framebuffer_size_callback(self._window, self.on_resize)
            glfw.set_key_callback(self._window, self.on_window_key)
            glfw.set_char_callback(self._window, self.on_window_char)
            glfw.set_window_close_callback(self._window, self.on_close)
            glfw.set_mouse_button_callback(self._window, self.on_window_mouse_button)
            glfw.set_cursor_pos_callback(self._window, self.on_pos)
            glfw.set_scroll_callback(self._window, self.on_scroll)

            glfw.set_window_pos(self._window, *window_position_default)
            # self.on_resize(self._window, *glfw.get_window_size(self._window))
            set_window_size()

            # gl_state settings
            active_window = glfw.get_current_context()
            glfw.make_context_current(self._window)
            basic_gl_setup()
            make_coord_system_norm_based()

            # refresh speed settings
            glfw.swap_interval(0)

            glfw.make_context_current(active_window)

    def close_window(self):
        if self._window:
            for p in self.g_pool_inner.plugins:
                p.alive = False
            self.g_pool_inner.plugins.clean()

            glfw.destroy_window(self._window)
            self._window = None
            self.window_should_close = False

    def open_close_window(self):
        if self._window:
            self.close_window()
        else:
            self.open_window()

    def on_resize(self, window, w, h):
        altura, largura = max(h, 1), max(w, 1)

        active_window = glfw.get_current_context()
        glfw.make_context_current(window)

        fator_escala = float(glfw.get_framebuffer_size(window)[0] / glfw.get_window_size(window)[0])
        largura, altura = int(largura * fator_escala), int(altura * fator_escala)

        camera_render_size = largura - int(self.icon_bar_width * self.g_pool_inner.gui.scale), altura
        if camera_render_size[0] < 0:
            camera_render_size = (0, camera_render_size[1])

        self.camera_render_size = camera_render_size

        glfw.make_context_current(window)
        adjust_gl_view(largura, altura)

        self.g_pool_inner.gui.update_window(largura, altura)
        self.g_pool_inner.gui.collect_menus()

        for p in self.g_pool_inner.plugins:
            p.on_window_resize(window, *camera_render_size)

        glfw.make_context_current(active_window)

    def on_window_key(self, window, key, scancode, action, mods):
        self.g_pool_inner.gui.update_key(key, scancode, action, mods)
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                self.on_close()

    def on_window_char(self, window, key):
        self.g_pool_inner.gui.update_char(key)

    def on_close(self, window=None):
        self.close_window()

    def on_window_mouse_button(self, window, button, action, mods):
        self.g_pool_inner.gui.update_button(button, action, mods)
        if action == glfw.PRESS:
            self.input['down'] = True
            self.input['mouse'] = glfw.get_cursor_pos(window)
        if action == glfw.RELEASE:
            self.input['down'] = False

    def on_pos(self, window, x, y):
        if self.input['down']:
            old_x, old_y = self.input['mouse']
            # self.trackball.drag_to(x-old_x,y-old_y)
            self.input['mouse'] = x, y

        x, y = x * self.hdpi_factor, y * self.hdpi_factor
        self.g_pool_inner.gui.update_mouse(x, y)

        pos = x, y
        pos = normalize(pos, self.camera_render_size)
        pos = denormalize(pos, self.g_pool_inner.capture.frame_size)
        for p in self.g_pool_inner.plugins:
            p.on_pos(pos)

    def on_scroll(self, window, x, y):
        self.g_pool_inner.gui.update_scroll(x, y * self.scroll_factor)
        # self.trackball.zoom_to(y)
        pass

    #
    #
    def cleanup(self):
        if self._window:
            self.close_window()


class Support_Marker(object):
    '''
    This is a class only to be used by Reference_Surface
    it decribes the used markers with the uv coords of its verts.
    '''

    def __init__(self, uid):
        self.uid = uid
        self.uv_coords = None
        self.collected_uv_coords = []
        self.robust_uv_cords = False

    def load_uv_coords(self, uv_coords):
        self.uv_coords = uv_coords
        self.robust_uv_cords = True

    def add_uv_coords(self, uv_coords):
        self.collected_uv_coords.append(uv_coords)
        self.uv_coords = uv_coords

    def compute_robust_mean(self, threshhold=.1):
        """
        treat 50% as outliers. Assume majory is right.
        """
        # a stacked list of marker uv coords. marker uv cords are 4 verts with each a uv position.
        uv = np.array(self.collected_uv_coords)
        # # the mean marker uv_coords including outliers
        base_line_mean = np.mean(uv, axis=0)
        # # devidation is the distance of each scalar (4*2 per marker to the mean value of this scalar acros our stacked list)
        deviation = uv - base_line_mean
        # # now we treat the four uv scalars as a vector in 8-d space and compute the distace to the mean
        distance = np.linalg.norm(deviation, axis=(1, 3)).reshape(-1)
        # lets get the .5 cutof;
        cut_off = sorted(distance)[len(distance) // 2]
        # filter the better half
        uv_subset = uv[distance <= cut_off]
        # claculate the mean of this subset
        uv_mean = np.mean(uv_subset, axis=0)
        # use it
        self.uv_coords = uv_mean
        self.robust_uv_cords = True


def draw_frustum(img_size, K, scale=1):
    # average focal length
    f = (K[0, 0] + K[1, 1]) / 2
    # compute distances for setting up the camera pyramid
    W = 0.5 * (img_size[0])
    H = 0.5 * (img_size[1])
    Z = f
    # scale the pyramid
    W /= scale
    H /= scale
    Z /= scale
    # draw it
    glColor4f(1, 0.5, 0, 0.5)
    glBegin(GL_LINE_LOOP)
    glVertex3f(0, 0, 0)
    glVertex3f(-W, H, Z)
    glVertex3f(W, H, Z)
    glVertex3f(0, 0, 0)
    glVertex3f(W, H, Z)
    glVertex3f(W, -H, Z)
    glVertex3f(0, 0, 0)
    glVertex3f(W, -H, Z)
    glVertex3f(-W, -H, Z)
    glVertex3f(0, 0, 0)
    glVertex3f(-W, -H, Z)
    glVertex3f(-W, H, Z)
    glEnd()


def draw_coordinate_system(l=1):
    # Draw x-axis line.
    glColor3f(1, 0, 0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(l, 0, 0)
    glEnd()

    # Draw y-axis line.
    glColor3f(0, 1, 0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, l, 0)
    glEnd()

    # Draw z-axis line.
    glColor3f(0, 0, 1)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, l)
    glEnd()


if __name__ == '__main__':
    rotation3d = np.array([1, 2, 3], dtype=np.float32)
    translation3d = np.array([50, 60, 70], dtype=np.float32)

    # transformation from Camera Optical Center:
    #   first: translate from Camera center to object origin.
    #   second: rotate x,y,z
    #   coordinate system is x,y,z positive (not like opengl, where the z-axis is flipped.)
    # print rotation3d[0],rotation3d[1],rotation3d[2], translation3d[0],translation3d[1],translation3d[2]

    # turn translation vectors into 3x3 rot mat.
    rotation3dMat, _ = cv2.Rodrigues(rotation3d)

    # to get the transformation from object to camera we need to reverse rotation and translation
    #
    tranform3d_to_camera_translation = np.eye(4, dtype=np.float32)
    tranform3d_to_camera_translation[:-1, -1] = - translation3d

    # rotation matrix inverse == transpose
    tranform3d_to_camera_rotation = np.eye(4, dtype=np.float32)
    tranform3d_to_camera_rotation[:-1, :-1] = rotation3dMat.T

    print(tranform3d_to_camera_translation)
    print(tranform3d_to_camera_rotation)
    print(np.matrix(tranform3d_to_camera_rotation) * np.matrix(tranform3d_to_camera_translation))




    # rMat, _ = cv2.Rodrigues(rotation3d)
    # self.from_camera_to_referece = np.eye(4, dtype=np.float32)
    # self.from_camera_to_referece[:-1,:-1] = rMat
    # self.from_camera_to_referece[:-1, -1] = translation3d.reshape(3)
    # # self.camera_pose_3d = np.linalg.inv(self.camera_pose_3d)
