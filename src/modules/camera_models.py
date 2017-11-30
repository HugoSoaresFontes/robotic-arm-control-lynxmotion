'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

# logging
import logging
import os

import cv2
import numpy as np

from file_methods import save_object, load_object

logger = logging.getLogger(__name__)
__version__ = 1

# these are calibration we recorded. They are estimates and generalize our setup. Its always better to calibrate each camera.
pre_recorded_calibrations = {}


def load_intrinsics(directory, cam_name, resolution):
    """
    Loads a pre-recorded intrinsics calibration for the given camera and resolution. If no pre-recorded calibration is available we fall back on default values.
    :param directory: The directory in which to look for the intrinsincs file
    :param cam_name: Name of the camera, e.g. 'Pupil Cam 1 ID2'
    :param resolution: Camera resolution given as a tuple.
    :return: Camera Model Object
    """
    file_path = os.path.join(directory, '{}.intrinsics'.format(cam_name.replace(" ", "_")))
    try:
        calib_dict = load_object(file_path, allow_legacy=False)

        if calib_dict['version'] < __version__:
            logger.warning('Deprecated camera calibration found.')
            logger.info('Please recalibrate using the Camera Intrinsics Estimation calibration.')
            os.rename(file_path, '{}.deprecated.v{}'.format(file_path, calib_dict['version']))

        intrinsics = calib_dict[str(resolution)]
        logger.info("Previously recorded calibration found and loaded!")
    except Exception as e:
        logger.info("No user calibration found for camera {} at resolution {}".format(cam_name, resolution))

        if cam_name in pre_recorded_calibrations and str(resolution) in pre_recorded_calibrations[cam_name]:
            logger.info("Loading pre-recorded calibration")
            intrinsics = pre_recorded_calibrations[cam_name][str(resolution)]
        else:
            logger.info("No pre-recorded calibration available")
            logger.warning("Loading dummy calibration")
            intrinsics = {'cam_type': 'dummy'}

    if intrinsics['cam_type'] == 'dummy':
        return Dummy_Camera(resolution, cam_name)
    elif intrinsics['cam_type'] == 'fisheye':
        return Fisheye_Dist_Camera(intrinsics['camera_matrix'], intrinsics['dist_coefs'], resolution, cam_name)
    elif intrinsics['cam_type'] == 'radial':
        return Radial_Dist_Camera(intrinsics['camera_matrix'], intrinsics['dist_coefs'], resolution, cam_name)


def save_intrinsics(directory, cam_name, resolution, intrinsics):
    """
    Saves camera intrinsics calibration to a file. For each unique camera name we maintain a single file containing all calibrations associated with this camera name.
    :param directory: Directory to which the intrinsics file will be written
    :param cam_name: Name of the camera, e.g. 'Pupil Cam 1 ID2'
    :param resolution: Camera resolution given as a tuple. This needs to match the resolution the calibration has been computed with.
    :param intrinsics: The camera intrinsics dictionary.
    :return:
    """
    # Try to load previous camera calibrations
    save_path = os.path.join(directory, '{}.intrinsics'.format(cam_name.replace(" ", "_")))
    try:
        calib_dict = load_object(save_path, allow_legacy=False)
    except:
        calib_dict = {}

    calib_dict['version'] = __version__
    calib_dict[str(resolution)] = intrinsics

    save_object(calib_dict, save_path)
    logger.info("Calibration for camera {} at resolution {} saved to {}".format(cam_name, resolution, save_path))


class Fisheye_Dist_Camera(object):
    """ Camera model assuming a lense with fisheye distortion.
        Provides functionality to make use of a fisheye camera calibration.
        The implementation of cv2.fisheye is buggy and some functions had to be customized.
    """

    def __init__(self, K, D, resolution, name):
        self.K = np.array(K)
        self.D = np.array(D)
        self.resolution = resolution
        self.name = name

    def undistort(self, img):
        """
        Undistortes an image based on the camera model.
        :param img: Distorted input image
        :return: Undistorted image
        """
        R = np.eye(3)

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            np.array(self.K),
            np.array(self.D),
            R,
            np.array(self.K),
            self.resolution,
            cv2.CV_16SC2
        )

        undistorted_img = cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        return undistorted_img

    def undistortPoints(self, dist_pts, use_distortion=True):
        """
        Undistorts points according to the camera model.
        cv2.fisheye.undistortPoints does *NOT* perform the same unprojection step the original cv2.undistortPoints does.
        Thus we implement this function ourselves.
        :param dist_pts: Distorted points. Can be a list of points or a single point.
        :return: Array of undistorted points with the same shape as the input
        """
        input_shape = dist_pts.shape

        dist_pts = dist_pts.reshape((-1, 2))
        eps = np.finfo(np.float32).eps

        f = np.array((self.K[0, 0], self.K[1, 1])).reshape(1, 2)
        c = np.array((self.K[0, 2], self.K[1, 2])).reshape(1, 2)
        if use_distortion:
            k = self.D.ravel().astype(np.float32)
        else:
            k = np.asarray([1. / 3., 2. / 15., 17. / 315., 62. / 2835.], dtype=np.float32)

        pi = dist_pts.astype(np.float32)
        pw = (pi - c) / f

        theta_d = np.linalg.norm(pw, ord=2, axis=1)
        theta = theta_d
        for j in range(10):
            theta2 = theta ** 2
            theta4 = theta2 ** 2
            theta6 = theta4 * theta2
            theta8 = theta6 * theta2
            theta = theta_d / (1 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8)

        scale = np.tan(theta) / (theta_d + eps)

        pu = pw * scale.reshape(-1, 1)

        pu.shape = input_shape
        return pu

    def projectPoints(self, object_points, rvec=None, tvec=None, use_distortion=True):
        """
        Projects a set of points onto the camera plane as defined by the camera model.
        :param object_points: Set of 3D world points
        :param rvec: Set of vectors describing the rotation of the camera when recording the corresponding object point
        :param tvec: Set of vectors describing the translation of the camera when recording the corresponding object point
        :return: Projected 2D points
        """
        skew = 0

        input_dim = object_points.ndim

        object_points = object_points.reshape((1, -1, 3))

        if rvec is None:
            rvec = np.zeros(3).reshape(1, 1, 3)
        else:
            rvec = np.array(rvec).reshape(1, 1, 3)

        if tvec is None:
            tvec = np.zeros(3).reshape(1, 1, 3)
        else:
            tvec = np.array(tvec).reshape(1, 1, 3)

        if use_distortion:
            _D = self.D
        else:
            _D = np.asarray([[1. / 3., 2. / 15., 17. / 315., 62. / 2835.]])

        image_points, jacobian = cv2.fisheye.projectPoints(
            object_points,
            rvec,
            tvec,
            self.K,
            _D,
            alpha=skew
        )

        if input_dim == 2:
            image_points.shape = (-1, 2)
        elif input_dim == 3:
            image_points.shape = (-1, 1, 2)
        return image_points

    def solvePnP(self, uv3d, xy):
        # xy_undist = self.undistortPoints(xy)
        # f = np.array((self.K[0, 0], self.K[1, 1])).reshape(1, 2)
        # c = np.array((self.K[0, 2], self.K[1, 2])).reshape(1, 2)
        # xy_undist = xy_undist * f + c
        # xy_undist = cv2.fisheye.undistortPoints(xy, self.K, self.D, P=self.K)
        if xy.ndim == 2:
            xy = np.expand_dims(xy, 0)

        xy_undist = cv2.fisheye.undistortPoints(
            xy.astype(np.float32),
            self.K,
            self.D,
            R=np.eye(3),
            P=self.K
        )

        xy_undist = np.squeeze(xy_undist)
        res = cv2.solvePnP(uv3d, xy_undist, self.K, np.array([[0, 0, 0, 0, 0]]), flags=cv2.SOLVEPNP_ITERATIVE)
        return res

    def save(self, directory, custom_name=None):
        """
        Saves the current calibration to corresponding camera's calibrations file
        :param directory: save directory
        :return:
        """
        intrinsics = {'camera_matrix': self.K.tolist(), 'dist_coefs': self.D.tolist(),
                      'resolution': self.resolution, 'cam_type': 'fisheye'}
        save_intrinsics(directory, custom_name or self.name, self.resolution, intrinsics)


class Radial_Dist_Camera(object):
    """ Camera model assuming a lense with radial distortion (this is the defaut model in opencv).
        Provides functionality to make use of a pinhole camera calibration that is also compensating for lense distortion
    """

    def __init__(self, K, D, resolution, name):
        self.K = np.array(K)
        self.D = np.array(D)
        self.resolution = resolution
        self.name = name

    def undistort(self, img):
        """
                Undistortes an image based on the camera model.
                :param img: Distorted input image
                :return: Undistorted image
                """
        undist_img = cv2.undistort(img, self.K, self.D)
        return undist_img

    def undistortPoints(self, dist_pts, use_distortion=True):
        """
        Undistorts points according to the camera model.
        :param dist_pts: Distorted points. Can be a list of points or a single point.
        :return: Array of undistorted points with the same shape as the input
        """
        dist_pts = np.array(dist_pts)
        input_shape = dist_pts.shape

        # If given a single point expand to a 1-point list
        if len(dist_pts.shape) == 1:
            dist_pts = dist_pts.reshape((1, 2))

        # Delete any posibly wrong 3rd dimension
        if dist_pts.ndim == 3:
            dist_pts = dist_pts.reshape((-1, 2))

        # Add third dimension the way cv2 wants it
        if dist_pts.ndim == 2:
            dist_pts = dist_pts.reshape((-1, 1, 2))

        if use_distortion:
            _D = self.D
        else:
            _D = np.asarray([[0., 0., 0., 0., 0.]])

        undist_pts = cv2.undistortPoints(
            dist_pts.astype(np.float32),
            self.K,
            _D,
        )

        # Restore whatever shape we had in the beginning
        undist_pts.shape = input_shape

        return undist_pts

    def projectPoints(self, object_points, rvec=None, tvec=None, use_distortion=True):
        """
        Projects a set of points onto the camera plane as defined by the camera model.
        :param object_points: Set of 3D world points
        :param rvec: Set of vectors describing the rotation of the camera when recording the corresponding object point
        :param tvec: Set of vectors describing the translation of the camera when recording the corresponding object point
        :return: Projected 2D points
        """
        input_dim = object_points.ndim

        object_points = object_points.reshape((1, -1, 3))

        if rvec is None:
            rvec = np.zeros(3).reshape(1, 1, 3)
        else:
            rvec = np.array(rvec).reshape(1, 1, 3)

        if tvec is None:
            tvec = np.zeros(3).reshape(1, 1, 3)
        else:
            tvec = np.array(tvec).reshape(1, 1, 3)

        if use_distortion:
            _D = self.D
        else:
            _D = np.asarray([[0., 0., 0., 0., 0.]])

        image_points, jacobian = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.K,
            _D
        )

        if input_dim == 2:
            image_points.shape = (-1, 2)
        elif input_dim == 3:
            image_points.shape = (-1, 1, 2)
        return image_points

    def solvePnP(self, uv3d, xy):
        res = cv2.solvePnP(uv3d, xy, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        return res

    def save(self, directory, custom_name=None):
        """
        Saves the current calibration to corresponding camera's calibrations file
        :param directory: save location
        :return:
        """
        intrinsics = {'camera_matrix': self.K.tolist(), 'dist_coefs': self.D.tolist(),
                      'resolution': self.resolution, 'cam_type': 'radial'}
        save_intrinsics(directory, custom_name or self.name, self.resolution, intrinsics)


class Dummy_Camera(Radial_Dist_Camera):
    """
    Dummy Camera model assuming no lense distortion and idealized camera intrinsics.
    """

    def __init__(self, resolution, name):
        camera_matrix = [[1000, 0., resolution[0] / 2.],
                         [0., 1000, resolution[1] / 2.],
                         [0., 0., 1.]]
        dist_coefs = [[0., 0., 0., 0., 0.]]
        super().__init__(camera_matrix, dist_coefs, resolution, name)

    def save(self, directory, custom_name=None):
        """
        Saves the current calibration to corresponding camera's calibrations file
        :param directory: save location
        :return:
        """
        intrinsics = {'camera_matrix': self.K.tolist(), 'dist_coefs': self.D.tolist(),
                      'resolution': self.resolution, 'cam_type': 'dummy'}
        save_intrinsics(directory, custom_name or self.name, self.resolution, intrinsics)
