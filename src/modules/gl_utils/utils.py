"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import glfw
import numpy as np
from OpenGL.GL import *

OpenGL.ERROR_LOGGING = False

__all__ = ['make_coord_system_norm_based',
           'make_coord_system_pixel_based',
           'make_coord_system_image_centred_norm_based',
           'adjust_gl_view',
           'clear_gl_screen',
           'basic_gl_setup',
           'cvmat_to_glmat',
           'is_window_visible',
           'Coord_System'
           ]


def is_window_visible(window):
    visible = glfw.get_window_attrib(window, glfw.VISIBLE)
    iconified = glfw.get_window_attrib(window, glfw.ICONIFIED)
    return visible and not iconified


def cvmat_to_glmat(m):
    mat = np.eye(4, dtype=np.float32)
    mat = mat.flatten()
    # convert to OpenGL matrix
    mat[0] = m[0, 0]
    mat[4] = m[0, 1]
    mat[12] = m[0, 2]
    mat[1] = m[1, 0]
    mat[5] = m[1, 1]
    mat[13] = m[1, 2]
    mat[3] = m[2, 0]
    mat[7] = m[2, 1]
    mat[15] = m[2, 2]
    return mat


def basic_gl_setup():
    glEnable(GL_POINT_SPRITE)
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)
    glClearColor(0.2, 0.2, 0.2, 0.)
    glEnable(GL_LINE_SMOOTH)


def clear_gl_screen():
    glClear(GL_COLOR_BUFFER_BIT)


def adjust_gl_view(w, h):
    """
    adjust view onto our scene.
    """
    h = max(h, 1)
    w = max(w, 1)
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, w, h, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def make_coord_system_pixel_based(img_shape, flip=False):
    height, width, channels = img_shape
    # Set Projection Matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if flip:
        glOrtho(width, 0, 0, height, -1, 1)  # origin in the top left corner just like the img np-array
    else:
        glOrtho(0, width, height, 0, -1, 1)  # origin in the top left corner just like the img np-array

    # Switch back to Model View Matrix
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def make_coord_system_image_centred_norm_based(img_shape, screen_shape, flip=False):
    comprimento_img, altura_img = img_shape[:2]
    comprimento, altura = screen_shape

    fatores = float(comprimento) / comprimento_img, float(altura) / altura_img
    fator = min(fatores)

    glClearColor(0.2, 0.2, 0.2, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glViewport(
        int((comprimento - fator * comprimento_img) // 2),
        int((altura - fator * altura_img) // 2),
        int(fator * comprimento_img),
        int(fator * altura_img)
    )
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if flip:
        glOrtho(1, 0, 1, 0, -1, 1)
    else:
        glOrtho(0, 1, 0, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def make_coord_system_norm_based(flip=False):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if flip:
        glOrtho(1, 0, 1, 0, -1, 1)
    else:
        glOrtho(0, 1, 0, 1, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


class Coord_System(object):
    """docstring for Coord_System"""

    def __init__(self, left, right, bottom, top):
        super(Coord_System, self).__init__()
        self.bounds = left, right, bottom, top

    def __enter__(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(*self.bounds, -1, 1)  # gl coord convention

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

    def __exit__(self, *exc):
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
