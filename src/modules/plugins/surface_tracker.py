"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

# logging
import logging
import os
from math import sqrt

import cv2
import glfw
import numpy as np
from OpenGL.GL import GL_POLYGON
from pyglui import ui
from pyglui.cygl.utils import draw_points, draw_polyline, RGBA

from square_marker_detect import detect_markers, detect_markers_robust, draw_markers, m_marker_to_screen
from file_methods import Persistent_Dict
from methods import normalize, denormalize
from plugin import Plugin
from reference_surface import Reference_Surface

logger = logging.getLogger(__name__)


class Surface_Tracker(Plugin):
    icon_chr = chr(0xec07)
    icon_font = 'pupil_icons'

    @property
    def pretty_class_name(self):
        return 'Detector de superfícies'

    def __init__(self, g_pool, mode="Mostrar marcadores e superfícies", min_marker_perimeter=100, invert_image=False,
                 robust_detection=True):
        super().__init__(g_pool)
        self.order = .2

        # all markers that are detected in the most recent frame
        self.markers = []

        self.load_surface_definitions_from_file()

        self.g_pool.surface_tracker = self

        # edit surfaces
        self.edit_surfaces = []
        self.edit_surf_verts = []
        self.marker_edit_surface = None

        # plugin state
        self.mode = mode
        self.running = True

        self.robust_detection = robust_detection
        self.aperture = 11
        self.min_marker_perimeter = min_marker_perimeter
        self.min_id_confidence = 0.0
        self.locate_3d = False
        self.invert_image = invert_image
        self.marker_real_perimeter = 1.2

        self.img_shape = None
        self._last_mouse_pos = 0, 0

        self.menu = None
        self.button = None
        self.add_button = None

    def load_surface_definitions_from_file(self):
        # all registered surfaces
        self.surface_definitions = Persistent_Dict(os.path.join(self.g_pool.user_dir, 'definicoes_superficies'))
        self.surfaces = [Reference_Surface(self.g_pool, saved_definition=d) for d in
                         self.surface_definitions.get('realtime_square_marker_surfaces', [])]

        if 'min_marker_perimeter' in self.surface_definitions:
            self.min_marker_perimeter = self.surface_definitions['min_marker_perimeter']

    def save_surface_definitions_to_file(self):
        self.surface_definitions["realtime_square_marker_surfaces"] = [rs.save_to_dict() for rs in self.surfaces if
                                                                       rs.defined]

        self.surface_definitions['min_marker_perimeter'] = self.min_marker_perimeter
        self.surface_definitions.save()

    def on_notify(self, notification):
        if notification['subject'] == 'surfaces_changed':
            logger.info('Surfaces changed')
            self.save_surface_definitions_to_file()

    def on_pos(self, pos):
        self._last_mouse_pos = normalize(pos, self.g_pool.capture.frame_size, flip_y=True)

    def on_click(self, pos, button, action):
        print(pos)
        if self.mode == 'Mostrar marcadores e superfícies':
            if action == glfw.PRESS:
                for s in self.surfaces:
                    toggle = s.get_mode_toggle(pos, self.img_shape)
                    if toggle == 'surface_mode':
                        if s in self.edit_surfaces:
                            self.edit_surfaces.remove(s)
                        else:
                            self.edit_surfaces.append(s)
                    elif toggle == 'marker_mode':
                        if self.marker_edit_surface == s:
                            self.marker_edit_surface = None
                        else:
                            self.marker_edit_surface = s

            if action == glfw.RELEASE:
                if self.edit_surf_verts:
                    # if we had draged a vertex lets let other know the surfaces changed.
                    # self.notify_all({'subject': 'surfaces_changed', 'delay': 2})
                    self.save_surface_definitions_to_file()
                self.edit_surf_verts = []

            elif action == glfw.PRESS:
                surf_verts = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
                x, y = pos

                for s in self.edit_surfaces:
                    if s.detected and s.defined:
                        for (vx, vy), i in zip(s.ref_surface_to_img(np.array(surf_verts)), range(4)):
                            vx, vy = denormalize((vx, vy), (self.img_shape[1], self.img_shape[0]), flip_y=True)
                            if sqrt((x - vx) ** 2 + (y - vy) ** 2) < 15:  # img pixels
                                self.edit_surf_verts.append((s, i))
                                return

                if self.marker_edit_surface:
                    for m in self.markers:
                        if m['perimeter'] >= self.min_marker_perimeter:
                            vx, vy = m['centroid']
                            if sqrt((x - vx) ** 2 + (y - vy) ** 2) < 15:
                                if m['id'] in self.marker_edit_surface.markers:
                                    self.marker_edit_surface.remove_marker(m)
                                    # self.notify_all({'subject':'surfaces_changed','delay':1})
                                else:
                                    self.marker_edit_surface.add_marker(m, self.markers, self.min_marker_perimeter,
                                                                        self.min_id_confidence)
                                    # self.notify_all({'subject':'surfaces_changed','delay':1})

    def add_surface(self, _):
        surf = Reference_Surface(self.g_pool)
        surf.on_finish_define = self.save_surface_definitions_to_file
        self.surfaces.append(surf)
        self.update_gui_markers()

    def remove_surface(self, i):
        remove_surface = self.surfaces[i]
        if remove_surface == self.marker_edit_surface:
            self.marker_edit_surface = None
        if remove_surface in self.edit_surfaces:
            self.edit_surfaces.remove(remove_surface)

        self.surfaces[i].cleanup()
        del self.surfaces[i]
        self.update_gui_markers()
        # self.notify_all({'subject': 'surfaces_changed'})

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Detector de superfície'

        self.button = ui.Thumb('running', self, label='S', hotkey='s')
        self.button.on_color[:] = (.1, .2, 1., .8)
        self.g_pool.quickbar.append(self.button)
        self.add_button = ui.Thumb('add_surface', setter=self.add_surface, getter=lambda: False, label='A', hotkey='a')
        self.g_pool.quickbar.append(self.add_button)
        self.update_gui_markers()

    def deinit_ui(self):
        self.g_pool.quickbar.remove(self.button)
        self.button = None
        self.g_pool.quickbar.remove(self.add_button)
        self.add_button = None
        self.remove_menu()

    def update_gui_markers(self):
        self.menu.elements[:] = []
        self.menu.append(ui.Info_Text(
            'Esse plugin detecta e mapeia marcadores visíveis na cena. A superfície pode ser definida com 1 ou mais marcadores clicando em *Adicionar superfície*. Superfícies podem ser alteradas clicando em *Modo de edição de superfícies*.'))
        self.menu.append(ui.Switch('robust_detection', self, label='Detecção robusta'))
        self.menu.append(ui.Switch('invert_image', self, label='Usar marcadores invertidos'))
        self.menu.append(
            ui.Slider('min_marker_perimeter', self, step=1, min=10, max=100, label="Perímetro de marcador mínimo"))
        self.menu.append(ui.Switch('locate_3d', self, label='Localização 3D'))
        self.menu.append(ui.Selector('mode', self, label="Modo",
                                     selection=['Mostrar marcadores e superfícies', 'Mostrar IDs dos marcadores']))
        self.menu.append(ui.Button("Adicionar Superfície", lambda: self.add_surface('_'), ))

        # avanced_menu = ui.Growing_Menu("Configurações avançadas")
        # avanced_menu.collapsed = True
        # avanced_menu.append(ui.Info_Text(
        #     'Gerencia parâmetros para dedução do tamanho das superfíceis. O perímetro real do marcador deve ser expresso em centímetros. Através dele é que a inferência é feita'))
        # avanced_menu.append(
        #     ui.Slider('marker_real_perimeter', self, step=0.1, min=0.1, max=8.1, label="Perímetro real de marcador"))
        #
        # self.menu.append(avanced_menu)

        for s in self.surfaces:
            idx = self.surfaces.index(s)
            s_menu = ui.Growing_Menu("Superfície {}".format(idx))
            s_menu.collapsed = True
            s_menu.append(ui.Text_Input('name', s))
            s_menu.append(ui.Text_Input('x', s.real_world_size, label='X size'))
            s_menu.append(ui.Text_Input('y', s.real_world_size, label='Y size'))
            # s_menu.append(ui.Text_Input('gaze_history_length', s, label='Gaze History Length [seconds]'))
            s_menu.append(ui.Button('Abrir janela de visualização', s.open_close_window))
            s_menu.append(ui.Button('Salvar frame', s.save_frame))

            # closure to encapsulate idx
            def make_remove_s(i):
                return lambda: self.remove_surface(i)

            remove_s = make_remove_s(idx)
            s_menu.append(ui.Button('Remover', remove_s))

            self.menu.append(s_menu)

        self.menu.append(ui.Button("Salvar definições", self.save_surface_definitions_to_file, ))

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return

        self.img_shape = frame.height, frame.width, 3

        if self.running:
            gray = frame.gray
            if self.invert_image:
                gray = 255 - gray

            if self.robust_detection:
                self.markers = detect_markers_robust(
                    gray, grid_size=5, aperture=self.aperture,
                    prev_markers=self.markers,
                    true_detect_every_frame=3,
                    min_marker_perimeter=self.min_marker_perimeter)
            else:
                self.markers = detect_markers(
                    gray, grid_size=5, aperture=self.aperture,
                    min_marker_perimeter=self.min_marker_perimeter)

            if self.mode == "Mostrar IDs dos marcadores":
                draw_markers(frame.gray, self.markers)

        # locate surfaces, map gaze
        for s in self.surfaces:
            s.locate(self.markers, self.min_marker_perimeter, self.min_id_confidence, self.locate_3d)
            # if s.detected:
            #     s.gaze_on_srf = s.map_data_to_surface(events.get('gaze_positions',[]),s.m_from_screen)
            #     s.update_gaze_history()
            # else:
            #     s.gaze_on_srf =[]

        events['surfaces'] = []
        for s in self.surfaces:
            if s.detected:
                events['surfaces'].append({'name': s.name, 'uid': s.uid, 'm_to_screen': s.m_to_screen.tolist(),
                                           'm_from_screen': s.m_from_screen.tolist(), 'gaze_on_srf': s.gaze_on_srf,
                                           'timestamp': frame.timestamp,
                                           'camera_pose_3d': s.camera_pose_3d.tolist() if s.camera_pose_3d is not None else None})

        if self.running:
            self.button.status_text = '{}/{}'.format(len([s for s in self.surfaces if s.detected]), len(self.surfaces))
        else:
            self.button.status_text = 'Detector pausado'

        if self.mode == 'Mostrar marcadores e superfícies':
            # edit surfaces by user
            if self.edit_surf_verts:
                pos = self._last_mouse_pos
                for s, v_idx in self.edit_surf_verts:
                    if s.detected:
                        new_pos = s.img_to_ref_surface(np.array(pos))
                        s.move_vertex(v_idx, new_pos)

    def get_init_dict(self):
        return {'mode': self.mode, 'min_marker_perimeter': self.min_marker_perimeter, 'invert_image': self.invert_image,
                'robust_detection': self.robust_detection}

    def gl_display(self):
        """
        Display marker and surface info inside world screen
        """
        if self.mode == "Mostrar marcadores e superfícies":
            for m in self.markers:
                hat = np.array([[[0, 0], [0, 1], [.5, 1.3], [1, 1], [1, 0], [0, 0]]], dtype=np.float32)
                hat = cv2.perspectiveTransform(hat, m_marker_to_screen(m))

                if m['perimeter'] >= self.min_marker_perimeter and m['id_confidence'] > self.min_id_confidence:
                    draw_polyline(hat.reshape((6, 2)), color=RGBA(0.1, 0.1, 1., .5))
                    draw_polyline(hat.reshape((6, 2)), color=RGBA(0.1, 0.1, 1., .3), line_type=GL_POLYGON)
                # elif m['id_confidence'] <= self.min_id_confidence:
                #     draw_polyline(hat.reshape((6, 2)), color=RGBA(1, 0.1, 1., .5))
                #     draw_polyline(hat.reshape((6, 2)), color=RGBA(1, 0.1, 1., .3), line_type=GL_POLYGON)
                else:
                    draw_polyline(hat.reshape((6, 2)), color=RGBA(0.1, 1., 1., .5))

            for s in self.surfaces:
                if s not in self.edit_surfaces and s is not self.marker_edit_surface:
                    s.gl_draw_frame(self.img_shape)

            for s in self.edit_surfaces:
                s.gl_draw_frame(self.img_shape, highlight=True, surface_mode=True)
                s.gl_draw_corners()

            if self.marker_edit_surface:
                inc = []
                exc = []
                for m in self.markers:
                    if m['perimeter'] >= self.min_marker_perimeter:
                        if m['id'] in self.marker_edit_surface.markers:
                            inc.append(m['centroid'])
                        else:
                            exc.append(m['centroid'])
                draw_points(exc, size=20, color=RGBA(1., 0.5, 0.5, .8))
                draw_points(inc, size=20, color=RGBA(0.5, 1., 0.5, .8))
                self.marker_edit_surface.gl_draw_frame(self.img_shape, color=(0.0, 0.9, 0.6, 1.0), highlight=True,
                                                       marker_mode=True)
        #
        # # elif self.mode == 'Show Heatmaps':
        # #     for s in self.surfaces:
        # #         if self.g_pool.app != 'player':
        # #                s.generate_heatmap()
        # #         s.gl_display_heatmap()
        #
        for s in self.surfaces:
            s.gl_display_in_window_()
            # if self.locate_3d:
            #     s.gl_display_in_window_3d(self.g_pool.image_tex)
            # else:
            #     s.gl_display_in_window(self.g_pool.image_tex)
            # pass

    def cleanup(self):
        """ called when the plugin gets terminated.
        This happens either voluntarily or forced.
        if you have a GUI or glfw window destroy it here.
        """
        self.save_surface_definitions_to_file()

        for s in self.surfaces:
            s.cleanup()
