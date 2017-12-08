import logging
import math
import os

import cv2
import numpy as np
from OpenGL.GL import GL_POLYGON
from pyglui import ui
from pyglui.cygl.utils import draw_points_norm, draw_polyline, draw_polyline_norm, RGBA
from scipy.spatial import Delaunay

import glfw
from file_methods import Persistent_Dict
from methods import normalize, denormalize
from object_detect import detect_targets_robust
from plugin import Plugin

from src.modules.sniffer import sniffer
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Tic_Tac_Toe_Position_Screen(Plugin):
    icon_chr = chr(0xe8b8)
    icon_font = 'pupil_icons'

    @property
    def pretty_class_name(self):
        return 'Posição do jogo da velha'

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self.points = [
            [0.0,     1.], [1. / 3,     1.], [2. / 3,     1.], [1.0,     1.],
            [0.0, 2. / 3], [1. / 3, 2. / 3], [2. / 3, 2. / 3], [1.0, 2. / 3],
            [0.0, 1. / 3], [1. / 3, 1. / 3], [2. / 3, 1. / 3], [1.0, 1. / 3],
            [0.0, 0.0], [1. / 3, 0.0], [2. / 3, 0.0], [1.0, 0.0],
        ]

        self.update_rects()

        self.mouse_released = False
        self.edit_mode = True
        self.move_robot = False
        self._id_rect = None
        self.enable_detection = True
        self.draw_contours = True
        self.detection_r, self.detection_g, self.detection_b = 0, 0, 0
        self.calibration_method = 'lapis'
        self.calibration_definitions = [(0, 0, 0, 0) for i in range(16)]
        self.activate_calibration = True
        self.consider_obstacles = False
        self.move_to_target_when_click = True
        self.pencil_height = 8.0
        self.move_robot_phi = 0.0
        self.move_robot_z = 8.0

        self.load_definitions_from_file()

    def pencil_pos(self, pos):
        theta = np.arctan2(
            pos[0] / math.sqrt(pos[0] ** 2 + pos[1]**2),
            pos[1] / math.sqrt(pos[0] ** 2 + pos[1] ** 2)
        )

        return (
            math.cos(theta) * (
                math.sqrt(pos[0] ** 2 + pos[1] ** 2) +
                math.sin(math.radians(pos[3])) * self.pencil_height
            ),
            math.sin(theta) * (
                math.sqrt(pos[0] ** 2 + pos[1] ** 2) +
                math.sin(math.radians(pos[3])) * self.pencil_height
            ),
            pos[2] - math.cos(math.radians(pos[3])) * self.pencil_height
        )

    def update_rects(self):
        self.rects = []

        for i, point in enumerate(self.points):
            if (i + 1) % 4 == 0:
                continue
            if i > 10:
                break
            self.rects.append(
                [self.points[i], self.points[i + 1], self.points[i + 5], self.points[i + 4], self.points[i]])

    def on_click(self, pos, button, action):
        if action == glfw.PRESS:
            print('click pos', pos)
            for i, point in enumerate(self.points):

                if np.linalg.norm(
                    np.array(pos) - np.array(denormalize(point, flip_y=True, size=self.g_pool.capture.frame_size))
                ) <= 10:
                    self._edit_id = i
                    if getattr(self, '_calibration_id', None) != i:
                        self._calibration_id = i
                    else:
                        self._calibration_id = None
                    break
            self.mouse_released = False
            self._last_mouse_click_pos = normalize(pos, self.g_pool.capture.frame_size, flip_y=True)

            print('click', self._last_mouse_click_pos)

            if self.move_robot and self.move_to_target_when_click:
                for i, target in enumerate(self.targets):
                    aux = Delaunay(target)
                    if aux.find_simplex(pos) >= 0:
                        print(i, self._unnormalized_targets[i], np.mean(target, axis=0))
                        self.to_target = True
                        # self._last_mouse_click_pos = normalize(self._unnormalized_targets[i][0][-1::-1], self.g_pool.capture.frame_size, flip_y=True)
                        self.move_to_target_when_click = False
                        break

            self.update_move(self._last_mouse_click_pos)
        elif action == glfw.RELEASE:
            self._edit_id = None
            self.mouse_released = True

    def recent_events(self, events):
        frame = events.get('frame')
        if not frame:
            return

        targets = detect_targets_robust(frame.bgr)
        self._unnormalized_targets = targets
        self.targets = []
        window_shape = self.g_pool.capture.frame_size
        frame_shape = frame.bgr.shape[:-1:]

        fator = (1.0 * window_shape[0] / frame_shape[0], 1.0 * window_shape[1] / frame_shape[1])

        for (center, axes, orientation) in targets:
            rect = []
            for ang in np.arange(0, 2 * np.pi, np.pi / 3):
                x = center[0] + math.sin(ang + math.radians(orientation)) * axes[1] / 2.
                y = center[1] + math.cos(ang + math.radians(orientation)) * axes[0] / 2.

                rect.append([fator[0] * x, fator[1] * y])

            self.targets.append(rect)

        self.img_shape = frame.height, frame.width, 3

    @property
    def perspective_matrices(self):
        response = []
        for i, point in enumerate(self.points):
            if (i + 1) % 4 == 0:
                continue
            if i > 10:
                break

            response.append(
                cv2.getPerspectiveTransform(
                    np.array(
                        [self.pencil_pos(self.calibration_definitions[i])[:-1:],     self.pencil_pos(self.calibration_definitions[i + 1])[:-1:],
                         self.pencil_pos(self.calibration_definitions[i + 5])[:-1:], self.pencil_pos(self.calibration_definitions[i + 4])[:-1:]],
                        np.float32),
                    np.array(
                        [self.points[i],                           self.points[i + 1],
                        self.points[i + 5],                        self.points[i + 4]], np.float32)
                )
            )

        return response

    @property
    def inverse_perspective_matrices(self):
        response = []
        for i, point in enumerate(self.points):
            if (i + 1) % 4 == 0:
                continue
            if i > 10:
                break

            response.append(
                cv2.getPerspectiveTransform(
                    np.array(
                        [self.points[i], self.points[i + 1],
                         self.points[i + 5], self.points[i + 4]], np.float32),
                    np.array(
                        [self.pencil_pos(self.calibration_definitions[i])[:-1:],
                         self.pencil_pos(self.calibration_definitions[i + 1])[:-1:],
                         self.pencil_pos(self.calibration_definitions[i + 5])[:-1:],
                         self.pencil_pos(self.calibration_definitions[i + 4])[:-1:]],
                        np.float32),
                )
            )

        return response

    def draw_targets(self):
        if getattr(self, 'targets', None) and self.draw_contours:
            for rect in self.targets:
                draw_polyline(rect, color=RGBA(0.3, 0.1, 0.5, .8))
                draw_polyline(rect, color=RGBA(0.3, 0.1, 0.5, .8), line_type=GL_POLYGON)

    def norm_pos_to_robot(self, pos):
        pos_e = np.array(normalize(pos, self.g_pool.capture.frame_size, flip_y=True))
        shape = pos_e.shape
        # pos_e.shape = (-1, 1, 2)

        pos_r = None
        poss_g = []
        for i, matrix in enumerate(self.perspective_matrices):
            # pos = cv2.perspectiveTransform(pos_e, matrix)
            # pos.shape = shape

            pos_g = np.array(pos_e, np.float32)
            pos_g.shape = (-1, 1, 2)
            pos_g = cv2.perspectiveTransform(pos_g, self.inverse_perspective_matrices[i])
            pos_g.shape = shape

            aux = Delaunay(self.rects[i])
            if aux.find_simplex(pos_e) >= 0:
                print('entrou')
                pos_r = pos_g
            else:
                poss_g.append(pos_g)

        if len(poss_g) == len(self.rects):
            pos_r = np.mean(poss_g, axis=0)

        return pos_r, len(poss_g) != len(self.rects)


    def draw_pos(self, show_pen=False):
        poss = []
        poss_g = []
        for i, matrix in enumerate(self.perspective_matrices):
            pos = np.array(self.pencil_pos(self.g_pool.braco.posicao)[:-1:], np.float32)

            shape = pos.shape
            pos.shape = (-1, 1, 2)
            pos = cv2.perspectiveTransform(pos, matrix)
            pos.shape = shape

            pos_g = np.array(self.g_pool.braco.posicao[:-2:][::-1], np.float32)
            pos_g.shape = (-1, 1, 2)
            pos_g = cv2.perspectiveTransform(pos_g, matrix)
            pos_g.shape = shape

            aux = Delaunay(self.rects[i])
            if aux.find_simplex(pos) >= 0:
                if show_pen:
                    draw_points_norm([pos], size=20, color=RGBA(0.5, 0.1, 0.3, .8))
                draw_points_norm([pos_g], size=40, color=RGBA(0.5, 0.1, 0.1, .8))
                break
            else:
                poss_g.append(pos_g)
                poss.append(pos)

        if len(poss) == len(self.rects):
            if show_pen:
                draw_points_norm([np.mean(poss, axis=0)], size=20, color=RGBA(0.5, 0.1, 0.3, .8))

            draw_points_norm([np.mean(poss_g, axis=0)], size=40, color=RGBA(0.5, 0.1, 0.1, .8))

    def update_point_definition(self):
        if getattr(self, '_calibration_id', None) is not None:
            self.calibration_definitions[self._calibration_id] = self.g_pool.braco.posicao

        self.update_gui()
        self.menu_definicoes.collapsed = False

    def load_definitions_from_file(self):
        self.tic_tac_toe_definitions = Persistent_Dict(os.path.join(self.g_pool.user_dir, 'definicoes_jogo_da_velha'))

        for name, value in self.tic_tac_toe_definitions.items():
            setattr(self, name, value)

    def save_definitions_to_file(self):
        self.tic_tac_toe_definitions.update({
            'edit_mode': self.edit_mode,
            'enable_detection': self.enable_detection,
            'calibration_method': self.calibration_method,
            'calibration_definitions': self.calibration_definitions,
            'activate_calibration': self.activate_calibration,
            'pencil_height': self.pencil_height,
            'points': self.points,
            'rects': self.rects
        })
        self.tic_tac_toe_definitions.save()

    def update_move(self, pos):
        changed = False

        # if not self.move_robot:
        #     return

        for id, rect in enumerate(self.rects):
            aux = Delaunay(rect)
            if aux.find_simplex(pos) >= 0 and self.move_robot:
                self._id_rect = id
                changed = True

        if not changed and not self.mouse_released and getattr(self, 'to_target', None) is not None:
            self.target_frame_pos = denormalize(pos, self.g_pool.capture.frame_size, flip_y=True)

        if changed and not self.mouse_released:
            self._last_move_point_pos = self._last_mouse_pos
            pos = np.array(self._last_mouse_pos, np.float32)
            shape = pos.shape
            pos.shape = (-1, 1, 2)

            pos = cv2.perspectiveTransform(pos, self.inverse_perspective_matrices[self._id_rect])
            pos.shape = shape
            #
            # if self.move_robot:
            #     )

            if self.move_robot and self.consider_obstacles and self.move_to_target_when_click and changed:
                pos_m = None

                if getattr(self, '_unnormalized_targets', None):
                    target = None
                    definitions = []
                    for i, rect in enumerate(self._unnormalized_targets):
                        pos_r, in_tic_tac_toe_matrix = self.norm_pos_to_robot(np.mean(self.targets[i], axis=0))
                        # print(pos_r, in_tic_tac_toe_matrix)

                        if not in_tic_tac_toe_matrix and not target and rect[0][1] < 300:
                            target = self.targets[i]
                            pos_m = pos_r
                        elif in_tic_tac_toe_matrix:
                            definitions.append(self.targets[i])

                    if target:
                        pin = np.mean(target, axis=0)
                        print("pin", pin)
                        obstacles = []
                        for definition in definitions:
                            obstacles += [np.mean(definition, axis=0)]


                        tab = sniffer(700, 700)

                        # goal
                        xg, yg = denormalize(self._last_mouse_click_pos, flip_y=True, size=self.g_pool.capture.frame_size)
                        tab.setCircularGoal(x0=int(xg), y0=int(yg), r=20)
                        tab.setGoal(x=int(xg), y=int(yg), r=20)

                        for obstacle in obstacles:
                            tab.setCircularObstacle(x0=int(obstacle[0]),
                                                    y0=int(obstacle[1]), r=75)

                        x, y = tab.Sniff(i_max=1000, x0=int(pin[0]), y0=int(pin[1]), step=2, max_step=30)
                        print("Goal found in: x=" + str(x[-1]) + " y=" + str(y[-1]))
                        print("Steps of the way found", len(x))
                        print("Optimizing the way")
                        opx, opy = tab.wayOptimize(x, y)
                        print("Steps of the way found: ", len(opx))

                        print('pos', pos_m)

                        # self.abrir()
                        # self.g_pool.braco.posicao = (pos_m[1], pos_m[0], self.move_robot_z + 4, self.move_robot_phi)
                        #
                        # self.g_pool.braco.posicao = (pos_m[1], pos_m[0], self.move_robot_z, self.move_robot_phi)
                        # self.fechar()
                        #
                        # for i, x in enumerate(opx):
                        #     y = opy[i]
                        #
                        #     pos, _ = self.norm_pos_to_robot(np.array([x, y]))
                        #     try:
                        #         self.g_pool.braco.posicao = (pos[1], pos[0], self.move_robot_z + 3, self.move_robot_phi)
                        #     except Exception as e:
                        #         print(e)


                        from mpl_toolkits.mplot3d import Axes3D
                        # cm = plt.cm.YlOrRd
                        cm = plt.cm.Greys

                        fig = plt.figure(figsize=(14, 8))
                        ax = Axes3D(fig)
                        ax.plot(x, y, 0, 'r--', label="Way", alpha=0.5)
                        ax.plot(opx, opy, 0, 'b', label="Optimized Way", alpha=0.5)
                        ax.plot_surface(tab.X, tab.Y, tab.potential, cmap=cm, alpha=1)
                        ax.set_xlabel('y', fontsize=15)
                        ax.set_ylabel('x', fontsize=15)
                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                        for angle in range(0, 360, 6):
                            ax.view_init(30, angle)
                            plt.draw()
                            plt.pause(.0001)

                        fig = plt.figure(figsize=(14, 8))
                        ax = fig.add_subplot(111)
                        ax.contourf(tab.X, tab.Y, tab.potential, cmap=cm)
                        ax.plot(x, y, 'r--', linewidth=2, label="Way")
                        ax.plot(opx, opy, 'b', linewidth=2, label="Optimized Way")
                        ax.set_xlabel('y')
                        ax.set_ylabel('x')
                        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                        plt.show()

                        return

            if getattr(self, 'to_target', None) is not None:
                self.abrir()
                self.g_pool.braco.posicao = (pos[1], pos[0], self.move_robot_z + 4, self.move_robot_phi)
            else:
                posicao = self.g_pool.braco.posicao
                self.g_pool.braco.posicao = (posicao[0], posicao[1], posicao[2] + 4, posicao[3])

            self.g_pool.braco.posicao = (pos[1], pos[0], self.move_robot_z, self.move_robot_phi)

            if getattr(self, 'to_target', None) is not None:
                self.fechar()
                self.to_target = None

    def abrir(self):
        self.g_pool.braco.servos[4].angulo = self.g_pool.braco.angulo_garra_soltar

    def fechar(self):
        self.g_pool.braco.servos[4].angulo = self.g_pool.braco.angulo_garra_pegar

    def print_detected_values(self):
        if not getattr(self, '_unnormalized_targets', None):
            return

        for rect in self._unnormalized_targets:
            print(rect[0], max(rect[1][0], rect[1][1]) / 2)

    def update_gui(self):
        # self.save_definitions_to_file()
        self.menu.elements[:] = []

        self.menu.append(
            ui.Info_Text('Esse plugin permite o gerenciamento do tabuleiro do jogo da velha, e a movimentação do braço ao efetuar clique'))

        self.menu.append(ui.Switch('edit_mode', self, label='Modo de edição'))
        self.menu.append(ui.Switch('move_robot', self, label='Mover braço ao clicar'))
        self.menu.append(ui.Switch('move_to_target_when_click', self, label='Pegar alvo ao clicar'))
        self.menu.append(ui.Switch('consider_obstacles', self, label='Considerar obstáculos ao mover'))
        self.menu.append(ui.Slider_Text_Input('move_robot_z', self, label='Z'))
        self.menu.append(ui.Slider_Text_Input('move_robot_phi', self, label='φ'))

        menu_objeto = ui.Growing_Menu('Configurações de detecção de objetos')

        menu_objeto.append(ui.Switch('enable_detection', self, label='Detecção habilitada'))
        menu_objeto.append(ui.Switch('draw_contours', self, label='Desenhar contornos'))
        menu_objeto.append(ui.Button('Detectar o objetos', self.print_detected_values))
        # menu_objeto.append(ui.Info_Text('Código RGB aproximado do objeto a ser detectado'))
        # menu_objeto.append(ui.Slider('detection_r', self, min=0, step=1, max=256, label='R'))
        # menu_objeto.append(ui.Slider('detection_g', self, min=0, step=1, max=256, label='G'))
        # menu_objeto.append(ui.Slider('detection_b', self, min=0, step=1, max=256, label='B'))

        menu_objeto.collapsed = True

        menu_calibracao = ui.Growing_Menu('Calibração da posição da garra no tabuleiro')
        menu_calibracao.append(ui.Info_Text('Para definir a calibração basta clicar em algum ponto do tabuleiro e posicionar a garra de tal forma que a ponta do lápis fique em cima do ponto. Depois, clique em "Atualizar definição para o ponto"'))
        menu_calibracao.append(ui.Switch('activate_calibration', self, label='Ativar calibração'))
        menu_calibracao.append(ui.Selector('calibration_method', self, selection=['lapis'], labels=['Lápis'], label='Método de calibração'))
        menu_calibracao.append(ui.Slider_Text_Input('pencil_height', self, label='Altura do lápis'))
        self.menu_definicoes = menu_definicoes = ui.Growing_Menu('Definições')
        menu_definicoes.collapsed = True
        menu_calibracao.append(menu_definicoes)

        for i in range(16):
            menu_definicoes.append(
                ui.Info_Text('P{0}{1} (X, Y, Z) -> ({2:.2f}, {3:.2f}, {4:.2f})'.format(
                    i // 4, i % 4,
                    *self.calibration_definitions[i])[:-1:]
                )
            )

        menu_calibracao.append(ui.Button('Atualizar definição para o ponto', self.update_point_definition))

        self.menu.append(menu_objeto)
        self.menu.append(menu_calibracao)
        self.menu.append(ui.Button('Salvar definições', self.save_definitions_to_file))

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Gerenciador de tabuleiro de jogo da velha'

        self.update_gui()

    def on_char(self, character):
        if character == 'G' or character == 'g':
            self.update_point_definition()
            self.save_definitions_to_file()

    def on_pos(self, pos):
        self._last_mouse_pos = normalize(pos, self.g_pool.capture.frame_size, flip_y=True)

        if getattr(self, '_edit_id', None) is not None and self.edit_mode:
            pos = [self._last_mouse_pos[0], self._last_mouse_pos[1]]
            if pos[0] < 0:
                pos[0] = 0
            if pos[0] > 1:
                pos[0] = 1

            if pos[1] > 1:
                pos[1] = 1
            if pos[1] < 0:
                pos[1] = 0
            self.points[self._edit_id] = pos
            self.update_rects()

        if self.move_robot:
            self.update_move(self._last_mouse_pos)

    def gl_display(self):
        draw_points_norm(self.points, size=20, color=RGBA(0.3, 0.1, 0.5, .8))

        if self.activate_calibration and getattr(self, '_calibration_id', None) is not None:
            draw_points_norm([self.points[self._calibration_id]], size=30, color=RGBA(0.1, 0.5, 0.3, .8))

        for rect in self.rects:
            draw_polyline_norm(rect, color=RGBA(0.3, 0.1, 0.5, .8))

        if self._id_rect is not None and self.move_robot:
            draw_polyline_norm(self.rects[self._id_rect], color=RGBA(0.3, 0.1, 0.5, .2), line_type=GL_POLYGON)
            if getattr(self, '_last_move_point_pos', None):
                draw_points_norm([np.array(self._last_move_point_pos)], size=50, color=RGBA(0.1, 0.5, 0.3, .8))


        self.draw_pos(self.activate_calibration)

        self.draw_targets()