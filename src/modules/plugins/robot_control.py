# logging
import logging
import os

from pyglui import ui

from hermes import Braco
from plugin import Plugin

from file_methods import Persistent_Dict
import numpy as np
logger = logging.getLogger(__name__)


class Robot_Control(Plugin):
    icon_chr = chr(0xe8b8)
    icon_font = 'pupil_icons'

    @property
    def pretty_class_name(self):
        return 'Gerenciador de braço robótico'

    def __init__(self, g_pool):
        super().__init__(g_pool)

        self.port = None

        if not getattr(self.g_pool, 'braco', None):
            self.braco = Braco(porta=None, autocommit=True)
            self.load_braco_definitions_from_file()

            self.g_pool.braco = self.braco
        else:
            self.braco = g_pool.braco

        # self.braco._conectado = True

        self._connected = False
        self.autocommit = False

        self.calibration_method = 'lapis'
        self.calibration_x, self.calibration_y, self.calibration_z = (0., 0., 0.)

        self.sliders_min = []
        self.sliders_max = []
        self.sliders_ang = []
        self.pos_calibracao = []

    def load_braco_definitions_from_file(self):
        self.braco_definitions = Persistent_Dict(os.path.join(self.g_pool.user_dir, 'definicoes_braco'))

        self.braco.importar_configuracao_dicionario(self.braco_definitions)

    def save_braco_definitions_to_file(self):
        self.braco_definitions.update(
            self.braco.__dict__
        )
        self.braco_definitions.save()

    @property
    def connected(self):
        return self.braco.conectado

    @connected.setter
    def connected(self, value):
        if not value and self.braco.conectado:
            self._connected = False
        elif value and self.braco.conectado:
            self._connected = True
        else:
            self._connected = False

    def init_ui(self):
        self.add_menu()
        self.menu.label = 'Gerenciador de braço robótico'

        self.button = ui.Thumb('connected', self, label='R', hotkey='r')
        self.button.on_color[:] = (.1, .2, 1., .8)
        self.g_pool.quickbar.append(self.button)

        self.menu.append(
            ui.Info_Text('Esse plugin permite a conexão com um gerenciador SSC32 para controle de servo motores'))

        self.update_gui()

    def gl_display(self):
        super(Robot_Control, self).gl_display()
        self.update_sliders()

    def deinit_ui(self):
        self.g_pool.quickbar.remove(self.button)
        self.button = None
        self.remove_menu()

    def update_sliders(self):
        for i, servo in enumerate(self.braco.servos):
            try:
                self.sliders_min[i].maximum = servo.angulo_variacao_maximo - 0.05
                self.sliders_max[i].minimum = servo.angulo_variacao_minimo
                self.sliders_ang[i].minimum = servo.angulo_minimo
                self.sliders_ang[i].maximum = servo.angulo_maximo - 0.05
            except Exception as e:
                pass

    def get_x(self):
        x = getattr(self, 'x', None)
        return (x if x is not None else self.braco.x) if not self.autocommit else self.braco.x

    def get_y(self):
        y = getattr(self, 'y', None)
        return (y if y is not None else self.braco.y) if not self.autocommit else self.braco.y

    def get_z(self):
        z = getattr(self, 'z', None)
        return (z if z is not None else self.braco.z) if not self.autocommit else self.braco.z

    def get_phi(self):
        # print(self.braco.posicao_angular)
        phi = getattr(self, 'phi', None)
        return (phi if phi is not None else self.braco.phi) if not self.autocommit else self.braco.phi

    def set_x(self, x):
        self._modified = True
        if self.autocommit:
            try:
                self.braco.x = x
            except Exception as e:
                logger.error(e)
        else:
            self.x = x

    def set_y(self, y):
        self._modified = True
        if self.autocommit:
            try:
                self.braco.y = y
            except Exception as e:
                logger.error(e)
        else:
            self.y = y

    def set_z(self, z):
        self._modified = True
        if self.autocommit:
            try:
                self.braco.z = z
            except Exception as e:
                logger.error(e)
        else:
            self.z = z

    def set_phi(self, phi):
        self._modified = True
        if self.autocommit:
            try:
                self.braco.phi = phi
            except Exception as e:
                logger.error(e)
        else:
            self.phi = phi

    def add_pos_calibracao(self):
        self.pos_calibracao.append((
            (self.calibration_x, self.calibration_y, self.calibration_z),
            self.braco.posicao_angular
        ))

        self.update_gui()

    def obter_constantes(self):
        if not self.pos_calibracao:
            return

        A, B = list(), list()

        for pos in self.pos_calibracao:
            A.append(self.braco.coeficientes_cinematica_direta(pos[1]))
            B.append(pos[0])

        self.braco._l1, \
        self.braco._l2, \
        self.braco._l3, \
        self.braco._l4, \
        self.braco._l5, = self.braco.encontrar_constantes(np.array(A), np.array(B))

    def connect(self, port):
        self.braco.conectar(port)
        # self.braco.movimentar_posicao_repouso()
        # self.braco.commit()

    def update_gui(self):
        self.menu.elements[:] = []
        self.sliders_min = []
        self.sliders_max = []
        self.sliders_ang = []

        # We add the capture selection menu
        self.menu.append(ui.Selector(
            'port',
            setter=self.connect,
            getter=lambda: self.port,
            selection=[None] + self.braco.portas_disponiveis(),
            labels=['Selecionar porta'] + [b for b in self.braco.portas_disponiveis()],
            label='Porta serial'
        ))

        servos_menu = ui.Growing_Menu('Definições dos servo-motores')
        servos_menu.collapsed = True

        ang_vel = ui.Growing_Menu('Angulação e velocidade dos servos')

        for servo in self.braco.servos:
            servo_menu = ui.Growing_Menu(servo.descricao)
            servo_menu.collapsed = True
            servo_menu.append(ui.Info_Text('Definição dos atributos estáticos'))
            servo_menu.append(ui.Text_Input('descricao', servo, label='Descrição'))
            servo_menu.append(ui.Slider('porta', servo, step=1, min=0, max=31, label="Porta"))

            servo_menu.append(ui.Switch('angulo_invertido', servo, label='Ângulo invertido'))
            servo_menu.append(
                ui.Slider('angulo_correcao', servo, min=-180, step=0.05, max=180 + 0.05, label='Ângulo de correção'))

            slider_min = ui.Slider('angulo_variacao_minimo', servo, step=0.05, min=-90,
                                   max=servo.angulo_variacao_maximo - 0.05,
                                   label="Âgulo mínimo de variação")
            slider_max = ui.Slider('angulo_variacao_maximo', servo, step=0.05, min=servo.angulo_variacao_minimo + 0.05,
                                   max=90.05,
                                   label="Âgulo máximo de variação")

            self.sliders_min.append(slider_min)
            self.sliders_max.append(slider_max)

            servo_menu.append(slider_min)
            servo_menu.append(slider_max)

            servo_menu.append(ui.Info_Text('Definição dos atributos dinâmicos'))

            slider_ang = ui.Slider('angulo', servo, step=0.05, min=servo.angulo_minimo, max=servo.angulo_maximo + 0.05,
                                   label="Ângulo")
            servo_menu.append(slider_ang)
            slider_vel = ui.Slider('velocidade_angular', servo, step=0.09, min=9, max=45.09, label="Velocidade")
            servo_menu.append(slider_vel)

            self.sliders_ang.append(slider_ang)

            servos_menu.append(servo_menu)

            ang_vel.append(ui.Info_Text(servo.descricao))
            ang_vel.append(slider_ang)
            ang_vel.append(slider_vel)

        ang_vel.collapsed = True

        braco_menu = ui.Growing_Menu('Definições do braço robótico')
        braco_menu.append(
            ui.Info_Text('Nessas configurações é possível definir os atributos padrões e dinâmicos do braço robótico'))
        braco_menu.append(ui.Switch('autocommit', self.braco, label='Movimento automático'))

        menu_contantes = ui.Growing_Menu('Constantes da cinemática')
        menu_contantes.collapsed = True
        menu_contantes.append(ui.Slider_Text_Input('_l1', self.braco, label='L1'))
        menu_contantes.append(ui.Slider_Text_Input('_l2', self.braco, label='L2'))
        menu_contantes.append(ui.Slider_Text_Input('_l3', self.braco, label='L3'))
        menu_contantes.append(ui.Slider_Text_Input('_l4', self.braco, label='L4'))
        menu_contantes.append(ui.Slider_Text_Input('_l5', self.braco, label='L5'))

        menu_posicao = ui.Growing_Menu('Posição em coordenadas da base')
        menu_posicao.append(ui.Switch('autocommit', self, label='Envio automático'))
        menu_posicao.append(ui.Switch('valid', self, label='Posição válida', setter=lambda x: None))

        menu_calibracao = ui.Growing_Menu('Calibração da cinemática direta')
        menu_calibracao.append(ui.Info_Text('Através desse menu é possível efetuar a calibração das constantes da cinemática direta.'))

        menu_calibracao.append(ui.Selector('calibration_method', self, setter=lambda x: None, selection=['lapis'], labels=['Lápis'],
                            label='Método de calibração'))
        # menu_calibracao.append(ui.Info_Text(
        #     '''Através da garra segure o lápis e indique um ponto (definido nas coordenadas da base) para o qual o movimento será feito. Então, usando os controles, mova o braço robótico até o ponto indicado e clique em "Adicionar posição". Repita o processo variando o ponto até que exista uma quantidade satisfatória (note que para o mesmo ponto é possível mover o braço em múltiplas configurações). Finalmente clique em "Otimizar variáveis de cinemática".'''
        # ))

        menu_calibracao.append(ui.Info_Text('Posicionamento em coordenadas da base esperado'))
        menu_calibracao.append(ui.Separator())
        menu_calibracao.append(ui.Slider_Text_Input('calibration_x', self, label='X'))
        menu_calibracao.append(ui.Slider_Text_Input('calibration_y', self, label='Y'))
        menu_calibracao.append(ui.Slider_Text_Input('calibration_z', self, label='Z'))

        menu_calibracao.append(menu_contantes)
        if self.pos_calibracao:
            menu_posicoes = ui.Growing_Menu('Posições definidas')

            for i, pos in enumerate(self.pos_calibracao):
                menu = ui.Growing_Menu('Posição {}'.format(i))
                menu.collapsed = True
                menu.append(
                    ui.Info_Text('(X, Y, Z) = ({}, {}, {})'.format(*pos[0])))
                menu.append(
                    ui.Info_Text('(θ1, θ2, θ3, θ4) = ({0:.2f}, {1:.2f}, {2:.2f}, {3:.2f})'.format(*pos[1])))

                def remover():
                    self.pos_calibracao.remove(pos)
                    self.update_gui()

                menu.append(ui.Button('Remover', remover))
                menu_posicoes.append(menu)

            menu_calibracao.append(menu_posicoes)
        menu_calibracao.append(ui.Button('Adicionar posição', self.add_pos_calibracao))
        menu_calibracao.append(ui.Button('Obter constantes', self.obter_constantes))

        braco_menu.append(menu_calibracao)

        menu_garra = ui.Growing_Menu('Configuações da garra')
        menu_garra.append(ui.Info_Text(
            "Esse menu gerencia os ângulos de abertura (utilizado para \"soltar\" o objeto) e fechamento da garra (que é utilizado para \"pegar\" um objeto)"))
        menu_garra.append(ui.Slider('angulo_garra_pegar', self.braco,
                                    min=self.braco.servos[-1].angulo_minimo,
                                    max=self.braco.servos[-1].angulo_maximo + 0.05,
                                    step=0.05,
                                    label='Ângulo de pegar'))
        menu_garra.append(ui.Slider('angulo_garra_soltar', self.braco,
                                    min=self.braco.servos[-1].angulo_minimo,
                                    max=self.braco.servos[-1].angulo_maximo + 0.05,
                                    step=0.05,
                                    label='Ângulo de soltar'))

        def abrir():
            self.g_pool.braco.servos[4].angulo = self.braco.angulo_garra_soltar

        def fechar():
            self.g_pool.braco.servos[4].angulo = self.braco.angulo_garra_pegar

        menu_garra.append(self.sliders_ang[-1])

        menu_garra.append(ui.Button('Abrir garra', abrir))
        menu_garra.append(ui.Button('Fechar garra', fechar))

        self.pos_sliders = [
            ui.Slider_Text_Input('x', self.braco, label='X', getter=self.get_x, setter=self.set_x),
            ui.Slider_Text_Input('y', self.braco, label='Y', getter=self.get_y, setter=self.set_y),
            ui.Slider_Text_Input('z', self.braco, label='Z', getter=self.get_z, setter=self.set_z),
            ui.Slider_Text_Input('phi', self.braco, label='φ', getter=self.get_phi, setter=self.set_phi),
        ]

        menu_posicao.append(self.pos_sliders[0])
        menu_posicao.append(self.pos_sliders[1])
        menu_posicao.append(self.pos_sliders[2])
        menu_posicao.append(self.pos_sliders[3])

        braco_menu.append(ang_vel)
        braco_menu.append(menu_posicao)
        menu_garra.collapsed = True
        braco_menu.append(menu_garra)

        braco_menu.append(ui.Button('Efetuar movimento', self.commit))
        braco_menu.append(ui.Button('Definir posição atual como de repouso', self.definir_posicao_retorno))
        braco_menu.append(ui.Button('Movimentar para posição de repouso', self.braco.movimentar_posicao_repouso))

        braco_menu.append(ui.Button('Salvar definições', self.save_braco_definitions_to_file))

        self.menu.elements.append(servos_menu)
        self.menu.elements.append(braco_menu)

    @property
    def valid(self):
        if getattr(self, '_modified', True):
            self._valid_cache = self.braco.simular_posicao((self.get_x(), self.get_y(), self.get_z(), self.get_phi()))
            self._modified = False
        return self._valid_cache

    def commit(self):
        try:
            self.braco.posicao = (self.get_x(), self.get_y(), self.get_z(), self.get_phi())
            self.x, self.y, self.z, self.phi = [None] * 4
            self._modified = False
        except:
            self._valid_cache = False

    def definir_posicao_retorno(self):
        self.braco.posicao_repouso = self.braco.posicao
