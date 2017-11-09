import functools
import glob
import re
import sys
import serial
import yaml
from typing import Callable
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import math


class Servo(object):
    def __init__(self, nome: str, porta: int,
                 angulo_minimo: float = -45.0, angulo_maximo: float = 45.0,
                 descricao: str = None, velocidade: float = 45.0, callback: Callable = None):
        """
        Instancia o objeto do tipo servo
        :param nome: Nome do servo
        :param porta: Porta na qual o servo está conectado (0 <= porta < 32)
        :param angulo_minimo: Ângulo mínimo de movimento do servo
        :param angulo_maximo: Ângulo máximo de movimento do servo
        :param callback: Função que será disparada quando o ângulo do servo for alterado
        """
        self.nome = nome
        self._porta = porta
        self.descricao = descricao

        # Função chamada quando o ângulo do servo for alterado
        self._callback = callback

        # Variável que diz se o servo foi movido
        self.alterado = True

        self.angulo_maximo = angulo_maximo
        self.angulo_minimo = angulo_minimo

        # Inicia a velocidade para o servo se for especificada
        self.velocidade_angular = velocidade
        # Inicializa o ângulo inicial como 0
        self.angulo = 0

    def __repr__(self):
        return '<Servo {0}{5}: vel={1}°/s pos={2:.2f}° {3:.2f}°..{4:.2f}°>'.format(
            self.nome, self.velocidade_angular,
            self.angulo, self.angulo_minimo, self.angulo_maximo,
            ' (%s)' % self.descricao if self.descricao else ''
        )

    @property
    def posicao(self) -> int:
        """
        :return: Posição absoluta do servo
        """
        return self._pos

    @posicao.setter
    def posicao(self, pos: int):
        """
        Permite a alteração da posição absoluta do servo
        :param pos: Posição absoluta do servo
        """
        assert (self._pos_min <= pos <= self._pos_max), \
            'Posicionamento inválido'

        self._pos = pos

        if self._callback:
            self._callback(self)

    @property
    def angulo(self) -> float:
        """
        :return: Ângulo em graus do servo
        """
        return self.posicao_para_angulo(self._pos)

    @angulo.setter
    def angulo(self, angulo: float):
        """
        Modifica a posição (em graus) do servo
        :param angulo: Ângulo em graus do servo (angulo_minimo <= angulo <= angulo_maximo)
        """
        self.posicao = self.angulo_para_posicao(angulo)

    @property
    def angulo_minimo(self) -> float:
        """
        :return: Ângulo mínimo permitido para o servo
        """
        return self.posicao_para_angulo(self._pos_min)

    @property
    def angulo_maximo(self) -> float:
        """
        :return: Ângulo máximo permitido para o servo
        """
        return self.posicao_para_angulo(self._pos_max)

    @angulo_maximo.setter
    def angulo_maximo(self, angulo: float):
        """
        Permite alterar o ângulo máximo de movimento do servo
        :param angulo: Ângulo máximo (em graus) para movimentação do servo
        :return:
        """
        # assert (-45.0 <= angulo <= 45.0), \
        #     'O ângulo máximo não pode ser inferior a -45 ou superior a 45'

        self._pos_max = self.angulo_para_posicao(angulo)

    @angulo_minimo.setter
    def angulo_minimo(self, angulo: float):
        """
        Permite alterar o ângulo mínimo de movimento do servo
        :param angulo: float
        :return:
        """
        # assert (-45.0 <= angulo <= 45.0), \
        #     'O ângulo mínimo não pode ser inferior a -45 ou superior a 45'

        self._pos_min = self.angulo_para_posicao(angulo)

    @property
    def posicao_maxima(self) -> int:
        """
        Retorna a posição máxima permitida para o servo
        :return: int
        """
        return self._pos_max

    @property
    def velocidade(self) -> int:
        return self._velocidade

    @velocidade.setter
    def velocidade(self, velocidade):
        assert not velocidade or 100 <= velocidade <= 500, \
            "Velocidade deve estar entre 100 e 500"

        self._velocidade = velocidade

    @property
    def velocidade_angular(self) -> float:
        return (0.09 * self._velocidade) if self._velocidade else None

    @velocidade_angular.setter
    def velocidade_angular(self, velocidade: float):
        self.velocidade = int(100 * velocidade / 9) if velocidade else None

    @property
    def posicao_minima(self) -> int:
        """
        Retorna a posição mínima permitida para o servo
        :return: int
        """
        return self._pos_min

    @posicao_minima.setter
    def posicao_minima(self, posicao: int):
        """
        Permite a alteração da posição mínima permitida para o servo

        :param posicao: int
        :return:
        """
        self._pos_min = posicao

    @posicao_maxima.setter
    def posicao_maxima(self, posicao: int):
        """
        Permite a alteração da posição máxima permitida para o servo

        :param posicao: int
        :return:
        """
        self._pos_max = posicao

    @staticmethod
    def angulo_para_posicao(angulo: float) -> int:
        """
        Permite converter um ângulo em graus para o posicionamento do padrão

        posicão = (100 / 9.0) * (angulo + 45) + 500
        500 <= posicão <= 2500
        :param angulo:
        :return:
        """
        assert not (180.0 < angulo + 90 < 0.0), "Angulo inválido"
        # Garante que o ângulo está entre 0 e 360
        # angulo = angulo - (angulo // 360) * 360
        posicao = (100 / 9.0) * (angulo + 90) + 500

        return int(posicao)

    @staticmethod
    def posicao_para_angulo(posicao: int):
        """
        Permite converter o posicionamento do padrão em um ângulo em graus

        ângulo = (posição - 500) * 0.045 + 45
        ângulo < 0 -> rotação para esquerda
        ângulo > 0 -> rotação para esquerda
        :param posicao: int
        :return:
        """

        assert (2500 >= posicao >= 500), "Posição inválida"
        angulo = (posicao - 500) * 0.09 - 90

        return angulo

    @property
    def __comando(self, **kwargs) -> str:
        comando = self.comando
        self.alterado = False
        return comando

    @property
    def comando(self) -> str:
        """
        Retorna o comando para realização do último posicionamento

        Se a última ação já tiver sido realizada não retorna comando
        :param kwargs:
        :return: str
        """
        if self.alterado:
            return '#{0}P{1}{2}'.format(self.porta, self._pos,
                                        ('S%s' % self._velocidade) if self._velocidade else '')
        else:
            return ''

    @comando.setter
    def comando(self, comando: str):
        """
        Permite modificar a comando para o servo
        :param comando: str
        :return:
        """
        parametros = re.match(r'#(?P<porta>\d+)P(?P<posicao>\d+)S(?P<velocidade>\d+)*$',
                              comando).groupdict(None)

        assert parametros, \
            'Comando inválido'

        assert int(parametros['porta']) == self.porta, \
            'Porta inválida para o servo especificado'

        self.posicao = int(parametros['posicao'])
        self.velocidade = int(parametros['velocidade']) if parametros['velocidade'] else None

    @property
    def porta(self):
        return self._porta

    @porta.setter
    def porta(self, porta):
        assert 0 <= porta <= 31, \
            'Porta inválida'

        self._porta = porta

    @property
    def __dict__(self) -> dict:
        definicao = {
            'porta': self.porta,
            'angulo': self.angulo,
            'angulo_minimo': self.angulo_minimo,
            'angulo_maximo': self.angulo_maximo,
        }

        if self.velocidade_angular:
            definicao.update({
                'velocidade_angular': self.velocidade_angular
            })

        if self.descricao:
            definicao.update({'descricao': self.descricao})

        return definicao


class SSC32(object):
    def __init__(self, porta: str=None, taxa_transferencia=115200, autocommit: bool=False):
        """
        Instancia a classe
        :param porta: Porta de conexão serial
        :param taxa_transferencia: Taxa de transferência da comunicação serial
        :param autocommit: Especifica se o movimento ocorrerá de forma automática
        """
        self.autocommit = autocommit
        self._conectado = False

        if not getattr(self, '_servos', None):
            self._servos = list()
            for i in range(0, 32):
                self._servos.append(
                    Servo('%d' % (i+1), i, callback=self._commit)
                )
            self._servos = tuple(self._servos)
        else:
            for servo in self._servos:
                servo._callback = self._commit

        if porta:
            self.conectar(porta, taxa_transferencia)

    def __getitem__(self, chave):
        """
        Retorna um servo com o índice ou nome passado no argumento
        :param chave: Índice da porta ou nome do servo
        :return:
        """

        for servo in self._servos:
            if type(chave) == str and servo.nome.upper() == chave.upper():
                return servo
            elif servo.porta == chave:
                return servo
        raise KeyError(chave)

    def __len__(self):
        """
        :return: Tamanho da lista de servos
        """
        return len(self._servos)

    def conectar(self, porta: str, taxa_transferencia: int=115200):
        """
        Permite a conexão serial
        :param porta: Porta de conexão serial
        :param taxa_transferencia: Taxa de transferência da conexão serial
        :return:
        """
        self._serial = serial.Serial(porta, taxa_transferencia)
        self._conectado = True

    @property
    def conectado(self):
        """
        :return: Estado de conexão
        """
        return self._conectado

    @property
    def conexao_serial(self):
        """
        :return: Objeto de conexão
        """
        return self._serial if self._conectado else None

    @property
    def servos(self) -> tuple:
        """
        :return: Lista de servos do braço
        """
        return self._servos

    def movimentar(self, definicao: list):
        """
        Permite a movimentação
        :param definicao: Lista de comandos de movimentos ->
        (
            (indice do servo, {'velocidade_angular':..., 'angulo':...}),
            ...
        )
        """

        autocommit = self.autocommit
        self.autocommit = False

        for (indice, movimento) in definicao:
            for (atributo, valor) in movimento.items():
                setattr(self.servos[indice], atributo, valor)

        if autocommit:
            self.commit()
            self.autocommit = autocommit

    @property
    def comando(self) -> str:
        """
        :return: Comando para movimentar os servos
        """
        return functools.reduce(lambda x, y: x + y.comando, self.servos, '')

    @comando.setter
    def comando(self, comando: str):
        """
        Permite executar de forma segura um comando personalizado na garra
        :param comando: Comando personalizado
        :return:
        """
        correspondencias = [
            c for c in re.finditer(r'#(?P<porta>\d+)P(?P<posicao>\d+)S(?P<velocidade>\d+)*',
                                   comando)
        ]

        for comando in correspondencias:
            (porta, comando) = int(comando.groupdict()['porta']), \
                               comando.group()

            self[porta].comando = comando

    def _commit(self, *args, commit=False):
        """
        Permite enviar o comando serial para o SSC32
        :param args: Servo que disparou o comando
        :param commit: Indica se o comando deve ser executado mesmo que o autocommit esteja falso
        """
        if self._conectado and (self.autocommit or commit):
            self._serial.write(self.__comando)

    def commit(self):
        """
        Envia o comando para porta serial
        """
        self._commit(commit=True)

    @property
    def __comando(self) -> str:
        """
        Modifica a flag dos servos
        :return: Comando para movimentar os servos
        """
        return functools.reduce(lambda x, y: x + y._Servo__comando, self.servos, '')

    @classmethod
    def portas_disponiveis(cls) -> list:
        """
        Lista as portas seriais disponíveis no sistema
        :raises EnvironmentError: Quando a plataforma não for suportada
        :return: Lista de portas disponíveis
        """
        if sys.platform.startswith('win'):
            portas = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            portas = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            portas = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Plataforma não suportada')

        resultado = []
        for porta in portas:
            try:
                s = serial.Serial(porta)
                s.close()
                resultado.append(porta)
            except (OSError, serial.SerialException):
                pass
        return resultado

    def importar_configuracao(self, arquivo: str):
        configuracoes = yaml.load(open(arquivo))
        servos = list()

        for (nome, definicao) in configuracoes['servos'].items():
            servo = Servo(nome, porta=definicao['porta'])

            for chave, valor in definicao.items():
                setattr(servo, chave, valor)

            servos.append(servo)

        self._servos = tuple(servos)

    def exportar_configuracao(self, arquivo):
        definicoes = dict()

        for servo in self._servos:
            definicoes[servo.nome] = servo.__dict__

        yaml.safe_dump({'servos': definicoes}, open(arquivo, 'w'), default_flow_style=False)


class CinematicaMixin(object):
    @property
    def posicao(self) -> tuple:
        return self.cinematica_direta()

    @posicao.setter
    def posicao(self, pos):
        self.cinematica_inversa(pos)

    @property
    def velocidade(self) -> tuple:
        return tuple(np.dot(self.jacobiano(), self.velocidade_de_juntas))

    @velocidade.setter
    def velocidade(self, velocidade: tuple):
        self.velocidade_de_juntas = tuple(np.dot(np.linalg.inv(self.jacobiano()), self.velocidade))

    @property
    @abstractmethod
    def velocidade_de_juntas(self) -> tuple: pass

    @velocidade_de_juntas.setter
    @abstractmethod
    def velocidade_de_juntas(self, velocidade: tuple): pass

    # @posicao.setter
    # def posicao(self, pos: tuple = None):
    #     if pos:
    #         assert (len(pos) == len(self.posicao)), \
    #             'O posicionamento do braço tem de ser definido por {0} variávies'.format(len(self.posicao))
    #
    #         self.x = pos[0]
    #         self.y = pos[1]
    #         self.z = pos[2]
    #         self.phi = pos[3]

    # @property
    # def cinematica_direta(self):
    #     return getattr(self, '_cinematica_direta', None)
    #
    # @property
    # def cinematica_inversa(self):
    #     return getattr(self, '_cinematica_inversa', None)
    #
    # @cinematica_direta.setter
    # def cinematica_direta(self, definicao: Callable):
    #     # self._cinematica_direta = lambda theta_0, theta_1, theta_2, theta_3, theta_4: \
    #     #     theta_0 + theta_1
    #     raise NotImplementedError('É necessário implementar o método')
    #
    # @cinematica_inversa.setter
    # def cinematica_inversa(self, definicao: Callable):
    #     raise NotImplementedError('É necessário implementar o método')
    #
    # @property
    # def jacobiano(self):
    #     return getattr(self, '_jacobiano', None)
    #
    # @jacobiano.setter
    # def jacobiano(self, definicao: Callable):
    #     raise NotImplementedError('É necessário implementar o método')

    @abstractmethod
    def cinematica_direta(self) -> tuple: pass

    @abstractmethod
    def cinematica_inversa(self, pos: tuple): pass

    @abstractmethod
    def jacobiano(self) -> np.ndarray: pass


class Braco(SSC32, CinematicaMixin):
    def __init__(self, *args, **kwargs):
        self._servos = (
            Servo('HS-485HB', 0, angulo_minimo=-90.0, angulo_maximo=90.0,
                  descricao='Servo da base'),
            Servo('HS-805BB', 1, angulo_minimo=-90.0, angulo_maximo=90.0,
                  descricao='Servo do ombro'),
            Servo('HS-755HB', 2, angulo_minimo=-90.0, angulo_maximo=90.0,
                  descricao='Servo da cotovelo'),
            Servo('HS-645MG', 3, angulo_minimo=-90.0, angulo_maximo=90.0,
                  descricao='Servo do punho'),
            Servo('HS-322HD', 4, angulo_minimo=-27.0, angulo_maximo=45.0,
                  descricao='Servo da garra'),
        )

        # self._l5 = 5.6
        self._l5 = 8.5
        self._l4 = 18.6
        self._l3 = 14.7
        self._l2 = 2.0
        self._l1 = 4.2

        super(Braco, self).__init__(*args, **kwargs)

    # @posicao.
    # def posicao(self, pos: tuple = None):
    #     pass

    @property
    def posicao_angular(self):
        return (
            -self.servos[0].angulo, self.servos[1].angulo + 90,
            self.servos[2].angulo - 90, self.servos[3].angulo,
            self.servos[4].angulo
        )


    def movimentar(self, posicao: tuple, velocidade: tuple=None):
        """
        Permite a movimentação
        :param posicao: Posição do servo
        """
        self.posicao = posicao

        autocommit = self.autocommit
        self.autocommit = False

        if autocommit:
            self.commit()
            self.autocommit = autocommit

    @property
    def x(self) -> float:
        """
        :return: Posição x do servo
        """
        return self.posicao[0]

    @x.setter
    def x(self, pos_x):
        self.posicao = (pos_x, self.y, self.z, self.phi)

    @property
    def y(self) -> float:
        """
        :return: Posição y do servo
        """
        return self.posicao[1]

    @y.setter
    def y(self, pos_y):
        self.posicao = (self.x, pos_y, self.z, self.phi)

    @property
    def z(self) -> float:
        """
        :return: Posição z do servo
        """
        return self.posicao[2]

    @z.setter
    def z(self, pos_z):
        self.posicao = (self.x, self.y, pos_z, self.phi)

    @property
    def phi(self) -> float:
        """
        :return: Orientação phi da garra do servo
        """
        return self.posicao[1] + self.posicao[2] + self.posicao[3]

    @phi.setter
    def phi(self, pos_phi):
        self.posicao = (self.x, self.y, self.z, pos_phi)

    @property
    def theta(self) -> float:
        """
        :return: Orientação phi da garra do servo
        """
        return self.posicao[4]

    @theta.setter
    def theta(self, pos_theta):
        self.servos[4].angulo = pos_theta

    @property
    def velocidade_de_juntas(self):
        return tuple([0, 0, 0, 0])

    @velocidade_de_juntas.setter
    def velocidade_de_juntas(self, velocidade: tuple):
        pass

    def cinematica_direta(self) -> tuple:
        angulos = [math.radians(servo.angulo) for servo in self.servos]

        angulos[1] += math.radians(90)
        angulos[2] += math.radians(-90)

        return (
            math.cos(angulos[0]) * (
                math.cos(angulos[1] + angulos[2] + angulos[3]) * self._l5 +
                math.cos(angulos[1] + angulos[2]) * self._l4 +
                math.cos(angulos[1]) * self._l3
            ),

            math.sin(angulos[0]) * (
                math.cos(angulos[1] + angulos[2] + angulos[3]) * self._l5 +
                math.cos(angulos[1] + angulos[2]) * self._l4 +
                math.cos(angulos[1]) * self._l3
            ),

            math.sin(angulos[1] + angulos[2] + angulos[3]) * self._l5 +
            math.sin(angulos[1] + angulos[2]) * self._l4 +
            math.sin(angulos[1]) * self._l3 +
            self._l2 + self._l1,

            math.degrees(angulos[1] + angulos[2] + angulos[3]),

            math.degrees(angulos[4]),
        )

    def _posicionamento_valido(self, pos: tuple):
        t1, t2, t3, t4, t5 = pos

        return \
            (self.servos[0].angulo_minimo <= t1 <= self.servos[0].angulo_maximo) and \
            (self.servos[1].angulo_minimo <= t2 <= self.servos[1].angulo_maximo) and \
            (self.servos[2].angulo_minimo <= t3 <= self.servos[2].angulo_maximo) and \
            (self.servos[3].angulo_minimo <= t4 <= self.servos[3].angulo_maximo) and \
            (self.servos[4].angulo_minimo <= t5 <= self.servos[4].angulo_maximo)

    def _distancia_de_movimento(self, pos: tuple) -> float:
        return np.linalg.norm(np.array(pos)[:-1] - np.array([x.angulo for x in self.servos[:-1]]))

    def cinematica_inversa(self, pos: tuple):
        assert (len(pos) == len(self.posicao)), 'O vetor posição deve ter {0} valores'.format(len(pos))

        # Define uma precisão tolerável os erros de arredondamento
        erro = 0.001

        x, y, z, phi, theta = pos
        # Definição de variáveis auxiliares
        s, t = \
            np.linalg.norm((x, y)) - math.cos(math.radians(phi)) * self._l5, \
            z - math.sin(math.radians(phi)) * self._l5 - self._l1 - self._l2

        u = (s**2 + t**2 - self._l4**2 + self._l3**2) / (2 * self._l3)

        sen_1 = (y / math.sqrt(x**2 + y**2), )
        cos_1 = (x / math.sqrt(x**2 + y**2), )

        cos_3 = ((s ** 2 + t ** 2 - self._l4 ** 2 - self._l3 ** 2) / (2 * self._l3 * self._l4), )
        sen_2 = ((2 * t * u + math.sqrt(max((2 * t * u)**2 - 4*(t**2 + s**2)*(u**2 - s**2), 0))) / (2 * (t**2 + s**2)), (2 * t * u - math.sqrt(max((2 * t * u)**2 - 4*(t**2 + s**2)*(u**2 - s**2), 0))) / (2 * (t**2 + s**2)))

        # Executa testes de arredondamento
        for valor in sen_1 + cos_1 + cos_3 + sen_2:
            if abs(valor) - erro > 1:
                raise ValueError('Posicionamento inválido')

        cos_3 = [(-1 if c < 0 else 1) * min(abs(c), 1) for c in cos_3]
        sen_2 = [(-1 if s < 0 else 1) * min(abs(s), 1) for s in sen_2]

        sen_3 = (math.sqrt(1 - cos_3[0] ** 2), -math.sqrt(1 - cos_3[0] ** 2))
        cos_2 = (math.sqrt(1 - sen_2[0] ** 2), math.sqrt(1 - sen_2[1] ** 2))

        theta_1 = np.arctan2(sen_1, cos_1)
        theta_2 = np.arctan2(sen_2, cos_2)
        theta_3 = np.arctan2(sen_3, cos_3)

        solucoes = [[math.degrees(t1), math.degrees(t2), math.degrees(t3), phi - math.degrees(t2) - math.degrees(t3), theta]
                    for t3 in theta_3 for t2 in theta_2 for t1 in theta_1]

        # Obtém quais são as configurações possíveis
        solucoes_possiveis = list(filter(lambda pos: self._posicionamento_valido(pos), solucoes))
        # print(solucoes_possiveis)

        # Obtém qual a melhor movimentação
        melhor_solucao = solucoes_possiveis[np.argmin([self._distancia_de_movimento(x) for x in solucoes_possiveis])]

        # print(melhor_solucao)
        self.servos[0].angulo = melhor_solucao[0]
        self.servos[1].angulo = melhor_solucao[1]
        self.servos[2].angulo = melhor_solucao[2]
        self.servos[3].angulo = melhor_solucao[3]
        self.servos[4].angulo = melhor_solucao[4]

        # hegrid
        # r, s, t = \
        #     x / math.cos(math.radians(self.servos[0].angulo)) - math.cos(math.radians(phi)) * self._l3, \
        #     y / math.sin(math.radians(self.servos[0].angulo)) - math.cos(math.radians(phi)) * self._l3, \
        #     z - math.sin(math.radians(phi)) * self._l5 - self._l1 - self._l2
        #
        # u = (r**2 + t**2 - self._l4**2 + self._l3**2) / (2 * self._l3)
        #
        # print(math.degrees(
        #     math.atan2(
        #         (r**2 + t**2 - self._l4**2 - self._l3**2) / (2 * self._l3 * self._l4),
        #         math.sqrt(1 - ((r**2 + t**2 - self._l4**2 - self._l3**2) / (2 * self._l3 * self._l4))**2)
        #     )
        # ))
        #
        # self.servos[2].angulo = math.degrees(
        #     math.atan2(
        #         (r**2 + t**2 - self._l4**2 - self._l3**2) / (2 * self._l3 * self._l4),
        #         math.sqrt(1 - ((r**2 + t**2 - self._l4**2 - self._l3**2) / (2 * self._l3 * self._l4))**2)
        #     )
        # )
        #
        # sen_2 = (2 * t * u + math.sqrt(4 * t**2 * u**2 - 4 * (t**2 + r**2) * (u**2 - r**2)) ) /\
        #         (2 * (t**2 + r**2))
        #
        # print(math.degrees(
        #     math.atan2(sen_2, math.sqrt(1 - sen_2))
        # ))
        #
        # self.servos[1].angulo = math.degrees(
        #     math.atan2(sen_2, math.sqrt(1 - sen_2))
        # )
        #
        # self.servos[3].angulo = phi - self.servos[1].angulo - self.servos[2].angulo


    def jacobiano(self):
        pass


# class Hermes(object):
#     def __init__(self, porta, taxa=115200, quantidade=32):
#         ssc = ssc32.SSC32(porta, taxa, quantidade)
