import functools
import glob
import re
import sys
import serial
import yaml
from typing import Callable


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
        self.alterado = True

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
        assert (-45.0 <= angulo <= 45.0), \
            'O ângulo máximo não pode ser inferior a -45 ou superior a 45'

        self._pos_max = self.angulo_para_posicao(angulo)

    @angulo_minimo.setter
    def angulo_minimo(self, angulo: float):
        """
        Permite alterar o ângulo mínimo de movimento do servo
        :param angulo: float
        :return:
        """
        assert (-45.0 <= angulo <= 45.0), \
            'O ângulo mínimo não pode ser inferior a -45 ou superior a 45'

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
        Permite converter um âgulo em graus para o posicionamento do padrão

        posicão = (200 / 9.0) * (angulo + 45) + 500
        500 <= posicão <= 2500
        :param angulo:
        :return:
        """
        assert not (90.0 < angulo + 45 < 0.0), "Angulo inválido"
        posicao = (200 / 9.0) * (angulo + 45) + 500

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
        angulo = (posicao - 500) * 0.045 - 45

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
        parametros = re.match(r'#(?P<porta>\d+)P(?P<posicao>\d+)S*(?P<velocidade>\d+)*$',
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
            definicao.update({
                'descricao': self.descricao
            })

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
        self._serial = serial.Serial(porta, taxa_transferencia, exclusive=True)
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
            # print('{0}\r'.format(self.__comando).encode())
            self._serial.write((self.__comando + '\r').encode())

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

        for (nome, definicao) in configuracoes.items():
            servo = Servo(nome, porta=definicao['porta'])

            for chave, valor in definicao.items():
                setattr(servo, chave, valor)

            servos.append(servo)

        self._servos = tuple(servos)

    def exportar_configuracao(self, arquivo):
        definicoes = dict()

        for servo in self._servos:
            definicoes[servo.nome] = servo.__dict__

        yaml.safe_dump(definicoes, open(arquivo, 'w'), default_flow_style=False)


class Braco(SSC32):
    def __init__(self, *args, **kwargs):
        self._servos = (
            Servo('HS-485HB', 0, angulo_minimo=-45.0, angulo_maximo=40.0,
                  descricao='Servo da base'),
            Servo('HS-805BB', 1, angulo_minimo=-13.5, angulo_maximo=22.5,
                  descricao='Servo do ombro'),
            Servo('HS-755HB', 2, angulo_minimo=-18.0, angulo_maximo=27.0,
                  descricao='Servo da cotovelo'),
            Servo('HS-645MG', 3, angulo_minimo=-45.0, angulo_maximo=45.0,
                  descricao='Servo do punho'),
            Servo('HS-322HD', 4, angulo_minimo=-9.0, angulo_maximo=40.5,
                  descricao='Servo da garra'),
        )

        super(Braco, self).__init__(*args, **kwargs)

# class Hermes(object):
#     def __init__(self, porta, taxa=115200, quantidade=32):
#         ssc = ssc32.SSC32(porta, taxa, quantidade)
