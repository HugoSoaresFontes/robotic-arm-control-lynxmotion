import functools, re


class Servo(object):
    def __init__(self, nome: str, porta: int, \
                 angulo_minimo: float=-45.0, angulo_maximo: float=45.0, \
                 descricao: str=None, velocidade: int=None):
        """
        Instancializa o objeto do tipo servo
        :param nome: str
        :param porta: int
        :param angulo_minimo: float
        :param angulo_maximo: float
        """
        self.nome = nome
        self.porta = porta
        self.descricao = descricao

        # Variável que diz se o servo foi movido
        self.alterado = True

        self.angulo_maximo = angulo_maximo
        self.angulo_minimo = angulo_minimo

        # Inicializa o âgulo inicial como 0
        self.angulo = 0
        # Inicia a velocidade para o servo se for especificada
        self.velocidade = velocidade

    def __repr__(self):
        return '<Servo {0}{5}: vel={1} pos={2:.2f}° {3:.2f}°..{4:.2f}°>'.format(
            self.nome, self.velocidade,
            self.angulo, self.angulo_minimo, self.angulo_maximo,
            ' (%s)' % self.descricao if self.descricao else ''
        )

    @property
    def posicao(self) -> int:
        """
        Retorna a posição absoluta do servo
        :return: int
        """
        return self._pos

    @posicao.setter
    def posicao(self, pos: int):
        """
        Permite a alteração da posição absoluta do servo
        :param pos:
        :return:
        """
        assert (self._pos_min <= pos <= self._pos_max), \
            'Posicionamento inválido'

        self._pos = pos

    @property
    def angulo(self) -> float:
        """
        Permite a alteração do ângulo (em graus) do servo
        :return:
        """
        return self.posicao_para_angulo(self._pos)

    @angulo.setter
    def angulo(self, angulo: float):
        """
        Modifica a posição (em graus) do servo
        :return:
        """
        self.posicao = self.angulo_para_posicao(angulo)

    @property
    def angulo_minimo(self) -> float:
        """
        Retorna o ângulo mínimo permitido para o servo
        :return: float
        """
        return self.posicao_para_angulo(self._pos_min)

    @property
    def angulo_maximo(self) -> float:
        """
        Retorna o ângulo máximo permitido para o servo
        :return: float
        """
        return self.posicao_para_angulo(self._pos_max)


    @angulo_maximo.setter
    def angulo_maximo(self, angulo: float):
        """
        Permite alterar o ângulo máximo de movimento do servo
        :param angulo: float
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

    def angulo_para_posicao(self, angulo: float) -> int:
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

    def posicao_para_angulo(self, posicao: int):
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
                                        '%s' % self._velocidade if self._velocidade else '')
        else:
            return ''

    @comando.setter
    def comando(self, comando: str):
        """
        Permite modificar a comando para o servo
        :param comando: str
        :return:
        """
        parametros = re.match(r'#(?P<porta>\d+)P(?P<posicao>\d+)(?P<velocidade>\d+)*$', \
            comando).groupdict(None)

        assert parametros, \
            'Comando inválido'

        assert parametros['porta'] == self.porta, \
            'Porta inválida para o servo especificado'

        self.posicao = parametros['posicao']
        self.velocidade = parametros['velocidade']


class Braco(object):
    def __init__(self, auto_commit: bool=False):
        """
        Instancializa o braco
        :param auto_commit: bool
        """
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

    def movimentar(self, definicao: list):
        """
        Permite a movimentação
        :param definicao:
        :return:
        """
        for (indice, movimento) in definicao:
            for (atributo, valor) in movimento.items():
                setattr(self.servos[indice], atributo, valor)

    @property
    def servos(self) -> tuple:
        """
        Retorna a lista de servos do braco
        :return: tuple
        """
        return self._servos

    @property
    def comando(self) -> str:
        """
        Retorna o comando para movimentar a garra
        :return: str
        """
        return functools.reduce(lambda x, y: x + y.comando, self.servos,
                                self.servos[0].comando)

    @property
    def __comando(self) -> str:
        return functools.reduce(lambda x, y: x + y._Servo__comando, self.servos,
                                self.servos[0]._Servo__comando)


# class Hermes(object):
#     def __init__(self, porta, taxa=115200, quantidade=32):
#         ssc = ssc32.SSC32(porta, taxa, quantidade)
