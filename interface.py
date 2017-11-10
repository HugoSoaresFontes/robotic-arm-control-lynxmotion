from hermes import Braco

from PyQt5.QtCore import QDateTime, Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (QApplication, QDialog, QDial, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget)

v_min, v_med, v_max = 100, 300, 500
servos = ["1", "2", "3", "4", "5"]
braco = Braco(porta='/dev/ttyS0', autocommit=True)

def normalizar(pontos):
    # Amplitudes dos pontos [[x_min, x_max], [y_min, y_max]]
    p_amplitudes = [ [-2.5, 7.5], [20, 30]]

    x_dim = 300
    y_dim = 300
    x_inicio = 50
    y_inicio = 0
    pontos_normalizados = []
    for ponto in pontos:
        x =  round(
                (ponto[0] - p_amplitudes[0][0]) / (p_amplitudes[0][1] - p_amplitudes[0][0]) * x_dim
            ) + x_inicio

        y = round(
                (ponto[1] - p_amplitudes[1][0]) / (p_amplitudes[1][1] - p_amplitudes[1][0]) * y_dim
            ) + y_inicio


        pontos_normalizados += [[x, y]]

    return pontos_normalizados

class QSlider(QSlider):
    def __init__(self, setup, parent=None):
        super(QSlider, self).__init__(parent)
        self.setMinimum(v_min)
        self.setMaximum(v_max)
        self.setValue(v_med)
        self.setTickPosition(QSlider.TicksBelow)
        self.valueChanged.connect(self.changed)
        self._servo = setup

    nome = ''
    def changed(self, valor):
        msg = [self.nome, valor]
        self._servo.velocidade = valor



class QDial(QDial):
    valueSignal = pyqtSignal()

    def __init__(self, setup, parent=None):
        super(QDial, self).__init__(parent)
        self.nome = setup.nome
        self._servo = setup
        self.setValue(0)
        self.setMinimum(setup.angulo_minimo)
        self.setMaximum(setup.angulo_maximo)
        self.setNotchesVisible(True)
        self.valueChanged.connect(self.changed)

    def changed(self, valor):
        # print(self.nome, valor)
        print('Posição: X( ', braco.posicao[1], ')   Y(' ,  braco.posicao[0], ') ',  'Z(' ,  braco.posicao[2], ')'  )
        try:
            self._servo.angulo = valor
        except:
            print('Posicionamento inválido')
        self.valueSignal.emit()


class Servo(object):

    def __init__(self, setup):
        super(Servo, self).__init__()
        self.posicao = [braco.posicao[1], braco.posicao[0], braco.posicao[2]]
        self.group_box = QGroupBox('{0} ({1})'.format(setup.nome, setup.descricao))
        self.group_box.setCheckable(True)
        self.group_box.setChecked(True)

        self.dial = QDial(setup)
        # self.dial.valueSignal.connect(self.nova_posicao)

        self.slider = QSlider(setup, Qt.Horizontal)
        self.slider.nome = '{0} ({1})'.format(setup.nome, setup.descricao)

        layout = QVBoxLayout()
        layout.addWidget(self.dial)
        layout.addWidget(self.slider)
        self.group_box.setLayout(layout)

class WidgetGallery(QDialog):
    # inicio do Densenho do tabuleiro
    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawLines(qp)
        qp.end()

    def drawLines(self, qp):
        #       p0           p1
        #       |            |
        #       |            |
        # p2----p3-----------p4-----p5
        #       |            |
        #       |            |
        # p6----p7-----------p8-----p9
        #       |            |
        #       |            |
        #      p10          p11
        pontos = [ [0.76, 21], [3.66, 20.59],
                   [-2.17, 25.03], [0.86, 24.05], [4.42, 24.87], [7.46, 24.39],
                   [-2.45, 28.33], [1.02, 28.14], [4.03, 28.67], [6.99, 27.95],
                   [1.15, 32], [3.9, 31.69]
                 ]
        pontos = normalizar(pontos)

        pen = QPen(Qt.black, 5, Qt.SolidLine)
        qp.setPen(pen)
        # qp.drawLine args: x_in, y_in, x_fim, y_fim

        # Coluna 1
        qp.drawLine(pontos[0][0], pontos[0][1], pontos[3][0], pontos[3][1])
        qp.drawLine(pontos[3][0], pontos[3][1], pontos[7][0], pontos[7][1])
        qp.drawLine(pontos[7][0], pontos[7][1], pontos[10][0], pontos[10][1])

        #Linha 1
        qp.drawLine(pontos[2][0], pontos[2][1], pontos[3][0], pontos[3][1])
        qp.drawLine(pontos[3][0], pontos[3][1], pontos[4][0], pontos[4][1])
        qp.drawLine(pontos[4][0], pontos[4][1], pontos[5][0], pontos[5][1])

        # Coluna 2
        qp.drawLine(pontos[1][0], pontos[1][1], pontos[4][0], pontos[4][1])
        qp.drawLine(pontos[4][0], pontos[4][1], pontos[8][0], pontos[8][1])
        qp.drawLine(pontos[8][0], pontos[8][1], pontos[11][0], pontos[11][1])


        #  Linha 2
        qp.drawLine(pontos[6][0], pontos[6][1], pontos[7][0], pontos[7][1])
        qp.drawLine(pontos[7][0], pontos[7][1], pontos[8][0], pontos[8][1])
        qp.drawLine(pontos[8][0], pontos[8][1], pontos[9][0], pontos[9][1])

        pen = QPen(Qt.red, 25, Qt.SolidLine)
        qp.setPen(pen)
        ponto = normalizar([ [braco.posicao[1], braco.posicao[0]-0.7 ] ])
        qp.drawLine(ponto[0][0], ponto[0][1], ponto[0][0], ponto[0][1])
        self.update()    # fim do Desenho do tabuleiro

    def marcar_posicao(self):
        self.posicao = [braco.posicao[1], braco.posicao[0], braco.posicao[2]]
        self.paintEvent

    # Controles
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)
        self.posicao = [braco.posicao[1], braco.posicao[0], braco.posicao[2]]

        braco.commit()

        self.servo1 = Servo(braco.servos[0])
        self.servo1.dial.valueSignal.connect(self.marcar_posicao)

        self.servo2 = Servo(braco.servos[1])
        self.servo3 = Servo(braco.servos[2])
        self.servo4 = Servo(braco.servos[3])
        self.servo5 = Servo(braco.servos[4])


        mainLayout = QGridLayout()
        mainLayout.addWidget(self.servo1.group_box, 1, 1)
        mainLayout.addWidget(self.servo2.group_box, 1, 2)
        mainLayout.addWidget(self.servo3.group_box, 2, 0)
        mainLayout.addWidget(self.servo4.group_box, 2, 1)
        mainLayout.addWidget(self.servo5.group_box, 2, 2)

        self.setLayout(mainLayout)
        self.setWindowTitle("Cinematica Direta")
        # self.setGeometry(300, 1000, 280, 270)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.setFixedSize(1250, 750)
    gallery.show()
    sys.exit(app.exec_())
