from PyQt5.QtCore import QDateTime, Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (QApplication, QDialog, QDial, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget)

a_min, a_med, a_max = -45, 0, 45
v_min, v_med, v_max = 100, 300, 500
servos = ["1", "2", "3", "4", "5"]
class QSlider(QSlider):
    def __init__(self, parent=None):
        super(QSlider, self).__init__(parent)
        self.setMinimum(v_min)
        self.setMaximum(v_max)
        self.setValue(v_med)
        self.setTickPosition(QSlider.TicksBelow)
        self.valueChanged.connect(self.changed)

    nome = ''
    def changed(self, valor):
        msg = [self.nome, valor]
        print(msg)

class QDial(QDial):
    def __init__(self, parent=None):
        super(QDial, self).__init__(parent)
        self.setValue(a_med)
        self.setMinimum(a_min)
        self.setMaximum(a_max)
        self.setNotchesVisible(True)
        self.valueChanged.connect(self.changed)

    nome = ''
    def changed(self, valor):
        msg = [self.nome, valor]
        print(msg)

class Servo(object):
    def __init__(self, nome):
        super(Servo, self).__init__()
        self.group_box = QGroupBox("Servo " + nome)
        self.group_box.setCheckable(True)
        self.group_box.setChecked(True)

        self.dial = QDial()
        self.dial.nome = nome

        self.slider = QSlider(Qt.Horizontal)
        self.slider.nome = nome

        layout = QVBoxLayout()
        layout.addWidget(self.dial)
        layout.addWidget(self.slider)
        self.group_box.setLayout(layout)

class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.servo1 = Servo("1")
        self.servo2 = Servo("2")
        self.servo3 = Servo("3")
        self.servo4 = Servo("4")
        self.servo5 = Servo("5")

        label = QLabel(self)
        pixmap = QPixmap('representacao.png')
        label.setPixmap(pixmap)

        mainLayout = QGridLayout()
        mainLayout.addWidget(label, 1, 0)
        mainLayout.addWidget(self.servo1.group_box, 1, 1)
        mainLayout.addWidget(self.servo2.group_box, 1, 2)
        mainLayout.addWidget(self.servo3.group_box, 2, 0)
        mainLayout.addWidget(self.servo4.group_box, 2, 1)
        mainLayout.addWidget(self.servo5.group_box, 2, 2)
        self.setLayout(mainLayout)

        self.setWindowTitle("Cinematica Direta")


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec_())
