from PyQt5.QtCore import QDateTime, Qt, QTimer, QObject, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)

a_min, a_med, a_max = -45, 0, 45
v_min, v_med, v_max = 100, 300, 500

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

class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)
        self.width = 1020
        self.height = 780

        self.originalPalette = QApplication.palette()
        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())


        self.criar_sliders()
        self.criar_dials()

        self.servo1()
        self.servo2()
        self.servo3()
        self.servo4()
        self.servo5()

        label = QLabel(self)
        pixmap = QPixmap('representacao.png')
        label.setPixmap(pixmap)

        mainLayout = QGridLayout()
        mainLayout.addWidget(label, 1, 0)
        mainLayout.addWidget(self.servo1, 1, 1)
        mainLayout.addWidget(self.servo2, 1, 2)
        mainLayout.addWidget(self.servo3, 2, 0)
        mainLayout.addWidget(self.servo4, 2, 1)
        mainLayout.addWidget(self.servo5, 2, 2)
        self.setLayout(mainLayout)

        self.setWindowTitle("Cinemática Direta")


    def criar_sliders(self):
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.nome = "1"
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.nome = "2"
        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.nome = "3"
        self.slider4 = QSlider(Qt.Horizontal)
        self.slider4.nome = "4"
        self.slider5 = QSlider(Qt.Horizontal)
        self.slider5.nome = "5"

    def criar_dials(self):
        self.dial1 = QDial()
        self.dial1.nome = "1"
        self.dial2 = QDial()
        self.dial2.nome = "2"
        self.dial3 = QDial()
        self.dial3.nome = "3"
        self.dial4 = QDial()
        self.dial4.nome = "4"
        self.dial5 = QDial()
        self.dial5.nome = "5"

    def servo1(self):
        self.servo1 = QGroupBox("Servo 1")
        self.servo1.setCheckable(True)
        self.servo1.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.dial1)
        layout.addWidget(self.slider1)
        self.servo1.setLayout(layout)

    def servo2(self):
        self.servo2 = QGroupBox("Servo 2")
        self.servo2.setCheckable(True)
        self.servo2.setChecked(True)

        layout = QGridLayout()
        layout.addWidget(self.dial2)
        layout.addWidget(self.slider2)
        self.servo2.setLayout(layout)

    def servo3(self):
        self.servo3 = QGroupBox("Servo 3")
        self.servo3.setCheckable(True)
        self.servo3.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.dial3)
        layout.addWidget(self.slider3)
        self.servo3.setLayout(layout)

    def servo4(self):
        self.servo4 = QGroupBox("Servo 4")
        self.servo4.setCheckable(True)
        self.servo4.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.dial4)
        layout.addWidget(self.slider4)
        self.servo4.setLayout(layout)

    def servo5(self):
        self.servo5 = QGroupBox("Servo 5")
        self.servo5.setCheckable(True)
        self.servo5.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.dial5)
        layout.addWidget(self.slider5)
        self.servo5.setLayout(layout)


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec_())
