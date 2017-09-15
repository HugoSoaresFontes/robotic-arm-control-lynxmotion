from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget)


class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.originalPalette = QApplication.palette()

        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)

        self.useStylePaletteCheckBox = QCheckBox("&Use style's standard palette")
        self.useStylePaletteCheckBox.setChecked(True)

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
        self.slider1.setMinimum(10)
        self.slider1.setMaximum(30)
        self.slider1.setValue(20)
        self.slider1.setTickPosition(QSlider.TicksBelow)
        self.slider1.setTickInterval(5)

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.setMinimum(10)
        self.slider2.setMaximum(30)
        self.slider2.setValue(20)
        self.slider2.setTickPosition(QSlider.TicksBelow)
        self.slider2.setTickInterval(5)

        self.slider3 = QSlider(Qt.Horizontal)
        self.slider3.setMinimum(10)
        self.slider3.setMaximum(30)
        self.slider3.setValue(20)
        self.slider3.setTickPosition(QSlider.TicksBelow)
        self.slider3.setTickInterval(5)

        self.slider4 = QSlider(Qt.Horizontal)
        self.slider4.setMinimum(10)
        self.slider4.setMaximum(30)
        self.slider4.setValue(20)
        self.slider4.setTickPosition(QSlider.TicksBelow)
        self.slider4.setTickInterval(5)

        self.slider5 = QSlider(Qt.Horizontal)
        self.slider5.setMinimum(10)
        self.slider5.setMaximum(30)
        self.slider5.setValue(20)
        self.slider5.setTickPosition(QSlider.TicksBelow)
        self.slider5.setTickInterval(5)

    def criar_dials(self):
        self.dial1 = QDial()
        self.dial1.setValue(30)
        self.dial1.setNotchesVisible(True)

        self.dial2 = QDial()
        self.dial2.setValue(30)
        self.dial2.setNotchesVisible(True)

        self.dial3 = QDial()
        self.dial3.setValue(30)
        self.dial3.setNotchesVisible(True)

        self.dial4 = QDial()
        self.dial4.setValue(30)
        self.dial4.setNotchesVisible(True)

        self.dial5 = QDial()
        self.dial5.setValue(30)
        self.dial5.setNotchesVisible(True)


    def valuechange(self, *args):
        print (self.sl)

    def servo1(self):
        self.servo1 = QGroupBox("Servo 1")
        self.servo1.setCheckable(True)
        self.servo1.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.slider1)
        layout.addWidget(self.dial1)
        self.servo1.setLayout(layout)

    def servo2(self):
        self.servo2 = QGroupBox("Servo 2")
        self.servo2.setCheckable(True)
        self.servo2.setChecked(True)

        layout = QGridLayout()
        layout.addWidget(self.slider2)
        layout.addWidget(self.dial2)
        self.servo2.setLayout(layout)

    def servo3(self):
        self.servo3 = QGroupBox("Servo 3")
        self.servo3.setCheckable(True)
        self.servo3.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.slider3)
        layout.addWidget(self.dial3)
        self.servo3.setLayout(layout)

    def servo4(self):
        self.servo4 = QGroupBox("Servo 4")
        self.servo4.setCheckable(True)
        self.servo4.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.slider4)
        layout.addWidget(self.dial4)
        self.servo4.setLayout(layout)

    def servo5(self):
        self.servo5 = QGroupBox("Servo 5")
        self.servo4.setCheckable(True)
        self.servo4.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.slider5)
        layout.addWidget(self.dial5)
        self.servo5.setLayout(layout)


if __name__ == '__main__':

    import sys

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec_())
