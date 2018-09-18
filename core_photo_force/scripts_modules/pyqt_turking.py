import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QScrollArea, QVBoxLayout
import PyQt5.Qt
from PyQt5.QtGui import QIcon, QPixmap


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 simple window - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.scrollarea = None
        self.scroll = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.scrollarea = QScrollArea(parent=self)
       
        # Create widget

        #   Scroll Area Properties

        label = QLabel(self.scrollarea)
        # scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        # scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollarea.setWidgetResizable(True)


        #   Scroll Area Layer add
        scroll_layout = QVBoxLayout(self)

        scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(scroll_layout)

        pixmap = QPixmap('/Users/nathanieljones/PycharmProjects/Force-Hackathon-Grain-Size/core_photo_force/data/well_6507_7_4/6507_7_4_24345_24366.jpg')
        label.setPixmap(pixmap)

        #self.resize(pixmap.width(), pixmap.height())



        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())