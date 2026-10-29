"""
CPLC Face Search - PyQt5 desktop application entry point.
"""
import sys
import os

# Ensure face_app/ is on sys.path regardless of launch directory
_FACE_APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = _FACE_APP_DIR
if _FACE_APP_DIR not in sys.path:
    sys.path.insert(0, _FACE_APP_DIR)

# Must apply GPU settings before deepface / tensorflow are imported
import gpu
gpu.apply_device()

# Pre-load tensorflow before PyQt5 to prevent Windows DLL initialization collision
try:
    import tensorflow  # noqa: F401
except Exception:
    pass

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon, QFont
from ui.login_window import LoginWindow



def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CPLC Face Search")
    app.setOrganizationName("CPLC")

    # App-wide font
    app.setFont(QFont("Segoe UI", 10))

    # Window icon
    icon_path = os.path.join(BASE_DIR, "static", "cplc_logo.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # Global dark stylesheet
    app.setStyleSheet(GLOBAL_STYLE)

    win = LoginWindow()
    win.show()

    sys.exit(app.exec_())


GLOBAL_STYLE = """
QWidget {
    background-color: #0a0f1e;
    color: #dce6f5;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 10pt;
}
QMainWindow, QDialog {
    background-color: #0a0f1e;
}
QFrame#panel {
    background-color: #101828;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
}
QLabel#heading {
    font-size: 16pt;
    font-weight: bold;
    color: #3b8fe8;
}
QLabel#subheading {
    font-size: 9pt;
    color: #7a9fc2;
}
QLabel#sectionTitle {
    font-size: 9pt;
    font-weight: bold;
    color: #7a9fc2;
    text-transform: uppercase;
    letter-spacing: 1px;
}
QPushButton {
    background-color: #1a3a6e;
    color: #dce6f5;
    border: 1px solid #2a5faa;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #2255a0;
    border-color: #3b8fe8;
}
QPushButton:pressed {
    background-color: #163060;
}
QPushButton:disabled {
    background-color: #101828;
    color: #3d5575;
    border-color: #1e3a5f;
}
QPushButton#btnPrimary {
    background-color: #1e60cc;
    color: #ffffff;
    border: none;
    font-size: 10pt;
    padding: 10px 20px;
}
QPushButton#btnPrimary:hover {
    background-color: #2677e8;
}
QPushButton#btnDanger {
    background-color: #8b1a1a;
    color: #ffffff;
    border: 1px solid #cc2222;
}
QPushButton#btnDanger:hover {
    background-color: #b02020;
}
QPushButton#btnGhost {
    background-color: transparent;
    border: 1px solid #1e3a5f;
    color: #7a9fc2;
}
QPushButton#btnGhost:hover {
    background-color: #101828;
    color: #dce6f5;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #0d1626;
    color: #dce6f5;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 7px 10px;
    selection-background-color: #1e60cc;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #3b8fe8;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background-color: #0d1626;
    border: 1px solid #1e3a5f;
    selection-background-color: #1e3a5f;
    color: #dce6f5;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1e3a5f;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #3b8fe8;
    border: 2px solid #1e60cc;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #1e60cc;
    border-radius: 3px;
}
QTableWidget {
    background-color: #0d1626;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    gridline-color: #1a2d4a;
    color: #dce6f5;
    selection-background-color: #1a3a6e;
    outline: none;
}
QTableWidget::item {
    padding: 6px 10px;
    border-bottom: 1px solid #1a2d4a;
}
QTableWidget::item:selected {
    background-color: #1a3a6e;
    color: #ffffff;
}
QHeaderView::section {
    background-color: #0a1428;
    color: #7a9fc2;
    border: none;
    border-bottom: 2px solid #1e3a5f;
    border-right: 1px solid #1a2d4a;
    padding: 8px 10px;
    font-weight: bold;
    font-size: 9pt;
    text-transform: uppercase;
}
QScrollBar:vertical {
    background: #0a0f1e;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #1e3a5f;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #2a5faa;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal {
    background: #0a0f1e;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #1e3a5f;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover { background: #2a5faa; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
QProgressBar {
    background-color: #0d1626;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    text-align: center;
    color: #dce6f5;
    font-size: 9pt;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1e60cc, stop:1 #3b8fe8);
    border-radius: 5px;
}
QCheckBox {
    color: #7a9fc2;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #2a5faa;
    border-radius: 4px;
    background: #0d1626;
}
QCheckBox::indicator:checked {
    background: #1e60cc;
    border-color: #3b8fe8;
}
QStatusBar {
    background-color: #050b18;
    color: #7a9fc2;
    border-top: 1px solid #1e3a5f;
    font-size: 9pt;
}
QGroupBox {
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 10px;
}
QGroupBox::title {
    color: #3b8fe8;
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    font-weight: bold;
    font-size: 9pt;
}
QTabWidget::pane {
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    background: #101828;
}
QTabBar::tab {
    background: #0a1428;
    color: #7a9fc2;
    border: 1px solid #1e3a5f;
    border-bottom: none;
    padding: 8px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #101828;
    color: #dce6f5;
    border-top: 2px solid #3b8fe8;
}
QTabBar::tab:hover:!selected {
    background: #0d1a30;
    color: #dce6f5;
}
QSplitter::handle {
    background: #1e3a5f;
}
QToolTip {
    background-color: #0d1626;
    color: #dce6f5;
    border: 1px solid #2a5faa;
    border-radius: 4px;
    padding: 4px 8px;
}
"""


if __name__ == "__main__":
    main()
