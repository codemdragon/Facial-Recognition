"""
Login window - placeholder UI.
Clicking "Login" just opens the main window.
No authentication logic yet.
"""
import os

from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFrame, QGraphicsDropShadowEffect, QSizePolicy,
    QSpacerItem,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = os.path.join(BASE_DIR, "static", "cplc_logo.png")


class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CPLC Face Search | Login")
        self.setFixedSize(480, 620)
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.MSWindowsFixedSizeDialogHint)
        self._main_win = None
        self._build_ui()

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # top accent bar
        bar = QFrame()
        bar.setFixedHeight(6)
        bar.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #cc1a1a, stop:0.45 #1e60cc, stop:1 #3b8fe8);"
        )
        root.addWidget(bar)

        # main card
        card = QFrame()
        card.setObjectName("loginCard")
        card.setStyleSheet("""
            QFrame#loginCard {
                background: #0d1626;
                border: 1px solid #1e3a5f;
                border-top: none;
                border-radius: 0px;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 8)
        card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(50, 40, 50, 40)
        card_layout.setSpacing(0)

        # Logo
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        if os.path.exists(LOGO_PATH):
            pix = QPixmap(LOGO_PATH).scaled(
                110, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            logo_label.setPixmap(pix)
        card_layout.addWidget(logo_label)
        card_layout.addSpacing(18)

        # App name
        title = QLabel("CPLC FACE SEARCH")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 20pt;
            font-weight: 900;
            letter-spacing: 4px;
            color: #3b8fe8;
        """)
        card_layout.addWidget(title)

        subtitle = QLabel("Citizens - Police Liaison Committee")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 9pt; color: #7a9fc2; letter-spacing: 1px;")
        card_layout.addWidget(subtitle)
        card_layout.addSpacing(36)

        # fields
        def field_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size: 8pt; color: #5a7fa0; font-weight: bold; margin-bottom: 3px;")
            return lbl

        # Server
        card_layout.addWidget(field_label("SERVER ADDRESS"))
        self.edt_server = QLineEdit("http://localhost:5000")
        self.edt_server.setPlaceholderText("http://server:5000")
        card_layout.addWidget(self.edt_server)
        card_layout.addSpacing(14)

        # Username
        card_layout.addWidget(field_label("USERNAME"))
        self.edt_user = QLineEdit()
        self.edt_user.setPlaceholderText("Enter username")
        card_layout.addWidget(self.edt_user)
        card_layout.addSpacing(14)

        # Password
        card_layout.addWidget(field_label("PASSWORD"))
        self.edt_pass = QLineEdit()
        self.edt_pass.setPlaceholderText("Enter password")
        self.edt_pass.setEchoMode(QLineEdit.Password)
        card_layout.addWidget(self.edt_pass)
        card_layout.addSpacing(28)

        # buttons
        btn_login = QPushButton("LOGIN")
        btn_login.setObjectName("btnPrimary")
        btn_login.setFixedHeight(46)
        btn_login.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1a3a8c, stop:1 #1e60cc);
                color: #ffffff;
                border: none;
                border-radius: 8px;
                font-size: 12pt;
                font-weight: bold;
                letter-spacing: 3px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #2050b0, stop:1 #3b8fe8);
            }
            QPushButton:pressed {
                background: #163060;
            }
        """)
        btn_login.clicked.connect(self._on_login)
        btn_login.setDefault(True)
        card_layout.addWidget(btn_login)
        card_layout.addSpacing(10)

        btn_skip = QPushButton("Continue without login")
        btn_skip.setObjectName("btnGhost")
        btn_skip.setFixedHeight(36)
        btn_skip.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #5a7fa0;
                border: 1px solid #1e3a5f;
                border-radius: 6px;
                font-size: 9pt;
            }
            QPushButton:hover {
                color: #dce6f5;
                border-color: #3b8fe8;
            }
        """)
        btn_skip.clicked.connect(self._on_login)
        card_layout.addWidget(btn_skip)

        # hint
        card_layout.addSpacing(20)
        hint = QLabel("Login is not yet active: click either button to continue.")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 8pt; color: #3d5575;")
        card_layout.addWidget(hint)

        card_layout.addStretch()

        root.addWidget(card)

        # ── bottom strip ────────────────────────────────────────────────────
        bottom = QFrame()
        bottom.setFixedHeight(32)
        bottom.setStyleSheet("background: #050b18; border-top: 1px solid #0d1e38;")
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(16, 0, 16, 0)
        ver = QLabel("v1.0.0  ·  Sindh Governor's Secretariat")
        ver.setStyleSheet("font-size: 8pt; color: #2a4060;")
        bl.addWidget(ver)
        bl.addStretch()
        root.addWidget(bottom)

    # ---------------------------------------------------------------- slots --
    def _on_login(self):
        from ui.main_window import MainWindow
        self._main_win = MainWindow()
        self._main_win.show()
        self.hide()
