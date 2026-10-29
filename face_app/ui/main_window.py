"""
CPLC Face Search - Main application window.

Layout
------
  Toolbar bar (logo + title + GPU chip + status dot)
  ┌──────────────────────┬──────────────────────────────────────────┐
  │  LEFT PANEL          │  RIGHT PANEL                             │
  │  ─ Folder config     │  ─ Results table (rank, thumb, file,     │
  │  ─ Drop / select img │    path, score bar)                      │
  │  ─ Search settings   │  ─ Detail card (selected row)            │
  │  ─ [Search] button   │                                          │
  │  ─ Index management  │                                          │
  └──────────────────────┴──────────────────────────────────────────┘
  Status bar
"""
import os
import threading

# Pre-load tensorflow before PyQt5 to prevent Windows DLL initialization collision
try:
    import tensorflow  # noqa: F401
except Exception:
    pass

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QMimeData,
    QPropertyAnimation, QEasingCurve,
)
from PyQt5.QtGui import (
    QPixmap, QIcon, QFont, QColor, QDragEnterEvent, QDropEvent,
    QPainter, QBrush, QPen,
)
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QSlider, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QGroupBox,
    QFileDialog, QMessageBox, QSizePolicy, QAbstractItemView,
    QCheckBox, QStatusBar, QScrollArea, QGridLayout,
    QListWidget, QListWidgetItem, QToolButton, QApplication,
)

# BASE_DIR = face_app/ directory (this file lives in face_app/ui/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = os.path.join(BASE_DIR, "static", "cplc_logo.png")
CONFIG_PATH = os.path.join(BASE_DIR, "search_config.json")

# Ensure face_app/ is on sys.path so vector_db / gpu are importable
import sys as _sys
if BASE_DIR not in _sys.path:
    _sys.path.insert(0, BASE_DIR)

from vector_db import FaceIndex, resolve  # noqa: E402  (needs path above)


# ── Model & Detector descriptions ────────────────────────────────────────────

MODEL_NOTES = {
    "ArcFace": "★ Best Accuracy: Angular margin loss. Top performer for occluded faces (glasses/hats/masks), extreme angles & dim lighting.",
    "Facenet512": "High Precision: 512-D vector space. Robust feature separation, strong balance of speed and matching accuracy.",
    "GhostFaceNet": "Fast & Lightweight: GhostNet backbone optimized for high throughput on standard CPUs or mobile hardware.",
    "Facenet": "Standard Baseline: 128-D embedding (Google). Fast inference, best for clear, well-lit frontal ID photos.",
    "VGG-Face": "Deep Portrait: 2622-D representation. Highly detailed for studio portraits and frontal mugshots (heavier model).",
    "SFace": "Surveillance Optimized: Lightweight cosine sphere model tailored for real-time camera streams.",
}

DETECTOR_NOTES = {
    "retinaface": "★ Best Detection: Pixel-level Deep CNN. Detects tiny, turned, tilted, blurry, occluded faces or faces in low light.",
    "mtcnn": "Landmark Alignment: Multi-task cascaded network with 5-point alignment. Solid for frontal & 45° portraits.",
    "opencv": "Ultra Fast: Haar Cascade CPU detection. Fastest build time, but struggles with turned heads or occlusions.",
    "dlib": "Classic HOG: Fast and reliable on un-occluded, upright frontal faces.",
}


# ── Worker threads ────────────────────────────────────────────────────────────

class BuildWorker(QThread):
    progress = pyqtSignal(int, int, str, bool)   # processed, total, file, ok
    finished = pyqtSignal(object, dict)           # index, stats
    error    = pyqtSignal(str)

    def __init__(self, db_root, model, detector, clusters, workers, force):
        super().__init__()
        self.db_root = db_root
        self.model = model
        self.detector = detector
        self.clusters = clusters
        self.workers = workers
        self.force = force

    def run(self):
        try:
            idx = FaceIndex.build(
                self.db_root,
                model=self.model,
                detector=self.detector,
                num_clusters=self.clusters,
                workers=self.workers,
                force=self.force,
                progress=lambda i, t, f, ok: self.progress.emit(i, t, f, ok),
            )
            self.finished.emit(idx, idx.build_stats)
        except Exception as exc:
            self.error.emit(str(exc))


class SearchWorker(QThread):
    finished = pyqtSignal(list)
    error    = pyqtSignal(str)

    def __init__(self, index, query_path, top_n, mode, search_mode="fused"):
        super().__init__()
        self.index = index
        self.query_path = query_path
        self.top_n = top_n
        self.mode = mode
        self.search_mode = search_mode

    def run(self):
        try:
            if self.mode == "sections":
                matches = self.index.section_matches(self.query_path, top_n=self.top_n,
                                                     search_mode=self.search_mode)
            else:
                matches = self.index.search(self.query_path, top_n=self.top_n,
                                            search_mode=self.search_mode)
            self.finished.emit(matches)
        except Exception as exc:
            self.error.emit(str(exc))



# ── Drop zone ─────────────────────────────────────────────────────────────────

class DropZone(QLabel):
    fileDropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(160)
        self._reset_text()
        self.setStyleSheet(self._style(False))

    def _reset_text(self):
        self.setText("Drop a photo here\nor click to browse")

    def _style(self, hover):
        border_color = "#3b8fe8" if hover else "#1e3a5f"
        bg = "rgba(30, 96, 204, 0.08)" if hover else "#0d1626"
        return f"""
            QLabel {{
                border: 2px dashed {border_color};
                border-radius: 10px;
                background: {bg};
                color: #5a7fa0;
                font-size: 10pt;
            }}
        """

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select query image", "",
                "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
            )
            if path:
                self.fileDropped.emit(path)

    def dragEnterEvent(self, ev: QDragEnterEvent):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()
            self.setStyleSheet(self._style(True))

    def dragLeaveEvent(self, ev):
        self.setStyleSheet(self._style(False))

    def dropEvent(self, ev: QDropEvent):
        self.setStyleSheet(self._style(False))
        urls = ev.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isfile(path):
                self.fileDropped.emit(path)

    def set_image(self, path):
        pix = QPixmap(path)
        if not pix.isNull():
            pix = pix.scaled(
                self.width() - 20, self.height() - 20,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(pix)
        self.setStyleSheet(self._style(False))

    def clear_image(self):
        self.clear()
        self._reset_text()


# ── Folder list widget ────────────────────────────────────────────────────────

class FolderListWidget(QWidget):
    """Displays the list of watched folders + add/remove controls."""
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.lst = QListWidget()
        self.lst.setStyleSheet("""
            QListWidget {
                background: #0d1626;
                border: 1px solid #1e3a5f;
                border-radius: 6px;
                font-size: 9pt;
                color: #dce6f5;
            }
            QListWidget::item {
                padding: 5px 8px;
                border-bottom: 1px solid #1a2d4a;
            }
            QListWidget::item:selected {
                background: #1a3a6e;
            }
        """)
        self.lst.setMaximumHeight(110)
        lay.addWidget(self.lst)

        btn_add = QPushButton("＋  Add folder")
        btn_add.setObjectName("btnGhost")
        btn_add.setFixedHeight(30)
        btn_add.clicked.connect(self._add)
        btn_row.addWidget(btn_add)

        btn_rem = QPushButton("✕  Remove")
        btn_rem.setObjectName("btnDanger")
        btn_rem.setFixedHeight(30)
        btn_rem.clicked.connect(self._remove)
        btn_row.addWidget(btn_rem)

        lay.addLayout(btn_row)

    def _add(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if folder and folder not in self.get_folders():
            self.lst.addItem(folder)
            self.changed.emit()

    def _remove(self):
        row = self.lst.currentRow()
        if row >= 0:
            self.lst.takeItem(row)
            self.changed.emit()

    def get_folders(self):
        return [self.lst.item(i).text() for i in range(self.lst.count())]

    def set_folders(self, folders):
        self.lst.clear()
        for f in folders:
            self.lst.addItem(f)


# ── Score badge ───────────────────────────────────────────────────────────────

def score_color(pct: float) -> str:
    """Return a hex color for a similarity percentage 0-100."""
    if pct >= 70:
        return "#22c55e"
    if pct >= 50:
        return "#3b8fe8"
    if pct >= 35:
        return "#f59e0b"
    return "#ef4444"


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CPLC Face Search")
        self.setMinimumSize(1100, 680)
        self.resize(1280, 820)
        self._index = None
        self._query_path = None
        self._matches = []
        self._build_worker = None
        self._search_worker = None

        self._load_config()
        self._build_ui()
        self._apply_saved_config()

        # Poll GPU on startup
        QTimer.singleShot(500, self._probe_gpu)

    # ─────────────────────────────────────────────────── Config persistence ──

    def _load_config(self):
        import json
        self._config = {
            "folders": [],
            "model": "Facenet",
            "detector": "opencv",
            "clusters": 8,
            "workers": 4,
            "top_n": 15,
            "min_score": 30.0,
            "device": "auto",
            "mode": "all",
            "search_strategy": "Fused (Best)",
        }
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, encoding="utf-8") as fh:
                    self._config.update(json.load(fh))
            except Exception:
                pass

    def _save_config(self):
        import json
        self._config["folders"] = self._folder_list.get_folders()
        self._config["model"] = self._cmb_model.currentText()
        self._config["detector"] = self._cmb_detector.currentText()
        self._config["clusters"] = self._spn_clusters.value()
        self._config["workers"] = self._spn_workers.value()
        self._config["top_n"] = self._sld_topn.value()
        self._config["min_score"] = self._sld_score.value()
        self._config["device"] = self._cmb_device.currentText()
        self._config["mode"] = "all" if self._chk_mode.isChecked() else "sections"
        self._config["search_strategy"] = self._cmb_strategy.currentText()
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
                json.dump(self._config, fh, indent=2)
        except Exception:
            pass

    def _apply_saved_config(self):
        cfg = self._config
        self._folder_list.set_folders(cfg.get("folders", []))
        idx = self._cmb_model.findText(cfg.get("model", "Facenet"))
        if idx >= 0:
            self._cmb_model.setCurrentIndex(idx)
        idx = self._cmb_detector.findText(cfg.get("detector", "opencv"))
        if idx >= 0:
            self._cmb_detector.setCurrentIndex(idx)
        self._spn_clusters.setValue(cfg.get("clusters", 8))
        self._spn_workers.setValue(cfg.get("workers", 4))
        self._sld_topn.setValue(cfg.get("top_n", 15))
        self._sld_score.setValue(int(cfg.get("min_score", 30)))
        idx = self._cmb_device.findText(cfg.get("device", "auto"))
        if idx >= 0:
            self._cmb_device.setCurrentIndex(idx)
        self._chk_mode.setChecked(cfg.get("mode", "all") == "all")
        saved_strategy = cfg.get("search_strategy", "Fused (Best)")
        idx = self._cmb_strategy.findText(saved_strategy)
        if idx >= 0:
            self._cmb_strategy.setCurrentIndex(idx)
        self._sync_model_desc()
        self._sync_detector_desc()
        self._update_topn_label()
        self._update_score_label()
        self._try_load_existing_index()

    def _try_load_existing_index(self):
        folders = self._folder_list.get_folders()
        if not folders:
            return
        db_root_str = "; ".join(folders)
        try:
            idx = FaceIndex.load(
                db_root_str,
                model=self._cmb_model.currentText(),
                detector=self._cmb_detector.currentText(),
                num_clusters=self._spn_clusters.value(),
            )
            if idx is not None:
                self._index = idx
                self._dot.setStyleSheet("color: #22c55e; font-size: 10pt; padding: 0 8px;")
                self._lbl_build_status.setStyleSheet("font-size: 8pt; color: #22c55e;")
                self._lbl_build_status.setText(f"Index loaded · {len(idx.relpaths)} faces")
                self._status_lbl.setText(f"Index ready: {len(idx.relpaths)} images")
        except Exception:
            pass

    # ────────────────────────────────────────────────────────── Build UI ──────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── top accent bar ───────────────────────────────────────────────────
        bar = QFrame()
        bar.setFixedHeight(5)
        bar.setStyleSheet(
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #cc1a1a, stop:0.3 #1e3a8c, stop:1 #3b8fe8);"
        )
        root.addWidget(bar)

        # ── header ───────────────────────────────────────────────────────────
        root.addWidget(self._make_header())

        # ── main content area ────────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background: #1e3a5f; }")
        splitter.addWidget(self._make_left_panel())
        splitter.addWidget(self._make_right_panel())
        splitter.setSizes([420, 860])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(12, 10, 12, 10)
        cl.setSpacing(10)
        cl.addWidget(splitter)
        root.addWidget(content)

        # ── status bar ───────────────────────────────────────────────────────
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._status_lbl = QLabel("Ready")
        self._status_lbl.setStyleSheet("color: #5a7fa0; padding: 0 6px;")
        sb.addWidget(self._status_lbl, 1)

        self._dot = QLabel("●")
        self._dot.setStyleSheet("color: #3d5575; font-size: 10pt; padding: 0 8px;")
        sb.addPermanentWidget(self._dot)

        self._gpu_lbl = QLabel("GPU: probing…")
        self._gpu_lbl.setStyleSheet("color: #3d5575; padding: 0 8px;")
        sb.addPermanentWidget(self._gpu_lbl)

    # ── header ───────────────────────────────────────────────────────────────

    def _make_header(self):
        hdr = QFrame()
        hdr.setFixedHeight(64)
        hdr.setStyleSheet("""
            QFrame {
                background: #050b18;
                border-bottom: 1px solid #1e3a5f;
            }
        """)
        lay = QHBoxLayout(hdr)
        lay.setContentsMargins(16, 0, 16, 0)
        lay.setSpacing(14)

        # Logo
        logo = QLabel()
        if os.path.exists(LOGO_PATH):
            pix = QPixmap(LOGO_PATH).scaled(44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo.setPixmap(pix)
        lay.addWidget(logo)

        # Title block
        title_blk = QVBoxLayout()
        title_blk.setSpacing(1)
        t1 = QLabel("CPLC FACE SEARCH")
        t1.setStyleSheet("font-size: 13pt; font-weight: 900; color: #3b8fe8; letter-spacing: 3px;")
        t2 = QLabel("Citizens - Police Liaison Committee  |  Sindh Governor's Secretariat")
        t2.setStyleSheet("font-size: 8pt; color: #3d5575; letter-spacing: 1px;")
        title_blk.addWidget(t1)
        title_blk.addWidget(t2)
        lay.addLayout(title_blk)

        lay.addStretch()

        # Index status pill
        self._idx_pill = QLabel("No index")
        self._idx_pill.setStyleSheet("""
            background: #0d1626;
            border: 1px solid #1e3a5f;
            border-radius: 12px;
            color: #3d5575;
            padding: 4px 14px;
            font-size: 8pt;
        """)
        lay.addWidget(self._idx_pill)

        # DB indicator
        from db import is_enabled, init_db
        try:
            db_ok = init_db()
        except Exception:
            db_ok = False
        db_pill = QLabel("PostgreSQL ✓" if db_ok else "PostgreSQL offline")
        db_pill.setStyleSheet(f"""
            background: #0d1626;
            border: 1px solid {"#22c55e" if db_ok else "#cc1a1a"};
            border-radius: 12px;
            color: {"#22c55e" if db_ok else "#cc3333"};
            padding: 4px 14px;
            font-size: 8pt;
        """)
        lay.addWidget(db_pill)

        return hdr

    # ── left panel ────────────────────────────────────────────────────────────

    def _make_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(415)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: #07101e;
                width: 8px;
                margin: 0;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #1e3a5f;
                min-height: 24px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3b8fe8;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                background: none;
            }
        """)

        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 8, 8)
        lay.setSpacing(10)

        # ── Folder configuration ─────────────────────────────────────────────
        gb_folders = QGroupBox("Image Folders")
        gb_folders.setStyleSheet("""
            QGroupBox {
                border: 1px solid #1e3a5f;
                border-radius: 8px;
                margin-top: 12px;
                padding: 10px 8px 8px 8px;
            }
            QGroupBox::title {
                color: #3b8fe8;
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                font-weight: bold;
                font-size: 9pt;
            }
        """)
        gfl = QVBoxLayout(gb_folders)
        gfl.setSpacing(6)

        hint = QLabel(
            "Add all folders containing reference images. "
            "Saved between sessions, just add your folders once per deployment."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #5a7fa0; font-size: 8pt;")
        gfl.addWidget(hint)

        self._folder_list = FolderListWidget()
        self._folder_list.changed.connect(self._save_config)
        gfl.addWidget(self._folder_list)

        lay.addWidget(gb_folders)

        # ── Query image ──────────────────────────────────────────────────────
        gb_query = QGroupBox("Query Image")
        gb_query.setStyleSheet(gb_folders.styleSheet())
        gql = QVBoxLayout(gb_query)
        gql.setSpacing(6)

        self._drop = DropZone()
        self._drop.fileDropped.connect(self._on_query_file)
        gql.addWidget(self._drop)

        self._lbl_query_name = QLabel("")
        self._lbl_query_name.setStyleSheet("color: #5a7fa0; font-size: 8pt;")
        self._lbl_query_name.setAlignment(Qt.AlignCenter)
        gql.addWidget(self._lbl_query_name)

        btn_clear = QPushButton("Clear image")
        btn_clear.setObjectName("btnGhost")
        btn_clear.setFixedHeight(28)
        btn_clear.clicked.connect(self._clear_query)
        gql.addWidget(btn_clear)

        lay.addWidget(gb_query)

        # ── Search settings ──────────────────────────────────────────────────
        gb_settings = QGroupBox("Search Settings")
        gb_settings.setStyleSheet(gb_folders.styleSheet())
        gsl = QVBoxLayout(gb_settings)
        gsl.setSpacing(8)

        LBL_W = 80  # fixed label column width

        def _row_lbl(txt):
            l = QLabel(txt)
            l.setFixedWidth(LBL_W)
            l.setStyleSheet("color: #7a9fc2; font-size: 8pt;")
            return l

        # Max results
        row0 = QHBoxLayout(); row0.setSpacing(8)
        row0.addWidget(_row_lbl("Results"))
        self._sld_topn = QSlider(Qt.Horizontal)
        self._sld_topn.setRange(1, 100)
        self._sld_topn.setValue(15)
        self._sld_topn.valueChanged.connect(self._update_topn_label)
        row0.addWidget(self._sld_topn, 1)
        self._lbl_topn = QLabel("15")
        self._lbl_topn.setFixedWidth(32)
        self._lbl_topn.setStyleSheet("color: #3b8fe8; font-weight: bold;")
        row0.addWidget(self._lbl_topn)
        gsl.addLayout(row0)

        # Min score
        row1 = QHBoxLayout(); row1.setSpacing(8)
        row1.addWidget(_row_lbl("Min score %"))
        self._sld_score = QSlider(Qt.Horizontal)
        self._sld_score.setRange(0, 100)
        self._sld_score.setValue(30)
        self._sld_score.valueChanged.connect(self._update_score_label)
        row1.addWidget(self._sld_score, 1)
        self._lbl_score = QLabel("30%")
        self._lbl_score.setFixedWidth(36)
        self._lbl_score.setStyleSheet("color: #3b8fe8; font-weight: bold;")
        row1.addWidget(self._lbl_score)
        gsl.addLayout(row1)

        # Search mode
        row2 = QHBoxLayout(); row2.setSpacing(8)
        row2.addWidget(_row_lbl("Search mode"))
        self._chk_mode = QCheckBox("Global (across all sections)")
        self._chk_mode.setChecked(True)
        self._chk_mode.setStyleSheet("color: #dce6f5; font-size: 9pt;")
        row2.addWidget(self._chk_mode, 1)
        gsl.addLayout(row2)

        # Search strategy (fused / full-face / periocular)
        row3 = QHBoxLayout(); row3.setSpacing(8)
        row3.addWidget(_row_lbl("Strategy"))
        self._cmb_strategy = QComboBox()
        self._cmb_strategy.addItems(["Fused (Best)", "Full Face", "Periocular (Upper)"])
        self._cmb_strategy.setCurrentIndex(0)
        self._cmb_strategy.setStyleSheet(
            "QComboBox { background: #0d1e38; color: #dce6f5; border: 1px solid #1e3a5f;"
            " border-radius: 4px; padding: 2px 6px; font-size: 9pt; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #0d1e38; color: #dce6f5;"
            " selection-background-color: #1e3a5f; }"
        )
        self._cmb_strategy.setToolTip(
            "Fused (Best): max of full-face and upper-face similarity — best for occluded faces\n"
            "Full Face: standard full-face ArcFace embedding\n"
            "Periocular (Upper): eyes/forehead only — ignores hands/masks over mouth"
        )
        self._cmb_strategy.currentIndexChanged.connect(self._save_config)
        row3.addWidget(self._cmb_strategy, 1)
        gsl.addLayout(row3)

        lay.addWidget(gb_settings)

        # ── Search button ────────────────────────────────────────────────────
        self._btn_search = QPushButton("SEARCH")
        self._btn_search.setFixedHeight(48)
        self._btn_search.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #1a3a8c, stop:1 #1e60cc);
                color: #ffffff;
                border: none;
                border-radius: 8px;
                font-size: 12pt;
                font-weight: bold;
                letter-spacing: 2px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #2050b0, stop:1 #3b8fe8);
            }
            QPushButton:pressed { background: #163060; }
            QPushButton:disabled {
                background: #101828;
                color: #3d5575;
            }
        """)
        self._btn_search.clicked.connect(self._on_search)
        lay.addWidget(self._btn_search)

        # Index management
        gb_index = QGroupBox("Index Management")
        gb_index.setStyleSheet(gb_folders.styleSheet())
        gil = QVBoxLayout(gb_index)
        gil.setSpacing(8)

        LBL_W = 80  # fixed label column width

        def _irow(label_text, widget):
            """Helper: fixed-width label + stretching widget in an HBox."""
            r = QHBoxLayout()
            r.setSpacing(8)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(LBL_W)
            lbl.setStyleSheet("color: #7a9fc2; font-size: 8pt;")
            widget.setMinimumHeight(26)
            r.addWidget(lbl)
            r.addWidget(widget, 1)
            return r

        self._cmb_model = QComboBox()
        self._cmb_model.addItems(["ArcFace", "Facenet512", "GhostFaceNet", "Facenet", "VGG-Face", "SFace"])
        for i in range(self._cmb_model.count()):
            m = self._cmb_model.itemText(i)
            if m in MODEL_NOTES:
                self._cmb_model.setItemData(i, MODEL_NOTES[m], Qt.ToolTipRole)
        gil.addLayout(_irow("Model", self._cmb_model))

        self._lbl_model_desc = QLabel()
        self._lbl_model_desc.setWordWrap(True)
        self._lbl_model_desc.setStyleSheet(
            "color: #8cb8e6; font-size: 7.5pt; margin-left: 88px; margin-bottom: 4px;"
        )
        gil.addWidget(self._lbl_model_desc)

        self._cmb_model.currentIndexChanged.connect(self._sync_model_desc)
        self._sync_model_desc()

        self._cmb_detector = QComboBox()
        self._cmb_detector.addItems(["retinaface", "mtcnn", "opencv", "dlib"])
        for i in range(self._cmb_detector.count()):
            d = self._cmb_detector.itemText(i)
            if d in DETECTOR_NOTES:
                self._cmb_detector.setItemData(i, DETECTOR_NOTES[d], Qt.ToolTipRole)
        gil.addLayout(_irow("Detector", self._cmb_detector))

        self._lbl_detector_desc = QLabel()
        self._lbl_detector_desc.setWordWrap(True)
        self._lbl_detector_desc.setStyleSheet(
            "color: #8cb8e6; font-size: 7.5pt; margin-left: 88px; margin-bottom: 4px;"
        )
        gil.addWidget(self._lbl_detector_desc)

        self._cmb_detector.currentIndexChanged.connect(self._sync_detector_desc)
        self._sync_detector_desc()

        self._spn_clusters = QSpinBox()
        self._spn_clusters.setRange(1, 256)
        self._spn_clusters.setValue(16)
        gil.addLayout(_irow("Clusters", self._spn_clusters))

        self._spn_workers = QSpinBox()
        self._spn_workers.setRange(1, 16)
        self._spn_workers.setValue(4)
        gil.addLayout(_irow("Workers", self._spn_workers))

        self._cmb_device = QComboBox()
        self._cmb_device.addItems(["auto", "gpu", "cpu"])
        gil.addLayout(_irow("Device", self._cmb_device))

        self._chk_force = QCheckBox("Force full rebuild")
        self._chk_force.setStyleSheet("color: #7a9fc2; font-size: 8pt;")
        gil.addWidget(self._chk_force)

        self._prog_build = QProgressBar()
        self._prog_build.setRange(0, 100)
        self._prog_build.setValue(0)
        self._prog_build.setVisible(False)
        self._prog_build.setFixedHeight(16)
        self._prog_build.setFormat("%v / %m  (%p%)")
        gil.addWidget(self._prog_build)

        self._lbl_build_status = QLabel("")
        self._lbl_build_status.setWordWrap(True)
        self._lbl_build_status.setStyleSheet("font-size: 8pt; color: #5a7fa0;")
        gil.addWidget(self._lbl_build_status)

        self._btn_build = QPushButton("Build Index")
        self._btn_build.setFixedHeight(34)
        self._btn_build.clicked.connect(self._on_build)
        gil.addWidget(self._btn_build)

        lay.addWidget(gb_index)
        lay.addStretch()

        scroll.setWidget(panel)
        return scroll

    # ── right panel ───────────────────────────────────────────────────────────

    def _make_right_panel(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 0, 0, 0)
        lay.setSpacing(8)

        # Results header
        hdr = QHBoxLayout()
        self._lbl_results_hdr = QLabel("Recognition Results")
        self._lbl_results_hdr.setStyleSheet(
            "font-size: 12pt; font-weight: bold; color: #dce6f5;"
        )
        hdr.addWidget(self._lbl_results_hdr)
        hdr.addStretch()

        btn_export = QPushButton("Export CSV")
        btn_export.setObjectName("btnGhost")
        btn_export.setFixedHeight(30)
        btn_export.clicked.connect(self._export_csv)
        hdr.addWidget(btn_export)

        btn_clear_res = QPushButton("Clear results")
        btn_clear_res.setObjectName("btnGhost")
        btn_clear_res.setFixedHeight(30)
        btn_clear_res.clicked.connect(self._clear_results)
        hdr.addWidget(btn_clear_res)

        lay.addLayout(hdr)

        # ── Results table ─────────────────────────────────────────────────────
        self._tbl = QTableWidget()
        self._tbl.setColumnCount(6)
        self._tbl.setHorizontalHeaderLabels(["", "Rank", "Score", "Filename", "Path", "Section"])
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tbl.setAlternatingRowColors(False)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setShowGrid(False)
        self._tbl.setStyleSheet("""
            QTableWidget {
                background: #0a0f1e;
                border: 1px solid #1e3a5f;
                border-radius: 8px;
                outline: none;
            }
            QTableWidget::item { border-bottom: 1px solid #0f1e35; }
            QTableWidget::item:selected { background: #1a3a6e; color: #fff; }
            QTableWidget::item:alternate { background: #0d1626; }
        """)

        hh = self._tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)   # thumb
        hh.setSectionResizeMode(1, QHeaderView.Fixed)   # rank
        hh.setSectionResizeMode(2, QHeaderView.Fixed)   # score
        hh.setSectionResizeMode(3, QHeaderView.Stretch) # filename
        hh.setSectionResizeMode(4, QHeaderView.Stretch) # path
        hh.setSectionResizeMode(5, QHeaderView.Fixed)   # section

        self._tbl.setColumnWidth(0, 72)   # thumb
        self._tbl.setColumnWidth(1, 52)   # rank
        self._tbl.setColumnWidth(2, 90)   # score
        self._tbl.setColumnWidth(5, 100)  # section

        self._tbl.setRowHeight(0, 68)
        self._tbl.currentItemChanged.connect(self._on_row_selected)
        lay.addWidget(self._tbl, 1)

        # ── Detail card ───────────────────────────────────────────────────────
        self._detail = DetailCard()
        self._detail.setVisible(False)
        lay.addWidget(self._detail)

        return panel

    # ──────────────────────────────────────────────────── Event handlers ──────

    def _sync_model_desc(self):
        if hasattr(self, "_cmb_model") and hasattr(self, "_lbl_model_desc"):
            m = self._cmb_model.currentText()
            self._lbl_model_desc.setText(MODEL_NOTES.get(m, ""))

    def _sync_detector_desc(self):
        if hasattr(self, "_cmb_detector") and hasattr(self, "_lbl_detector_desc"):
            d = self._cmb_detector.currentText()
            self._lbl_detector_desc.setText(DETECTOR_NOTES.get(d, ""))

    def _update_topn_label(self):
        self._lbl_topn.setText(str(self._sld_topn.value()))

    def _update_score_label(self):
        self._lbl_score.setText(f"{self._sld_score.value()}%")

    def _on_query_file(self, path):
        self._query_path = path
        self._drop.set_image(path)
        self._lbl_query_name.setText(os.path.basename(path))

    def _clear_query(self):
        self._query_path = None
        self._drop.clear_image()
        self._lbl_query_name.setText("")

    def _on_row_selected(self, current, previous):
        if current is None:
            self._detail.setVisible(False)
            return
        row = current.row()
        if row < len(self._matches):
            m = self._matches[row]
            self._detail.update_match(m, self._index)
            self._detail.setVisible(True)

    def _clear_results(self):
        self._tbl.setRowCount(0)
        self._matches = []
        self._detail.setVisible(False)
        self._lbl_results_hdr.setText("Recognition Results")

    def _probe_gpu(self):
        try:
            import gpu as _gpu
            info = _gpu.runtime_status()
            if info.get("tensorflow_gpu") and info.get("gpus"):
                names = " + ".join(g["name"] for g in info["gpus"])
                self._gpu_lbl.setText(f"GPU: {names}")
                self._gpu_lbl.setStyleSheet("color: #22c55e; padding: 0 8px;")
            elif info.get("tensorflow_gpu"):
                self._gpu_lbl.setText("GPU: active")
                self._gpu_lbl.setStyleSheet("color: #22c55e; padding: 0 8px;")
            else:
                self._gpu_lbl.setText("GPU: CPU mode")
                self._gpu_lbl.setStyleSheet("color: #f59e0b; padding: 0 8px;")
        except Exception:
            self._gpu_lbl.setText("GPU: unavailable")

    # ── Build ─────────────────────────────────────────────────────────────────

    def _on_build(self):
        folders = self._folder_list.get_folders()
        if not folders:
            self._warn("No folders configured. Add at least one image folder.")
            return

        self._save_config()

        # Apply device before worker starts
        import gpu as _gpu
        _gpu.save_settings({"device": self._cmb_device.currentText()})
        _gpu.apply_device()

        db_root = "; ".join(folders)
        self._btn_build.setEnabled(False)
        self._prog_build.setVisible(True)
        self._prog_build.setValue(0)
        self._lbl_build_status.setText("Starting index build…")
        self._dot.setStyleSheet("color: #f59e0b; font-size: 10pt; padding: 0 8px;")

        self._build_worker = BuildWorker(
            db_root,
            self._cmb_model.currentText(),
            self._cmb_detector.currentText(),
            self._spn_clusters.value(),
            self._spn_workers.value(),
            self._chk_force.isChecked(),
        )
        self._build_worker.progress.connect(self._on_build_progress)
        self._build_worker.finished.connect(self._on_build_done)
        self._build_worker.error.connect(self._on_build_error)
        self._build_worker.start()

    def _on_build_progress(self, processed, total, filename, ok):
        if total > 0:
            pct = int(processed / total * 100)
            self._prog_build.setMaximum(total)
            self._prog_build.setValue(processed)
        fname = os.path.basename(filename)
        self._lbl_build_status.setText(f"Embedding {processed}/{total} · {fname}")
        self._status_lbl.setText(f"Building index: {processed}/{total}")

    def _on_build_done(self, index, stats):
        self._index = index
        self._prog_build.setVisible(False)
        self._btn_build.setEnabled(True)
        self._dot.setStyleSheet("color: #22c55e; font-size: 10pt; padding: 0 8px;")
        count = stats.get("total_files", 0)
        emb = stats.get("embedded", 0)
        kept = stats.get("kept", 0)
        msg = f"Index ready · {count} files scanned · {emb} embedded · {kept} reused"
        self._lbl_build_status.setText(msg)
        self._status_lbl.setText(msg)
        self._idx_pill.setText(f"Index: {len(index.relpaths)} faces")
        self._idx_pill.setStyleSheet("""
            background: #0d1626;
            border: 1px solid #22c55e;
            border-radius: 12px;
            color: #22c55e;
            padding: 4px 14px;
            font-size: 8pt;
        """)
        # Record in PostgreSQL
        try:
            import db as _db
            sec_dict = {
                info["name"]: info["members"]
                for _, info in index.sections.items()
            }
            _db.record_index_build(
                index.db_root, index.model, index.detector,
                index.num_clusters, len(index.relpaths),
                index.relpaths, sec_dict
            )
        except Exception:
            pass

    def _on_build_error(self, msg):
        self._prog_build.setVisible(False)
        self._btn_build.setEnabled(True)
        self._dot.setStyleSheet("color: #ef4444; font-size: 10pt; padding: 0 8px;")
        self._lbl_build_status.setStyleSheet("font-size: 8pt; color: #ef4444;")
        self._lbl_build_status.setText(f"Build error: {msg}")
        self._status_lbl.setText("Build failed.")

    # ── Search ────────────────────────────────────────────────────────────────

    def _on_search(self):
        if self._index is None:
            self._warn("No index built yet. Build the index first.")
            return
        if not self._query_path:
            self._warn("No query image selected. Drop or choose an image.")
            return

        self._btn_search.setEnabled(False)
        self._status_lbl.setText("Searching…")
        self._clear_results()

        mode = "all" if self._chk_mode.isChecked() else "sections"
        top_n = self._sld_topn.value()

        _strategy_map = {
            "Fused (Best)":       "fused",
            "Full Face":          "full",
            "Periocular (Upper)": "upper_periocular",
        }
        search_mode = _strategy_map.get(self._cmb_strategy.currentText(), "fused")

        self._search_worker = SearchWorker(
            self._index, self._query_path, top_n, mode, search_mode=search_mode
        )
        self._search_worker.finished.connect(self._on_search_done)
        self._search_worker.error.connect(self._on_search_error)
        self._search_worker.start()

    def _on_search_done(self, matches):
        self._btn_search.setEnabled(True)
        min_pct = self._sld_score.value()
        # Filter by minimum score
        filtered = [m for m in matches if m["similarity"] * 100 >= min_pct]
        self._matches = filtered

        self._lbl_results_hdr.setText(
            f"Recognition Results  ·  {len(filtered)} match{'es' if len(filtered) != 1 else ''}"
            + (f"  ({len(matches) - len(filtered)} below {min_pct}% threshold)" if len(matches) > len(filtered) else "")
        )
        self._render_results(filtered)
        self._status_lbl.setText(
            f"{len(filtered)} results found (min score: {min_pct}%)"
        )

        # Log to PostgreSQL
        try:
            import db as _db
            fname = os.path.basename(self._query_path)
            _db.record_search(fname, self._sld_topn.value(),
                              self._sld_score.value() / 100.0, matches)
        except Exception:
            pass

    def _on_search_error(self, msg):
        self._btn_search.setEnabled(True)
        self._status_lbl.setText(f"Search error: {msg}")
        QMessageBox.critical(self, "Search Error", msg)

    def _render_results(self, matches):
        self._tbl.setRowCount(0)
        for rank, m in enumerate(matches, 1):
            self._tbl.insertRow(rank - 1)
            self._tbl.setRowHeight(rank - 1, 68)

            # Thumbnail
            thumb_lbl = QLabel()
            thumb_lbl.setAlignment(Qt.AlignCenter)
            thumb_lbl.setFixedSize(66, 66)
            try:
                from vector_db import resolve
                abs_path = resolve(self._index.db_root, m["rel"])
                if abs_path:
                    pix = QPixmap(abs_path).scaled(
                        62, 62, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    thumb_lbl.setPixmap(pix)
            except Exception:
                pass
            self._tbl.setCellWidget(rank - 1, 0, thumb_lbl)

            # Rank
            rank_item = QTableWidgetItem(str(rank))
            rank_item.setTextAlignment(Qt.AlignCenter)
            rank_item.setForeground(QColor("#7a9fc2"))
            self._tbl.setItem(rank - 1, 1, rank_item)

            # Score: coloured badge via widget
            pct = m["similarity"] * 100
            score_wgt = self._make_score_badge(pct)
            self._tbl.setCellWidget(rank - 1, 2, score_wgt)

            # Filename
            rel = m.get("rel", "")
            fname = os.path.basename(rel)
            fn_item = QTableWidgetItem(fname)
            fn_item.setToolTip(rel)
            self._tbl.setItem(rank - 1, 3, fn_item)

            # Path (directory)
            path_item = QTableWidgetItem(os.path.dirname(rel))
            path_item.setForeground(QColor("#5a7fa0"))
            path_item.setToolTip(rel)
            self._tbl.setItem(rank - 1, 4, path_item)

            # Section
            sec_item = QTableWidgetItem(m.get("section", ""))
            sec_item.setTextAlignment(Qt.AlignCenter)
            sec_item.setForeground(QColor("#5a7fa0"))
            self._tbl.setItem(rank - 1, 5, sec_item)

    def _make_score_badge(self, pct: float) -> QWidget:
        """A mini widget that shows a coloured score bar + percentage text."""
        wgt = QWidget()
        wgt.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(wgt)
        lay.setContentsMargins(6, 8, 6, 8)
        lay.setSpacing(3)

        color = score_color(pct)
        lbl = QLabel(f"{pct:.1f}%")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f"color: {color}; font-size: 11pt; font-weight: bold;")
        lay.addWidget(lbl)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(pct))
        bar.setTextVisible(False)
        bar.setFixedHeight(5)
        bar.setStyleSheet(f"""
            QProgressBar {{
                background: #0d1626;
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 2px;
            }}
        """)
        lay.addWidget(bar)
        return wgt

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_csv(self):
        if not self._matches:
            self._warn("No results to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save results as CSV", "cplc_results.csv",
            "CSV files (*.csv)"
        )
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Rank", "Score%", "Filename", "RelPath", "Section"])
            for i, m in enumerate(self._matches, 1):
                writer.writerow([
                    i,
                    f"{m['similarity'] * 100:.2f}",
                    os.path.basename(m["rel"]),
                    m["rel"],
                    m.get("section", ""),
                ])
        self._status_lbl.setText(f"Exported {len(self._matches)} rows → {path}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _warn(self, msg):
        QMessageBox.warning(self, "CPLC Face Search", msg)

    def closeEvent(self, ev):
        self._save_config()
        ev.accept()


# ── Detail card ───────────────────────────────────────────────────────────────

class DetailCard(QFrame):
    """Shows full-size thumb + details of the selected result row."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(160)
        self.setStyleSheet("""
            QFrame {
                background: #0d1626;
                border: 1px solid #1e3a5f;
                border-radius: 8px;
            }
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(16)

        self._img = QLabel()
        self._img.setFixedSize(120, 130)
        self._img.setAlignment(Qt.AlignCenter)
        self._img.setStyleSheet("border: 1px solid #1e3a5f; border-radius: 6px; background: #0a0f1e;")
        lay.addWidget(self._img)

        info = QVBoxLayout()
        info.setSpacing(4)
        self._lbl_fname = QLabel()
        self._lbl_fname.setStyleSheet("font-size: 11pt; font-weight: bold; color: #dce6f5;")
        self._lbl_fname.setWordWrap(True)
        info.addWidget(self._lbl_fname)

        self._lbl_path = QLabel()
        self._lbl_path.setStyleSheet("font-size: 8pt; color: #5a7fa0;")
        self._lbl_path.setWordWrap(True)
        info.addWidget(self._lbl_path)

        self._lbl_score = QLabel()
        self._lbl_score.setStyleSheet("font-size: 13pt; font-weight: bold;")
        info.addWidget(self._lbl_score)

        self._lbl_section = QLabel()
        self._lbl_section.setStyleSheet("""
            background: rgba(59, 143, 232, 0.15);
            border: 1px solid rgba(59, 143, 232, 0.4);
            border-radius: 10px;
            color: #3b8fe8;
            padding: 2px 10px;
            font-size: 8pt;
        """)
        self._lbl_section.setMaximumWidth(140)
        info.addWidget(self._lbl_section)
        info.addStretch()

        lay.addLayout(info, 1)

    def update_match(self, match: dict, index):
        from vector_db import resolve
        rel = match.get("rel", "")
        pct = match["similarity"] * 100
        color = score_color(pct)

        self._lbl_fname.setText(os.path.basename(rel))
        self._lbl_path.setText(rel)
        self._lbl_score.setText(f"{pct:.1f}% match")
        self._lbl_score.setStyleSheet(f"font-size: 13pt; font-weight: bold; color: {color};")
        self._lbl_section.setText(match.get("section", ""))

        self._img.clear()
        if index:
            try:
                abs_path = resolve(index.db_root, rel)
                if abs_path and os.path.isfile(abs_path):
                    pix = QPixmap(abs_path).scaled(
                        116, 126, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self._img.setPixmap(pix)
            except Exception:
                pass
