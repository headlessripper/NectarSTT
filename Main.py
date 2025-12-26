# Third-party dependencies
# PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton,
    QSpacerItem, QSizePolicy, QLineEdit, QGridLayout, QMessageBox,
    QTextEdit, QCheckBox, QHBoxLayout, QFrame, QFileDialog, QInputDialog,
    QPlainTextEdit, QSplitter, QStackedWidget, QStackedLayout, QFormLayout,
    QSlider, QScrollArea, QGraphicsOpacityEffect, QListWidgetItem, QComboBox, QGroupBox,
    QTabWidget, QCompleter, QMenu, QToolTip, QToolButton, QGraphicsDropShadowEffect,
    QDialogButtonBox, QDialog, QSpinBox, QDoubleSpinBox
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QElapsedTimer,
    QObject, QSize, QPropertyAnimation, QEvent,
    QFileSystemWatcher, QStringListModel, QPoint, QUrl, QEasingCurve, QVariantAnimation, QRect,
    pyqtProperty, pyqtSlot, QRunnable, QThreadPool # Added pyqtSlot for completeness although not explicitly imported
)
from PyQt6.QtGui import (
    QIcon, QFont, QPixmap, QPainter, QPainterPath, QDesktopServices, QShortcut,
    QColor, QSyntaxHighlighter, QAction,
    QPen, QKeySequence, QGuiApplication, QCursor, QBrush, 
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput, QSoundEffect

from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt6.QtCore import QThread, pyqtSignal, QEventLoop, QObject
import sys
from click import style
import sounddevice as sd
from scipy.io.wavfile import write
import subprocess
import tempfile
import os
import re
import numpy as np
import threading
import json

import os

def find_MainFolder(relative_path):
    """
    Finds a file relative to the application's folder (outside PyInstaller _MEIPASS),
    works in both script and built exe.
    Returns absolute path to the file.
    """
    # Base folder: folder where the exe/script is located
    if getattr(sys, "frozen", False):
        # PyInstaller exe
        base_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        base_dir = os.path.dirname(os.path.realpath(__file__))

    # Possible paths
    abs_path = os.path.join(base_dir, relative_path)
    if os.path.exists(abs_path):
        return abs_path

    # Fallback: check parent directory
    abs_path_parent = os.path.abspath(os.path.join(base_dir, os.pardir, relative_path))
    if os.path.exists(abs_path_parent):
        return abs_path_parent

    # If still not found, just return intended path (so app can create it if needed)
    return abs_path


#--------------------------------------------------------------------------------
# Send Function
#--------------------------------------------------------------------------------

def send_to_AlphaLLM(question):
    import socket
    
    SERVER_ADDRESS = ('127.0.0.1', 5005)
    AUTH_KEY = ("NEC-892657") # 🔒 Security Key  # must match the server’s key

    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #client.settimeout(10)  # optional safety timeout
        client.connect(SERVER_ADDRESS)

        # Prepend the key before the question
        secure_message = f"{AUTH_KEY} {question}"
        client.sendall(secure_message.encode('utf-8'))

        response = client.recv(8192).decode('utf-8').strip()
        client.close()

        return response

    except (socket.error, socket.timeout) as e:
        return f"[Error] Could not connect to NectarLLM: {e}"    

# -------------------- Configuration --------------------
CONFIG_FILE = "config.json"
menu_btn_icon = find_MainFolder("Main-Engine/Images/btn-icon/menu-black.png")
menu_hover_btn_icon = find_MainFolder("Main-Engine/Images/btn-icon/menu-white.png")
PIPER_EXE_PATH = find_MainFolder("Main-Engine/TTS-Engine/piper-cpu/piper.exe")
SOUND_EFFECTS_PATH = find_MainFolder("Main-Engine/Sound/Effects/Drops/droplet.mp3")
EXE_PATH = find_MainFolder("Main-Engine/STT-Engine/whisper-cli.exe")
MODEL_PATH = find_MainFolder("Main-Engine/Model/whisper_Model/ggml-base.bin")

class Config:
    ONNEX_MODEL_PATH = find_MainFolder("Main-Engine/Model/en_onnx-ryan-high_Model/en_US-ryan-high.onnx")
    VOICE_SPEED = 1.3
    FS = 16000
    SILENCE_THRESHOLD = 500
    CHUNK_SLEEP_MS = 50
    MIN_SPEECH_DURATION = 0.2

    @classmethod
    def load(cls):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                for key, value in data.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)

    @classmethod
    def save(cls):
        data = {}
        for key, value in cls.__dict__.items():
            if key.startswith("__"):
                continue
            if callable(value):     # <-- FIX HERE
                continue
            if isinstance(value, (int, float, str)):
                data[key] = value

        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)

# -------------------------------------------------------

ONNX_MODELS = {
        "Ryan-High": "Main-Engine/Model/en_onnx-ryan-high_Model/en_US-ryan-high.onnx",
        "Lessac-Medium": "Main-Engine/Model/en_onnx-lessac-medium_Model/en_US-lessac-medium.onnx",
        "Lessac-High": "Main-Engine/Model/en_onnx-lessac-high_Model/en_US-lessac-high.onnx",
        "Amy-Medium": "Main-Engine/Model/en_onnx-amy-medium_Model/en_US-amy-medium.onnx"
    
    }

class Notify(QWidget):
    def __init__(self, message, duration=3000, parent=None):
        super().__init__(parent, flags=Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        self.label = QLabel(message, self)
        self.label.setStyleSheet("""
            QLabel {
                background-color: #333333;
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
                padding: 10px 20px;
                border-radius: 8px;
            }
        """)
        self.label.adjustSize()
        self.resize(self.label.size())

        # Center on parent if given, else center on screen
        if parent:
            parent_rect = parent.geometry()
            self.move(
                parent_rect.center().x() - self.width() // 2,
                parent_rect.top() + 50
            )
        else:
            screen = self.screen().geometry()
            self.move(
                screen.center().x() - self.width() // 2,
                50
            )

        # Auto-close timer
        QTimer.singleShot(duration, self.close)
        self.show()

class SpeechWorker(QObject):
    finished = pyqtSignal(str)  # emit temp file path after synthesis
    error = pyqtSignal(str)

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        try:
            # Create a temporary file for the TTS output
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
                output_path = tmpfile.name

            # Build the piper command
            cmd = [
                PIPER_EXE_PATH,
                "--model", Config.ONNEX_MODEL_PATH,
                "--output_file", output_path,
                "--use-cuda", "1",
                "--length-scale", str(Config.VOICE_SPEED),
            ]

            # Prevent terminal popup on Windows
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW

            # Run Piper SILENTLY
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,     # <-- hides terminal window
            )

            stdout, stderr = process.communicate(input=self.text.encode("utf-8"))

            if process.returncode != 0:
                raise RuntimeError(f"Piper failed: {stderr.decode()}")

            # Emit the temp file path when done
            self.finished.emit(output_path)

        except Exception as e:
            self.error.emit(str(e))
            
class AnimatedDot(QLabel):
    dotClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # sensible default (small dot when inactive)
        self._radius = 60
        self._color = QColor('#000000')  # gray by default
        self._opacity = 1.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # radius animation (pulsing)
        self._radius_anim = QPropertyAnimation(self, b'radius')
        self._radius_anim.setStartValue(12)
        self._radius_anim.setEndValue(22)
        self._radius_anim.setDuration(900)
        self._radius_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        # make it loop by reversing direction on finished
        self._radius_anim.finished.connect(self._reverse_radius_animation)

        # opacity animation (subtle)
        self._opacity_anim = QPropertyAnimation(self, b'opacity')
        self._opacity_anim.setStartValue(0.7)
        self._opacity_anim.setEndValue(1.0)
        self._opacity_anim.setDuration(900)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._opacity_anim.finished.connect(self._reverse_opacity_animation)

        # color animation (optional color shift)
        self._color_anim = QVariantAnimation()
        self._color_anim.setStartValue(QColor('#000000'))  # gray -> lighter gray
        self._color_anim.setEndValue(QColor('#faf7f7'))
        self._color_anim.setDuration(600)
        self._color_anim.valueChanged.connect(self.set_color)

        # keep a small preferred size
        self.setFixedSize(48, 48)

    # emit signal on click
    def mousePressEvent(self, event):
        self.dotClicked.emit()
        super().mousePressEvent(event)

    def _reverse_radius_animation(self):
        # toggle direction to create a ping-pong loop
        if self._radius_anim.direction() == QPropertyAnimation.Direction.Forward:
            self._radius_anim.setDirection(QPropertyAnimation.Direction.Backward)
        else:
            self._radius_anim.setDirection(QPropertyAnimation.Direction.Forward)
        self._radius_anim.start()

    def _reverse_opacity_animation(self):
        if self._opacity_anim.direction() == QPropertyAnimation.Direction.Forward:
            self._opacity_anim.setDirection(QPropertyAnimation.Direction.Backward)
        else:
            self._opacity_anim.setDirection(QPropertyAnimation.Direction.Forward)
        self._opacity_anim.start()

    def set_active(self, is_active: bool):
        """
        Turn the pulsing animations on/off.
        When deactivating we STOP animations and reset to sensible visuals.
        """
        if is_active:
            self.setToolTip('Listening')
            # ensure color animation uses gray -> light
            self._color_anim.setStartValue(QColor('#ffffff'))
            self._color_anim.setEndValue(QColor('#000000'))
            self._color_anim.setDirection(QVariantAnimation.Direction.Forward)
            self._color_anim.start()

            self._radius_anim.setDirection(QPropertyAnimation.Direction.Forward)
            self._radius_anim.start()

            self._opacity_anim.setDirection(QPropertyAnimation.Direction.Forward)
            self._opacity_anim.start()
        else:
            self.setToolTip('Click To And Start Speaking')
            # stop animations cleanly (do NOT disconnect signal handlers)
            try:
                self._radius_anim.stop()
            except Exception:
                pass
            try:
                self._opacity_anim.stop()
            except Exception:
                pass
            try:
                self._color_anim.stop()
            except Exception:
                pass

            # reset to compact inactive appearance
            self._radius = 60
            self._opacity = 1.0
            self._color = QColor('#ffffff')  # gray
            self.update()

    # color setter used by QVariantAnimation
    def set_color(self, color: QColor):
        # QVariantAnimation may pass QColor or str; handle both
        if isinstance(color, QColor):
            self._color = color
        else:
            self._color = QColor(color)
        self.update()

    def get_radius(self):
        return self._radius

    def set_radius(self, value):
        self._radius = int(value)
        self.update()

    def get_opacity(self):
        return self._opacity

    def set_opacity(self, value):
        self._opacity = float(value)
        self.update()

    radius = pyqtProperty(int, fget=get_radius, fset=set_radius)
    opacity = pyqtProperty(float, fget=get_opacity, fset=set_opacity)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # draw centered glow and the main circle
        center = self.rect().center()
        glow_color = QColor(self._color)
        glow_color.setAlphaF(0.15 * self._opacity)
        painter.setPen(Qt.PenStyle.NoPen)

        # outer glow (slightly larger than radius)
        painter.setBrush(QBrush(glow_color))
        painter.drawEllipse(center, int(self._radius + 60), int(self._radius + 60))

        # inner circle
        painter.setOpacity(self._opacity)
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(center, self._radius, self._radius)

    def closeEvent(self, event):
        # safe stop of animations
        try:
            self._radius_anim.stop()
            self._opacity_anim.stop()
            self._color_anim.stop()
        except Exception:
            pass
        event.accept()

class HoverButton(QPushButton):
    def __init__(self, normal_icon, hover_icon, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normal_icon = QIcon(normal_icon)
        self.hover_icon = QIcon(hover_icon)
        self.setIcon(self.normal_icon)
        self.setFixedSize(32, 32)
        self.setStyleSheet("""
            QPushButton {
                border-radius: 12px;
                background-color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #0d0d0d;
            }
        """)

    def enterEvent(self, event):
        self.setIcon(self.hover_icon)  # change icon on hover
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setIcon(self.normal_icon)  # revert icon when not hovering
        super().leaveEvent(event)        

class Main(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NectarSTT")
        self.setWindowIcon(QIcon(find_MainFolder(r"Main-Engine/Images/Icon/icon.png")))
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint   # <-- add this
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        Config.load()

        # Set up the sound
        # Set up audio player
        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)
        self.player.setSource(QUrl.fromLocalFile(SOUND_EFFECTS_PATH))  # your MP3 file
        self.audio_output.setVolume(1.0)  # 0.0 to 1.0

        # Connect to mediaStatusChanged to cleanup after finished
        self.player.mediaStatusChanged.connect(self.cleanup_player)

        # -------------------
        # MAIN WINDOW STYLES
        # -------------------
        self.setStyleSheet("""
            QWidget {
                background-color: #0e0e0e;
                border-radius: 12px;
                color: white;
                font-family: 'Segoe UI', Arial;
            }

            QLabel {
                background-color: Transparent;
                font-size: 15px;
                color: #e6e6e6;
            }

            QPushButton {
                background-color: #1b1b1b;
                border: 1px solid #333;
                border-radius: 8px;
                padding: 6px 12px;
                color: white;
            }

            QPushButton:hover {
                background-color: #2a2a2a;
            }

        """)

        self.resize(520, 360)


        # -------------------------------------
        # INITIAL STATE (dot starts inactive)
        # -------------------------------------
        self.listening = False
        self.stop_recording = False

        self.menu_btn = HoverButton(menu_btn_icon, menu_hover_btn_icon)
        self.menu_btn.setFixedSize(32, 32)
        self.menu_btn.clicked.connect(self.show_menu)
        self. menu_btn.setStyleSheet("""
            QPushButton {
                border-radius: 12px;
                background-color: #ffffff;
                color: #fff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #010101;
            }
            QPushButton:pressed {
                background-color: #0d0d0d;
            }
        """)
        
        self.dot = AnimatedDot()
        self.dot.set_active(False)     # <-- start inactive
        self.dot.setFixedSize(260, 260)
        self.dot.setStyleSheet("""
           QToolTip {
                background-color: Transparent;
                color: #ffffff;
                border: Transparent;
                padding: 6px;
            }
            """)

        self.dot.dotClicked.connect(self.on_dot_clicked)

        # === Scrollable text output ===
        self.text_scroll = QScrollArea()
        self.text_scroll.setStyleSheet("""
            /* Vertical Scrollbar */
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 0px;
                border: none;
            }

            QScrollBar::handle:vertical {
                background: #ffffff;
                min-height: 20px;
                border-radius: 4px;
            }

            QScrollBar::handle:vertical:hover {
                background: #ffffff;
                border-radius: 4px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: Transparent;
                height: 0px;
                border: none;
                border-radius: 4px;
            }

            /* Horizontal Scrollbar */
            QScrollBar:horizontal {
                background: transparent;
                height: 8px;
                margin: 0px;
                border: none;
            }

            QScrollBar::handle:horizontal {
                background: rgba(0, 0, 0, 0.2);
                min-width: 20px;
                border-radius: 4px;
            }

            QScrollBar::handle:horizontal:hover {
                background: rgba(0, 0, 0, 0.4);
            }

            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {
                background: none;
                width: 0px;
                border: none;
            }
            """)
        self.text_scroll.setWidgetResizable(True)
        self.text_scroll.setFixedWidth(400)       # same sizes you already use
        self.text_scroll.setFixedHeight(80)       # controls visible area

        self.textout = QLabel("Click here")
        self.setStyleSheet("background-color: transparent; border: none;")
        self.textout.setWordWrap(True)
        self.textout.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        # Let label dynamically grow inside the scroll area
        self.textout.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # Put label inside the scroll area
        self.text_scroll.setWidget(self.textout)

        layout = QVBoxLayout(self)
        layout.addWidget(self.menu_btn, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        layout.addStretch()
        layout.addWidget(self.dot, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        layout.addWidget(self.text_scroll , alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        layout.setContentsMargins(20, 20, 20, 20)

        # optional redraw timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.dot.update)
        self._timer.start(16)

    def show_menu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #000000;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 5px;
                font: 12pt "Segoe UI", "Arial";
            }

            QMenu::item {
                padding: 5px 25px 5px 20px;
                border-radius: 5px;
            }

            QMenu::item:selected {
                background-color: #ffffff; /* blue hover */
                color: #000000;
            }

            QMenu::separator {
                height: 1px;
                background: #555;
                margin: 5px 10px 5px 10px;
            }
        """)

        setting_action = QAction("Settings", self)
        setting_action.triggered.connect(self.open_settings)
        menu.addAction(setting_action)

        menu.addSeparator()

        about_action = QAction("About NectarSTT", self)
        about_action.triggered.connect(self.show_about_dialog)
        menu.addAction(about_action)

        menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.fade_close)
        menu.addAction(exit_action)

        cursor_pos = QCursor.pos()
        menu.exec(cursor_pos)

    def fade_close(self):

        fade_out = QPropertyAnimation(self, b"windowOpacity")
        fade_out.setDuration(600)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)
        fade_out.finished.connect(self.close)
        fade_out.start()

        # Keep reference alive
        self.fade_out_animation = fade_out

    def show_about_dialog(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("About NectarSTT")

        # Set message text
        msg.setText(
            "NectarSTT\n\n"
            "Version 1.0\n"
            "Developed by Samuel Ikenna Great\n"
            "This application provides speech-to-text and text-to-speech functionality "
            "for Nectar-X-Studio, using Whisper and Piper."
        )

        # Load custom icon
        icon_path = find_MainFolder("Main-Engine/Images/icon/icon.png")
        msg.setIconPixmap(QPixmap(icon_path).scaled(64, 64))

        # Optional: set window icon
        msg.setWindowIcon(QIcon(icon_path))

        msg.exec()

    def open_settings(self):
        # Prevent reopening multiple times
        if hasattr(self, "settings_widget") and self.settings_widget.isVisible():
            return

        # Create a small internal settings widget
        self.settings_widget = QWidget(self)
        self.settings_widget.setWindowTitle("Self Service")
        self.settings_widget.resize(500, 550)
        self.settings_widget.setWindowFlags(Qt.WindowType.Dialog)
        #self.settings_widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.settings_widget.setStyleSheet("""
            QWidget {
                background-color: #000000;
                border: transparent;
                border-radius: 12px;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QCheckBox {
                color: white;
                font-size: 13px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #2d2d2d;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 6px;
                font-size: 14px;
                color: #ffffff;
            }

            QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #5a5a5a;
            }

            QComboBox {
                background: #2d2d2d;
                border-radius: 6px;
                padding: 8px;
                font-size: 16px;
                color: #ffffff;
            }

            QComboBox::drop-down {
                border: none;
                background: #000000; 
                border-radius: 6px;
                width: 25px;
            }
                                
            QComboBox QAbstractItemView {
                background-color: #000000;
                color: white;
                selection-background-color: #000111;
                selection-color: #ffffff;
            }
            
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                background-color: #000000;
                border-left: 1px solid #000000;
                border-top-right-radius: 6px;
            }

            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                background-color: #000000;
                border-left: 1px solid #000000;
                border-bottom-right-radius: 6px;
            }
            QDoubleSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover {
                background-color: #2d2d2d;        /* Green hover effect */
            }
            /* UP BUTTON (QSpinBox + QDoubleSpinBox) */
            QSpinBox::up-button,
            QDoubleSpinBox::up-button {
                image: url("Main-Engine/Images/down-arrow/up-arrow.svg");
                width: 16px;
                height: 16px;
            }

            /* DOWN BUTTON (QSpinBox + QDoubleSpinBox) */
            QSpinBox::down-button,
            QDoubleSpinBox::down-button {
                image: url("Main-Engine/Images/down-arrow/down-arrow.svg");
                width: 16px;
                height: 16px;
            }
        """)
        
        layout = QVBoxLayout(self.settings_widget)
        #layout.setContentsMargins(15, 15, 15, 15)

        # --- Path fields ---
        #self.add_path_field(layout, "EXE_PATH", Config.EXE_PATH)
        #self.add_path_field(layout, "MODEL_PATH", Config.MODEL_PATH)
        self.add_model_dropdown(layout)
        #self.add_path_field(layout, "PIPER_EXE_PATH", Config.PIPER_EXE_PATH)
        #self.add_path_field(layout, "SOUND_EFFECTS_PATH", Config.SOUND_EFFECTS_PATH)
        #self.add_path_field(layout, "menu_btn_icon", Config.menu_btn_icon)
        #self.add_path_field(layout, "menu_hover_btn_icon", Config.menu_hover_btn_icon)

        # --- Numeric fields ---
        self.add_float_field(layout, "VOICE_SPEED", Config.VOICE_SPEED)
        self.add_int_field(layout, "FS", Config.FS)
        self.add_int_field(layout, "SILENCE_THRESHOLD", Config.SILENCE_THRESHOLD)
        self.add_int_field(layout, "CHUNK_SLEEP_MS", Config.CHUNK_SLEEP_MS)
        self.add_float_field(layout, "MIN_SPEECH_DURATION", Config.MIN_SPEECH_DURATION)

        # --- Save Button ---
        save_btn = QPushButton("Save Config and Close")
        layout.addWidget(save_btn)
        save_btn.clicked.connect(self.save_settings_and_close)

        self.layout().addWidget(self.settings_widget)

        # Position it at the center of the main window
        parent_rect = self.geometry()
        x = parent_rect.x() + (parent_rect.width() - self.settings_widget.width()) // 8
        y = parent_rect.y() + (parent_rect.height() - self.settings_widget.height()) // 2
        self.settings_widget.move(680, y)

        # Fade-in animation
        self.settings_widget.setWindowOpacity(0.0)
        self.settings_widget.show()

        fade_in = QPropertyAnimation(self.settings_widget, b"windowOpacity")
        fade_in.setDuration(600)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.Type.InOutQuad)
        fade_in.start()

        # Keep reference so animation doesn’t get garbage collected
        self.fade_in_animation = fade_in

    def save_settings_and_close(self):
        Config.save()
        if hasattr(self, "settings_widget"):
            self.settings_widget.close()
        Notify("Settings saved!", parent=self)

    # ---------------- Helper functions ----------------
    def add_path_field(self, layout, attr_name, value):
        label = QLabel(attr_name)
        layout.addWidget(label)

        hlayout = QHBoxLayout()
        line_edit = QLineEdit(value)
        hlayout.addWidget(line_edit)

        browse_btn = QPushButton("Browse")
        hlayout.addWidget(browse_btn)

        layout.addLayout(hlayout)

        # Update Config in real time
        line_edit.textChanged.connect(lambda text: setattr(Config, attr_name, text))
        browse_btn.clicked.connect(lambda: self.browse_file(line_edit, attr_name))

    def browse_file(self, line_edit, attr_name):
        path, _ = QFileDialog.getOpenFileName(self, f"Select {attr_name}")
        if path:
            line_edit.setText(path)
            setattr(Config, attr_name, path)

    def add_float_field(self, layout, attr_name, value):
        label = QLabel(attr_name)
        layout.addWidget(label)
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setSingleStep(0.1)
        spin.setValue(value)
        spin.setMaximum(10000)  # arbitrary large max
        layout.addWidget(spin)
        spin.valueChanged.connect(lambda val: setattr(Config, attr_name, val))

    def add_int_field(self, layout, attr_name, value):
        label = QLabel(attr_name)
        layout.addWidget(label)
        spin = QSpinBox()
        spin.setMaximum(1000000)  # arbitrary large max
        spin.setValue(value)
        layout.addWidget(spin)
        spin.valueChanged.connect(lambda val: setattr(Config, attr_name, val))

    def add_model_dropdown(self, layout):
        label = QLabel("Voice Model")
        layout.addWidget(label)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(ONNX_MODELS.keys())
        layout.addWidget(self.model_dropdown)

        # Set initial dropdown selection based on current Config value
        current_path = Config.ONNEX_MODEL_PATH
        for name, path in ONNX_MODELS.items():
            if path == current_path:
                self.model_dropdown.setCurrentText(name)

        # When user selects a model → update Config
        self.model_dropdown.currentTextChanged.connect(self.update_model_path)

    def update_model_path(self, model_name):
        selected_path = ONNX_MODELS[model_name]
        Config.ONNEX_MODEL_PATH = selected_path

    # -------------------------------------
    # CLICK TOGGLE HANDLER
    # -------------------------------------
    def on_dot_clicked(self):
        if not self.listening:
            self.activate_button()
        else:
            self.deactivate_button()

    # -------------------------------------
    # ACTIVATION FUNCTION
    # -------------------------------------
    def activate_button(self):
        self.listening = True
        self.dot.set_active(True)
        self.stop_recording = False

        # Start listening in a thread so UI stays responsive
        import threading
        threading.Thread(target=self._continuous_listening, daemon=True).start()

    # -------------------------------------
    # DEACTIVATION FUNCTION
    # -------------------------------------
    def deactivate_button(self):
        self.listening = False
        self.dot.set_active(False)

        self.stop_recording = True

    def _continuous_listening(self):
        text = self.record_until_silence_ui_controlled()

        if text:
            self.textout.setText(text)
            response = send_to_AlphaLLM(text)
            self.textout.setText(response)

            threading.Thread(target=self.speak_response, args=(response,), daemon=True).start()
            
    def speak_response(self, text):
        self.worker = SpeechWorker(text)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)

        self.worker.finished.connect(self.play_audio)
        self.worker.error.connect(lambda e: self.textout.setText(f"Voice error: {e}"))
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def play_audio(self, file_path):
        """Play synthesized MP3 in the main thread."""
        try:
            self.player = QMediaPlayer()
            self.audio_output = QAudioOutput()
            self.player.setAudioOutput(self.audio_output)
            self.player.setSource(QUrl.fromLocalFile(file_path))
            self.audio_output.setVolume(1.0)
            self.player.play()

            # Wait for playback to end before deleting
            self.player.mediaStatusChanged.connect(
                lambda status: self._delete_file_after_playback(status, file_path)
            )
        except Exception as e:
            Notify(f"Error playing voice: {e}", parent=self)

    def _delete_file_after_playback(self, status, file_path):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            # Delay deletion slightly to ensure file is released
            QTimer.singleShot(100, lambda: self._safe_remove(file_path))

    def _safe_remove(self, file_path):
        try:
            os.remove(file_path)
            Notify(f"Deleted temp file: {file_path}", parent=self)
        except PermissionError:
            # If still locked, try again shortly
            QTimer.singleShot(200, lambda: self._safe_remove(file_path))

    def record_until_silence_ui_controlled(self):
        """Record from microphone until the user clicks the dot to stop, return clean text."""
        self.textout.setText("\nListening... Speak now.")

        recording = []
        speech_detected = False
        frames_recorded = 0

        def callback(indata, frames, time, status):
            nonlocal recording, speech_detected, frames_recorded
            if status:
                self.textout.setText(f"InputStream warning: {status}")
            recording.append(indata.copy())
            frames_recorded += frames

            energy = np.abs(indata).mean() * 32767
            if energy >= Config.SILENCE_THRESHOLD:
                speech_detected = True

        try:
            with sd.InputStream(channels=1, samplerate=Config.FS, callback=callback):
                while not self.stop_recording:
                    sd.sleep(Config.CHUNK_SLEEP_MS)
        except Exception as e:
            self.textout.setText(f"Error accessing microphone: {e}")
            return None

        self.textout.setText("Recording stopped. Processing...")

        if not speech_detected or frames_recorded / Config.FS < Config.MIN_SPEECH_DURATION:
            self.textout.setText("No speech detected.")
            return None

        # Process audio
        audio = np.concatenate(recording, axis=0)
        audio_int16 = (audio * 32767).astype(np.int16)

        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                write(tmp.name, Config.FS, audio_int16)
                tmp_file = tmp.name

            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(
                [EXE_PATH, "-m", MODEL_PATH, "-f", tmp_file],
                capture_output=True,
                text=True,
                creationflags=creationflags
            )

            if result.returncode != 0:
                self.textout.setText("Transcription error:", result.stderr.strip())
                return None
            else:
                # CLEAN OUTPUT: remove timestamps and blank audio lines
                lines = result.stdout.strip().splitlines()
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    if not line or '[BLANK_AUDIO]' in line:
                        continue
                    # Remove timestamp using regex
                    line = re.sub(r'^\[.*?\]\s*', '', line)
                    clean_lines.append(line)
                return '  '.join(clean_lines)  # join sentences with double space

        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.dot.update()

    def showEvent(self, event):
        screen_geometry = self.screen().geometry()
        start_x = -self.width()
        start_y = (screen_geometry.height() - self.height()) // 2
        self.move(start_x, start_y)

        end_x = (screen_geometry.width() - self.width()) // 2
        end_y = start_y

        # Slide-in animation with single bounce
        self.pos_anim = QPropertyAnimation(self, b"pos")
        self.pos_anim.setDuration(1500)  # slower slide
        self.pos_anim.setStartValue(QPoint(start_x, start_y))
        self.pos_anim.setKeyValueAt(0.8, QPoint(end_x + 50, end_y))  # bounce
        self.pos_anim.setEndValue(QPoint(end_x, end_y))
        self.pos_anim.setEasingCurve(QEasingCurve.Type.OutBounce)
        self.pos_anim.start()

        # Play sound immediately after bounce ends
        QTimer.singleShot(1600, self.player.play)  # slightly after animation ends

        super().showEvent(event)

    def cleanup_player(self, status):
        from PyQt6.QtMultimedia import QMediaPlayer
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.stop()
            self.player.deleteLater()  # free resources
            self.audio_output.deleteLater()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Main()
    w.show()
    sys.exit(app.exec())
