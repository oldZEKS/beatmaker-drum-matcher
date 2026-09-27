"""
gui.py - Beatmaker Drum Matcher V2 desktop interface.
"""

import os
import sys
import tempfile
import winsound
import sqlite3

import numpy as np

from PySide6.QtCore import Qt, QUrl, QPoint, QThread, Signal
from PySide6.QtGui import QPainter, QColor, QFont, QDrag, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QScrollArea,
    QFrame, QProgressBar, QMessageBox
)

from audio_features import load_audio, detect_all_onsets
from matcher import match_sample, DB_NAME
from indexer import index_folder


STYLE_SHEET = """
QMainWindow { background-color: #121316; }
QWidget { color:#E2E8F0; font-family:'Segoe UI'; font-size:13px; }
QFrame#Card { background-color:#1A1D24; border:1px solid #2B303C; border-radius:8px; }
QFrame#MatchItem { background-color:#1E222B; border:1px solid #2B303C; border-radius:6px; }
QFrame#MatchItem:hover { background-color:#252A36; border:1px solid #00E599; }
QLabel#Title { font-size:18px; font-weight:700; color:#FFFFFF; }
QLabel#Subtitle { font-size:12px; color:#718096; }
QLabel#MetricLabel { font-size:11px; color:#A0AEC0; }
QLabel#MetricVal { font-size:13px; font-weight:600; color:#FFFFFF; }
QPushButton { background-color:#2B303C; border:1px solid #3E4656; border-radius:5px; padding:6px 14px; font-weight:600; color:#E2E8F0; }
QPushButton:hover { background-color:#384050; border-color:#00E599; color:#FFFFFF; }
QPushButton#PrimaryBtn, QPushButton#MatchSliceBtn { background-color:#00E599; color:#0B0E14; border:none; font-weight:bold; }
QPushButton#ModifierBtn { background-color:#1E222B; border:1px solid #323846; padding:4px 10px; font-size:12px; }
QPushButton#ModifierBtn:checked { background-color:#0077ED; border-color:#3399FF; color:#FFFFFF; }
QPushButton#DebleedBtn { background-color:#1E222B; border:1px solid #6B46C1; color:#D6BCFA; font-size:12px; padding:4px 10px; }
QPushButton#DebleedBtn:checked { background-color:#6B46C1; border-color:#9F7AEA; color:#FFFFFF; }
QPushButton#PlayBtn { background-color:#00E599; color:#0A0D12; border-radius:14px; min-width:28px; max-width:28px; min-height:28px; max-height:28px; padding:0; }
QComboBox { background-color:#2B303C; border:1px solid #3E4656; border-radius:4px; padding:4px 8px; color:#FFFFFF; min-width:100px; }
QComboBox QAbstractItemView { background-color:#1E222B; selection-background-color:#00E599; selection-color:#0A0D12; }
QProgressBar { background-color:#13151A; border:1px solid #2B303C; border-radius:3px; text-align:center; font-size:10px; height:12px; }
QProgressBar::chunk { background-color:#00E599; border-radius:2px; }
QScrollArea { border:none; background-color:transparent; }
"""


class WaveformWidget(QWidget):
    slice_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.audio_data = None
        self.sr = 44100
        self.slices = []
        self.selected_slice_idx = 0
        self.setMouseTracking(True)

    def set_audio_and_slices(self, audio_data, sr, slices):
        self.audio_data, self.sr, self.slices = audio_data, sr, slices
        self.selected_slice_idx = 0
        self.update()

    def set_selected_slice(self, idx):
        self.selected_slice_idx = idx
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.audio_data is None:
            return
        width = max(1, self.width())
        sample = int(event.pos().x() / width * len(self.audio_data))
        for i, (start, end) in enumerate(self.slices):
            if start <= sample < end:
                self.slice_clicked.emit(i)
                return

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101217"))
        if self.audio_data is None or len(self.audio_data) == 0:
            painter.setPen(QColor("#718096"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Drop a song, loop or drum hit here")
            return

        data = np.asarray(self.audio_data)
        w, h = self.width(), self.height()
        bins = max(1, w)
        step = max(1, len(data) // bins)
        mid = h // 2
        painter.setPen(QColor("#00E599"))

        for x in range(w):
            chunk = data[x * step:min(len(data), (x + 1) * step)]
            if len(chunk) == 0:
                continue
            amp = float(np.max(np.abs(chunk)))
            y = int(amp * (h * 0.42))
            painter.drawLine(x, mid - y, x, mid + y)

        for i, (start, end) in enumerate(self.slices):
            x1 = int(start / len(data) * w)
            x2 = int(end / len(data) * w)
            if i == self.selected_slice_idx:
                painter.fillRect(x1, 0, max(2, x2 - x1), h, QColor(0, 229, 153, 38))
            painter.setPen(QColor(80, 90, 105))
            painter.drawLine(x1, 0, x1, h)


class MatchItemWidget(QFrame):
    def __init__(self, rank, score, cand, sims, parent=None):
        super().__init__(parent)
        self.setObjectName("MatchItem")
        self.rank = rank
        self.score = score
        self.cand = cand
        self.sims = sims
        self.sample_path = cand["path"]
        self.drag_start_pos = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(12)

        score_box = QVBoxLayout()
        rank_lbl = QLabel(f"#{rank}")
        rank_lbl.setStyleSheet("font-size:11px;color:#718096;font-weight:bold;")
        score_lbl = QLabel(f"{score*100:.1f}%")
        score_lbl.setStyleSheet("font-size:15px;font-weight:800;color:#00E599;")
        score_box.addWidget(rank_lbl)
        score_box.addWidget(score_lbl)
        layout.addLayout(score_box)

        info = QVBoxLayout()
        name = QLabel(cand["filename"])
        name.setStyleSheet("font-weight:600;font-size:13px;color:#FFFFFF;")
        type_badge = "LOOP" if cand.get("is_loop", 0) else "1-SHOT"
        meta = QLabel(
            f"Kit: {cand.get('kit_name','?')} • {type_badge} • "
            f"{cand.get('category','other').upper()}"
        )
        meta.setStyleSheet("font-size:11px;color:#A0AEC0;")
        info.addWidget(name)
        info.addWidget(meta)
        layout.addLayout(info, stretch=2)

        self._add_meter(layout, "Attack", sims["sim_attack"])
        self._add_meter(layout, "Body", sims["sim_body"])
        self._add_meter(layout, "Tail", sims["sim_tail"])
        self._add_meter(layout, "Timbre", sims["sim_mel24"])
        self._add_meter(layout, "Role", sims["role_compatibility"])

        tuning = sims.get("tuning_cents", 0)
        tuning_lbl = QLabel(
            "IN KEY" if abs(tuning) < 15 else f"{tuning/100:+.1f} st"
        )
        tuning_lbl.setAlignment(Qt.AlignCenter)
        tuning_lbl.setStyleSheet(
            "padding:3px 8px;border-radius:4px;font-weight:bold;font-size:11px;"
            "background:#064E3B;color:#34D399;border:1px solid #059669;"
            if abs(tuning) < 15 else
            "padding:3px 8px;border-radius:4px;font-weight:bold;font-size:11px;"
            "background:#451A03;color:#FBBF24;border:1px solid #D97706;"
        )
        layout.addWidget(tuning_lbl)

        play = QPushButton("▶")
        play.setObjectName("PlayBtn")
        play.clicked.connect(self.play_sample)
        play.setToolTip("Preview sample")
        layout.addWidget(play)

        drag_hint = QLabel("DRAG")
        drag_hint.setAlignment(Qt.AlignCenter)
        drag_hint.setStyleSheet(
            "background:#252A36;color:#A0AEC0;border:1px dashed #3E4656;"
            "padding:6px 10px;border-radius:4px;font-size:10px;font-weight:bold;"
        )
        layout.addWidget(drag_hint)

    def _add_meter(self, parent, label, value):
        box = QVBoxLayout()
        lab = QLabel(label)
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet("font-size:10px;color:#718096;")
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(np.clip(value, 0, 1) * 100))
        bar.setFormat(f"{int(np.clip(value, 0, 1) * 100)}%")
        bar.setFixedWidth(50)
        box.addWidget(lab)
        box.addWidget(bar)
        parent.addLayout(box)

    def play_sample(self):
        try:
            winsound.PlaySound(self.sample_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            print(f"Playback error: {exc}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton) or self.drag_start_pos is None:
            return
        if (event.pos() - self.drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return

        drag = QDrag(self)
        from PySide6.QtCore import QMimeData
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(self.sample_path)])
        drag.setMimeData(mime)

        pixmap = QPixmap(140, 32)
        pixmap.fill(QColor("#1A1D24"))
        painter = QPainter(pixmap)
        painter.setPen(QColor("#00E599"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, self.cand["filename"][:18])
        painter.end()
        drag.setPixmap(pixmap)
        drag.exec(Qt.CopyAction)


class BackgroundIndexer(QThread):
    finished = Signal(int)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        try:
            count = index_folder(self.folder_path)
        except Exception as exc:
            print(exc)
            count = 0
        self.finished.emit(count)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beatmaker Drum Matcher — V2")
        self.resize(1250, 880)
        self.setAcceptDrops(True)

        self.full_audio_data = None
        self.current_ref_path = None
        self.current_slice_audio = None
        self.sr = 44100
        self.slices = []
        self.current_slice_idx = 0
        self.current_modifier = None
        self.apply_debleed = False
        self.temp_slice_wav = None

        self._build_ui()
        self._refresh_stats()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(18, 16, 18, 16)
        main.setSpacing(12)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("BEATMAKER DRUM MATCHER")
        title.setObjectName("Title")
        subtitle = QLabel("Find the sample you'd actually use — not just the most acoustically similar file.")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        self.focus_btn = QPushButton("Percussive Focus")
        self.focus_btn.setObjectName("DebleedBtn")
        self.focus_btn.setCheckable(True)
        self.focus_btn.setToolTip("Reduce sustained/harmonic bleed before analysing the selected hit.")
        self.focus_btn.clicked.connect(self._rerun)
        header.addWidget(self.focus_btn)

        index_btn = QPushButton("+ Index Drum Kit")
        index_btn.clicked.connect(self._browse_index)
        header.addWidget(index_btn)

        browse = QPushButton("Open Song / Audio")
        browse.setObjectName("PrimaryBtn")
        browse.clicked.connect(self._browse_audio)
        header.addWidget(browse)
        main.addLayout(header)

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)

        self.waveform = WaveformWidget()
        self.waveform.slice_clicked.connect(self.select_slice)
        card_layout.addWidget(self.waveform)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.clicked.connect(self.prev_slice)
        self.next_btn = QPushButton("▶")
        self.next_btn.clicked.connect(self.next_slice)
        nav.addWidget(self.prev_btn)
        self.slice_lbl = QLabel("No audio loaded")
        self.slice_lbl.setStyleSheet("font-weight:bold;color:#00E599;")
        nav.addWidget(self.slice_lbl)
        nav.addWidget(self.next_btn)
        nav.addSpacing(12)

        play = QPushButton("Play Hit (Space)")
        play.clicked.connect(self.play_current_slice)
        nav.addWidget(play)

        play_full = QPushButton("Play Full (R)")
        play_full.clicked.connect(self.play_full)
        nav.addWidget(play_full)

        nav.addStretch()

        self.match_btn = QPushButton("MATCH SELECTED HIT")
        self.match_btn.setObjectName("MatchSliceBtn")
        self.match_btn.clicked.connect(self._rerun)
        nav.addWidget(self.match_btn)
        card_layout.addLayout(nav)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Role:"))
        self.cat_combo = QComboBox()
        self.cat_combo.addItems(["Auto Detect", "Kick", "808", "Snare", "Clap", "Rim", "Hat", "Perc"])
        self.cat_combo.currentIndexChanged.connect(self._rerun)
        controls.addWidget(self.cat_combo)

        controls.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["One-Shots Only", "Loops Only", "All Samples"])
        self.type_combo.currentIndexChanged.connect(self._rerun)
        controls.addWidget(self.type_combo)

        controls.addWidget(QLabel("Preference:"))
        self.mod_combo = QComboBox()
        self.mod_combo.addItems(["Standard", "Punchier", "Tighter", "Darker", "Brighter"])
        self.mod_combo.currentIndexChanged.connect(self._rerun)
        controls.addWidget(self.mod_combo)

        self.role_metric = QLabel("Role: —")
        self.role_metric.setStyleSheet("font-weight:bold;color:#00E599;")
        controls.addWidget(self.role_metric)
        controls.addStretch()

        card_layout.addLayout(controls)
        main.addWidget(card)

        self.results = QScrollArea()
        self.results.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(8)
        self.results_layout.addStretch()
        self.results.setWidget(self.results_container)
        main.addWidget(self.results, stretch=1)

        self.status = QLabel("Ready — index your drum kits, then drop a song or drum loop.")
        self.status.setStyleSheet("color:#718096;font-size:12px;")
        main.addWidget(self.status)

    def _browse_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select song, loop or drum hit", "",
            "Audio (*.wav *.flac *.aif *.aiff *.mp3 *.ogg *.m4a);;All files (*.*)"
        )
        if path:
            self.load_reference(path)

    def _browse_index(self):
        folder = QFileDialog.getExistingDirectory(self, "Select drum/sample library")
        if folder:
            self.status.setText("Indexing sample library…")
            self.indexer = BackgroundIndexer(folder)
            self.indexer.finished.connect(self._index_done)
            self.indexer.start()

    def _index_done(self, count):
        self.status.setText(f"Indexed {count} new samples.")
        self._refresh_stats()
        self._rerun()

    def _refresh_stats(self):
        if not os.path.exists(DB_NAME):
            self.status.setText("No library indexed yet.")
            return
        try:
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM samples")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT kit_name) FROM samples")
            kits = cur.fetchone()[0]
            conn.close()
            self.status.setText(f"Library: {total:,} samples across {kits:,} kits.")
        except sqlite3.Error:
            pass

    def load_reference(self, path):
        try:
            self.status.setText(f"Analysing {os.path.basename(path)}…")
            data, sr = load_audio(path)
            self.full_audio_data = data
            self.sr = sr
            self.current_ref_path = path
            self.slices = detect_all_onsets(data, sr)
            self.current_slice_idx = 0
            self.waveform.set_audio_and_slices(data, sr, self.slices)
            self.select_slice(0)
        except Exception as exc:
            QMessageBox.critical(self, "Could not load audio", str(exc))

    def select_slice(self, idx):
        if not self.slices or not (0 <= idx < len(self.slices)):
            return
        self.current_slice_idx = idx
        self.waveform.set_selected_slice(idx)
        start, end = self.slices[idx]
        self.current_slice_audio = self.full_audio_data[start:end]
        self.slice_lbl.setText(
            f"Hit #{idx + 1}/{len(self.slices)} "
            f"({start/self.sr:.2f}s–{end/self.sr:.2f}s)"
        )
        self._rerun()

    def _rerun(self):
        if self.current_slice_audio is None:
            return

        category = self.cat_combo.currentText().lower()
        category = None if category == "auto detect" else category
        sample_type = {
            "One-Shots Only": "oneshot",
            "Loops Only": "loop",
            "All Samples": "all",
        }[self.type_combo.currentText()]
        modifier = {
            "Standard": None,
            "Punchier": "punchier",
            "Tighter": "tighter",
            "Darker": "darker",
            "Brighter": "brighter",
        }[self.mod_combo.currentText()]

        try:
            matches, ref = match_sample(
                self.current_slice_audio,
                category=category,
                top_k=25,
                modifier=modifier,
                apply_debleed=self.focus_btn.isChecked(),
                sample_type=sample_type,
            )
            self.role_metric.setText(
                f"Role: {ref['category'].upper()} "
                f"({ref['category_confidence']*100:.0f}%)"
            )
            self._populate(matches)
            self.status.setText(
                f"{len(matches)} candidates • {ref['category'].upper()} reference"
            )
        except Exception as exc:
            self.status.setText(f"Match error: {exc}")

    def _populate(self, matches):
        while self.results_layout.count() > 1:
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for rank, (score, cand, sims) in enumerate(matches, 1):
            self.results_layout.insertWidget(
                rank - 1, MatchItemWidget(rank, score, cand, sims)
            )

    def play_current_slice(self):
        if self.current_slice_audio is None:
            return
        try:
            if not self.temp_slice_wav:
                self.temp_slice_wav = os.path.join(
                    tempfile.gettempdir(), "beatmaker_matcher_slice.wav"
                )
            import soundfile as sf
            sf.write(self.temp_slice_wav, self.current_slice_audio, self.sr)
            winsound.PlaySound(
                self.temp_slice_wav, winsound.SND_FILENAME | winsound.SND_ASYNC
            )
        except Exception as exc:
            self.status.setText(f"Playback error: {exc}")

    def play_full(self):
        if self.current_ref_path:
            try:
                winsound.PlaySound(
                    self.current_ref_path, winsound.SND_FILENAME | winsound.SND_ASYNC
                )
            except Exception as exc:
                self.status.setText(f"Playback error: {exc}")

    def prev_slice(self):
        if self.slices:
            self.select_slice((self.current_slice_idx - 1) % len(self.slices))

    def next_slice(self):
        if self.slices:
            self.select_slice((self.current_slice_idx + 1) % len(self.slices))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.play_current_slice()
        elif event.key() == Qt.Key_Left:
            self.prev_slice()
        elif event.key() == Qt.Key_Right:
            self.next_slice()
        elif event.key() == Qt.Key_R:
            self.play_full()
        else:
            super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                self.load_reference(path)
                break


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
