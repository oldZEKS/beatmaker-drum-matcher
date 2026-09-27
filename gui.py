"""
gui.py - Beatmaker Drum Matcher with In-Loop Transient Slicer & 24-Band Mel Micro-Timbre.
"""

import os
import sys
import tempfile
import winsound
import numpy as np
import sqlite3

from PySide6.QtCore import Qt, QUrl, QPoint, QThread, Signal
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient, QFont,
    QDrag, QPixmap
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QScrollArea,
    QFrame, QProgressBar, QMessageBox
)

from audio_features import (
    extract_features, load_audio, debleed_audio, detect_all_onsets
)
from matcher import match_sample, DB_NAME
from indexer import index_folder, DB_NAME


STYLE_SHEET = """
QMainWindow {
    background-color: #121316;
}
QWidget {
    color: #E2E8F0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}
QFrame#Card {
    background-color: #1A1D24;
    border: 1px solid #2B303C;
    border-radius: 8px;
}
QFrame#MatchItem {
    background-color: #1E222B;
    border: 1px solid #2B303C;
    border-radius: 6px;
}
QFrame#MatchItem:hover {
    background-color: #252A36;
    border: 1px solid #00E599;
}
QLabel#Title {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
}
QLabel#Subtitle {
    font-size: 12px;
    color: #718096;
}
QLabel#MetricLabel {
    font-size: 11px;
    color: #A0AEC0;
}
QLabel#MetricVal {
    font-size: 13px;
    font-weight: 600;
    color: #FFFFFF;
}
QPushButton {
    background-color: #2B303C;
    border: 1px solid #3E4656;
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: 600;
    color: #E2E8F0;
}
QPushButton:hover {
    background-color: #384050;
    border-color: #00E599;
    color: #FFFFFF;
}
QPushButton:pressed {
    background-color: #1F232B;
}
QPushButton#PrimaryBtn {
    background-color: #00E599;
    color: #0B0E14;
    border: none;
}
QPushButton#PrimaryBtn:hover {
    background-color: #00FFAC;
}
QPushButton#MatchSliceBtn {
    background-color: #00E599;
    color: #0B0E14;
    border: none;
    font-weight: bold;
    padding: 6px 16px;
}
QPushButton#MatchSliceBtn:hover {
    background-color: #00FFAC;
}
QPushButton#ModifierBtn {
    background-color: #1E222B;
    border: 1px solid #323846;
    padding: 4px 10px;
    font-size: 12px;
}
QPushButton#ModifierBtn:checked {
    background-color: #0077ED;
    border-color: #3399FF;
    color: #FFFFFF;
}
QPushButton#DebleedBtn {
    background-color: #1E222B;
    border: 1px solid #6B46C1;
    color: #D6BCFA;
    font-size: 12px;
    padding: 4px 10px;
}
QPushButton#DebleedBtn:checked {
    background-color: #6B46C1;
    border-color: #9F7AEA;
    color: #FFFFFF;
}
QPushButton#PlayBtn {
    background-color: #00E599;
    color: #0A0D12;
    border-radius: 14px;
    font-size: 12px;
    font-weight: bold;
    min-width: 28px;
    max-width: 28px;
    min-height: 28px;
    max-height: 28px;
    padding: 0px;
}
QPushButton#PlayBtn:hover {
    background-color: #00FFAC;
}
QComboBox {
    background-color: #2B303C;
    border: 1px solid #3E4656;
    border-radius: 4px;
    padding: 4px 8px;
    color: #FFFFFF;
    min-width: 100px;
}
QComboBox QAbstractItemView {
    background-color: #1E222B;
    selection-background-color: #00E599;
    selection-color: #0A0D12;
    border: 1px solid #3E4656;
}
QProgressBar {
    background-color: #13151A;
    border: 1px solid #2B303C;
    border-radius: 3px;
    text-align: center;
    font-size: 10px;
    height: 12px;
}
QProgressBar::chunk {
    background-color: #00E599;
    border-radius: 2px;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
"""


class WaveformWidget(QWidget):
    slice_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.audio_data = None
        self.sr = 44100
        self.slices = []  # list of (start_idx, end_idx)
        self.selected_slice_idx = 0
        self.is_drag_active = False
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

    def set_audio_and_slices(self, audio_data, sr=44100, slices=None):
        self.audio_data = audio_data
        self.sr = sr
        self.slices = slices if slices else [(0, len(audio_data))]
        self.selected_slice_idx = 0
        self.update()

    def set_selected_slice(self, idx):
        if 0 <= idx < len(self.slices):
            self.selected_slice_idx = idx
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.audio_data is not None and len(self.audio_data) > 0:
            w = self.width()
            click_x = event.pos().x()
            click_sample = int((click_x / float(w)) * len(self.audio_data))

            for i, (start, end) in enumerate(self.slices):
                if start <= click_sample < end or (i == len(self.slices) - 1 and click_sample >= start):
                    self.selected_slice_idx = i
                    self.slice_clicked.emit(i)
                    self.update()
                    break
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        bg_color = QColor("#161920") if not self.is_drag_active else QColor("#1C2522")
        painter.setBrush(QBrush(bg_color))
        border_pen = QPen(QColor("#00E599" if self.is_drag_active else "#2B303C"), 1.5)
        painter.setPen(border_pen)
        painter.drawRoundedRect(1, 1, w - 2, h - 2, 6, 6)

        if self.audio_data is None or len(self.audio_data) == 0:
            painter.setPen(QColor("#5A6578"))
            font = QFont("Segoe UI", 12, QFont.Bold)
            painter.setFont(font)
            text = "DROP DRUM LOOP OR HIT (WAV / FLAC / AIFF) or CLICK BROWSE"
            painter.drawText(0, 0, w, h, Qt.AlignCenter, text)
            return

        total_samples = len(self.audio_data)

        # Highlight Selected Slice Region with glowing overlay
        if 0 <= self.selected_slice_idx < len(self.slices):
            s_start, s_end = self.slices[self.selected_slice_idx]
            x_start = int((s_start / total_samples) * w)
            x_end = int((s_end / total_samples) * w)
            slice_w = max(4, x_end - x_start)

            highlight_brush = QBrush(QColor(0, 229, 153, 35))
            painter.fillRect(x_start, 2, slice_w, h - 4, highlight_brush)
            painter.setPen(QPen(QColor("#00E599"), 1.5))
            painter.drawRect(x_start, 2, slice_w, h - 4)

        # Waveform Drawing
        mid_y = h / 2.0
        painter.setPen(QPen(QColor("#242A36"), 1))
        painter.drawLine(0, int(mid_y), w, int(mid_y))

        n_points = max(10, w)
        chunk_size = max(1, total_samples // n_points)

        gradient = QLinearGradient(0, 0, 0, h)
        gradient.setColorAt(0.0, QColor("#00FFAC"))
        gradient.setColorAt(0.5, QColor("#00E599"))
        gradient.setColorAt(1.0, QColor("#00A36C"))
        pen = QPen(QBrush(gradient), 1.5)
        painter.setPen(pen)

        for col in range(n_points):
            start = col * chunk_size
            end = min(start + chunk_size, total_samples)
            if start >= end:
                continue
            slice_data = self.audio_data[start:end]
            max_val = np.max(slice_data)
            min_val = np.min(slice_data)

            y1 = mid_y - (max_val * (mid_y - 12))
            y2 = mid_y - (min_val * (mid_y - 12))
            painter.drawLine(col, int(y1), col, int(y2))

        # Vertical Slice Lines & Numbers
        if len(self.slices) > 1:
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            for i, (s_start, s_end) in enumerate(self.slices):
                x = int((s_start / total_samples) * w)
                is_sel = (i == self.selected_slice_idx)
                line_color = QColor("#00FFAC") if is_sel else QColor("#FF0055")
                painter.setPen(QPen(line_color, 1.5, Qt.DashLine if not is_sel else Qt.SolidLine))
                painter.drawLine(x, 4, x, h - 4)

                # Badge label
                painter.setPen(QColor("#FFFFFF" if is_sel else "#A0AEC0"))
                painter.drawText(x + 4, 18, f"#{i+1}")


class MatchItemWidget(QFrame):
    def __init__(self, rank, score, cand, sims, ref_path, parent=None):
        super().__init__(parent)
        self.setObjectName("MatchItem")
        self.rank = rank
        self.score = score
        self.cand = cand
        self.sims = sims
        self.sample_path = cand['path']
        self.ref_path = ref_path
        self.drag_start_pos = None

        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(14)

        # 1. Play Button
        self.play_btn = QPushButton("▶")
        self.play_btn.setObjectName("PlayBtn")
        self.play_btn.setToolTip("Audition Sample (or press Space)")
        self.play_btn.clicked.connect(self.play_sample)
        layout.addWidget(self.play_btn)

        # 2. Rank & Score Badge
        score_layout = QVBoxLayout()
        score_layout.setSpacing(1)
        rank_lbl = QLabel(f"#{self.rank}")
        rank_lbl.setStyleSheet("font-size: 11px; color: #718096; font-weight: bold;")
        
        pct = round(self.score * 100, 1)
        pct_lbl = QLabel(f"{pct}%")
        color = "#00E599" if pct >= 80 else ("#FFA726" if pct >= 65 else "#718096")
        pct_lbl.setStyleSheet(f"font-size: 15px; font-weight: 800; color: {color};")
        
        score_layout.addWidget(rank_lbl)
        score_layout.addWidget(pct_lbl)
        layout.addLayout(score_layout)

        # 3. Filename & Kit Details
        name_layout = QVBoxLayout()
        name_layout.setSpacing(2)
        name_lbl = QLabel(self.cand['filename'])
        name_lbl.setStyleSheet("font-weight: 600; font-size: 13px; color: #FFFFFF;")
        
        is_kick = self.cand['category'] in ('kick', '808')
        extra_info = ""
        if is_kick and self.cand.get('pitch_drop_st', 0) > 0:
            extra_info = f" • Sweep: -{self.cand['pitch_drop_st']} st"

        is_loop_val = self.cand.get('is_loop', 0)
        type_badge = "LOOP" if is_loop_val == 1 else "1-SHOT"
        type_color = "#FF0055" if is_loop_val == 1 else "#00E599"

        kit_lbl = QLabel(f"Kit: {self.cand['kit_name']} • <span style='color: {type_color}; font-weight: bold;'>{type_badge}</span> • {self.cand['category'].upper()}{extra_info}")
        kit_lbl.setTextFormat(Qt.RichText)
        kit_lbl.setStyleSheet("font-size: 11px; color: #A0AEC0;")
        
        name_layout.addWidget(name_lbl)
        name_layout.addWidget(kit_lbl)
        layout.addLayout(name_layout, stretch=2)


        # 4. Tuning Badge
        tuning = self.sims['tuning_cents']
        if abs(tuning) < 15:
            tune_text = "IN-KEY"
            tune_style = "background-color: #064E3B; color: #34D399; border: 1px solid #059669;"
        else:
            cents_st = round(tuning / 100.0, 1)
            sign = "+" if cents_st > 0 else ""
            tune_text = f"{sign}{cents_st} st"
            tune_style = "background-color: #451A03; color: #FBBF24; border: 1px solid #D97706;"

        tune_badge = QLabel(tune_text)
        tune_badge.setAlignment(Qt.AlignCenter)
        tune_badge.setStyleSheet(f"padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 11px; {tune_style}")
        layout.addWidget(tune_badge)

        # 5. Visual Breakdown Meters (Attack, Tone, Mel Timbre, Sustain, Decay)
        meters_layout = QHBoxLayout()
        meters_layout.setSpacing(8)

        self.add_meter(meters_layout, "Attack", self.sims['sim_attack'])
        self.add_meter(meters_layout, "Tone", self.sims['sim_pitch'])
        self.add_meter(meters_layout, "Mel-Timbre", self.sims['sim_mel24'])
        self.add_meter(meters_layout, "Sustain", self.sims['sim_sustain'])
        self.add_meter(meters_layout, "Decay", self.sims['sim_decay'])
        layout.addLayout(meters_layout, stretch=3)

        # 6. Drag Handle for DAW Export
        drag_hint = QLabel("⠿ DRAG TO DAW")
        drag_hint.setAlignment(Qt.AlignCenter)
        drag_hint.setStyleSheet(
            "background-color: #252A36; color: #A0AEC0; border: 1px dashed #3E4656; "
            "padding: 6px 10px; border-radius: 4px; font-weight: bold; font-size: 10px;"
        )
        drag_hint.setCursor(Qt.OpenHandCursor)
        drag_hint.setToolTip("Click and drag directly into FL Studio, Ableton, or Explorer")
        layout.addWidget(drag_hint)

    def add_meter(self, parent_layout, label_text, value):
        v = QVBoxLayout()
        v.setSpacing(2)
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-size: 10px; color: #718096;")
        lbl.setAlignment(Qt.AlignCenter)
        
        pbar = QProgressBar()
        pbar.setRange(0, 100)
        pbar.setValue(int(value * 100))
        pbar.setFormat(f"{int(value * 100)}%")
        pbar.setFixedWidth(52)
        
        v.addWidget(lbl)
        v.addWidget(pbar)
        parent_layout.addLayout(v)

    def play_sample(self):
        try:
            winsound.PlaySound(self.sample_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"Playback error: {e}")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton) or not self.drag_start_pos:
            return
        if (event.pos() - self.drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return

        drag = QDrag(self)
        from PySide6.QtCore import QMimeData
        mime = QMimeData()
        url = QUrl.fromLocalFile(self.sample_path)
        mime.setUrls([url])
        drag.setMimeData(mime)

        pixmap = QPixmap(140, 36)
        pixmap.fill(QColor("#1A1D24"))
        p = QPainter(pixmap)
        p.setPen(QColor("#00E599"))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(pixmap.rect(), Qt.AlignCenter, self.cand['filename'][:18])
        p.end()
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(70, 18))

        drag.exec(Qt.CopyAction)


class BackgroundIndexer(QThread):
    finished = Signal(int)

    def __init__(self, folder_path, db_path=DB_NAME):
        super().__init__()
        self.folder_path = folder_path
        self.db_path = db_path

    def run(self):
        count = index_folder(self.folder_path, db_path=self.db_path)
        self.finished.emit(count)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beatmaker Drum Matcher — Loop Slicer & 24-Band Mel Timbre")
        self.resize(1200, 860)
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

        self.init_ui()
        self.load_default_stats()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(14)

        # 1. Header Bar
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("BEATMAKER DRUM MATCHER")
        title.setObjectName("Title")
        sub = QLabel("In-Loop Slicer • 24-Band Mel Micro-Timbre • Direct DAW Drag-Out")
        sub.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(sub)
        header.addLayout(title_box)

        header.addStretch()

        self.debleed_btn = QPushButton("✨ De-Bleed (Song Slice)")
        self.debleed_btn.setObjectName("DebleedBtn")
        self.debleed_btn.setCheckable(True)
        self.debleed_btn.setToolTip("Attenuate background room reverb and hi-hat bleed before analyzing")
        self.debleed_btn.clicked.connect(self.toggle_debleed)
        header.addWidget(self.debleed_btn)

        self.add_kit_btn = QPushButton("+ Index Drum Kit Folder")
        self.add_kit_btn.setToolTip("Scan another drum folder into your database")
        self.add_kit_btn.clicked.connect(self.browse_and_index_kit)
        header.addWidget(self.add_kit_btn)

        self.browse_btn = QPushButton("Browse Audio File / Loop...")
        self.browse_btn.setObjectName("PrimaryBtn")
        self.browse_btn.clicked.connect(self.browse_reference_file)
        header.addWidget(self.browse_btn)

        main_layout.addLayout(header)

        # 2. Reference Hit / Loop Card
        ref_card = QFrame()
        ref_card.setObjectName("Card")
        ref_layout = QVBoxLayout(ref_card)
        ref_layout.setContentsMargins(14, 12, 14, 12)
        ref_layout.setSpacing(10)

        # Interactive Waveform Display with Slice Markers
        self.waveform = WaveformWidget()
        self.waveform.slice_clicked.connect(self.on_waveform_slice_clicked)
        ref_layout.addWidget(self.waveform)

        # Slicer Navigation Bar
        slice_nav = QHBoxLayout()
        slice_nav.setSpacing(8)

        self.prev_slice_btn = QPushButton("◀ Prev Hit")
        self.prev_slice_btn.clicked.connect(self.prev_slice)
        self.prev_slice_btn.setEnabled(False)
        slice_nav.addWidget(self.prev_slice_btn)

        self.slice_info_lbl = QLabel("No Audio Loaded")
        self.slice_info_lbl.setStyleSheet("font-weight: bold; color: #00E599; font-size: 13px;")
        slice_nav.addWidget(self.slice_info_lbl)

        self.next_slice_btn = QPushButton("Next Hit ▶")
        self.next_slice_btn.clicked.connect(self.next_slice)
        self.next_slice_btn.setEnabled(False)
        slice_nav.addWidget(self.next_slice_btn)

        slice_nav.addSpacing(15)

        self.play_slice_btn = QPushButton("▶ Play Hit (Space)")
        self.play_slice_btn.clicked.connect(self.play_current_slice)
        self.play_slice_btn.setEnabled(False)
        slice_nav.addWidget(self.play_slice_btn)

        self.play_full_btn = QPushButton("🔁 Play Full File (R)")
        self.play_full_btn.clicked.connect(self.play_full_audio)
        self.play_full_btn.setEnabled(False)
        slice_nav.addWidget(self.play_full_btn)

        slice_nav.addStretch()

        self.match_slice_btn = QPushButton("🎯 Match Selected Hit")
        self.match_slice_btn.setObjectName("MatchSliceBtn")
        self.match_slice_btn.clicked.connect(self.run_match_on_current_slice)
        self.match_slice_btn.setEnabled(False)
        slice_nav.addWidget(self.match_slice_btn)

        ref_layout.addLayout(slice_nav)

        # Controls & Metrics
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(14)

        ctrl_layout.addWidget(QLabel("Category:"))
        self.cat_combo = QComboBox()
        self.cat_combo.addItems(["Auto Detect", "Kick", "Snare", "Clap", "Rim", "Hat", "808", "Perc"])
        self.cat_combo.currentIndexChanged.connect(self.on_category_changed)
        ctrl_layout.addWidget(self.cat_combo)

        ctrl_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["One-Shots Only", "Loops Only", "All Samples"])
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        ctrl_layout.addWidget(self.type_combo)

        ctrl_layout.addSpacing(10)

        # Metrics display
        self.m_punch = self.create_metric_widget("Attack Punch", "—")
        self.m_rise = self.create_metric_widget("Rise-Time", "—")
        self.m_body = self.create_metric_widget("Body Tone", "—")
        self.m_decay = self.create_metric_widget("Decay T30", "—")
        self.m_noise = self.create_metric_widget("Noise / Wire", "—")

        ctrl_layout.addLayout(self.m_punch)
        ctrl_layout.addLayout(self.m_rise)
        ctrl_layout.addLayout(self.m_body)
        ctrl_layout.addLayout(self.m_decay)
        ctrl_layout.addLayout(self.m_noise)

        ctrl_layout.addStretch()
        ref_layout.addLayout(ctrl_layout)
        main_layout.addWidget(ref_card)


        # 3. Filter & Modifier Toolbar
        mod_bar = QHBoxLayout()
        mod_bar.setSpacing(8)
        mod_lbl = QLabel("Vibe Modifiers:")
        mod_lbl.setStyleSheet("font-weight: bold; color: #A0AEC0;")
        mod_bar.addWidget(mod_lbl)

        self.mod_btns = {}
        for mod_name, label in [
            (None, "Standard Match"),
            ("punchier", "+ Punchier Attack"),
            ("tighter", "+ Tighter Decay"),
            ("darker", "+ Darker"),
            ("brighter", "+ Brighter")
        ]:
            b = QPushButton(label)
            b.setObjectName("ModifierBtn")
            b.setCheckable(True)
            if mod_name is None:
                b.setChecked(True)
            b.clicked.connect(lambda checked, m=mod_name: self.set_modifier(m))
            self.mod_btns[mod_name] = b
            mod_bar.addWidget(b)

        mod_bar.addStretch()

        self.results_count_lbl = QLabel("Top Matches")
        self.results_count_lbl.setStyleSheet("font-weight: 600; color: #00E599;")
        mod_bar.addWidget(self.results_count_lbl)

        main_layout.addLayout(mod_bar)

        # 4. Results List (Scroll Area)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(8)
        self.results_layout.addStretch()
        self.scroll.setWidget(self.results_container)
        main_layout.addWidget(self.scroll, stretch=1)

        # 5. Status Footer
        footer = QHBoxLayout()
        self.status_lbl = QLabel("Ready. Drag and drop any reference drum hit or drum loop above.")
        self.status_lbl.setStyleSheet("color: #718096; font-size: 12px;")
        footer.addWidget(self.status_lbl)

        footer.addStretch()

        self.db_stat_lbl = QLabel("Library: 0 samples")
        self.db_stat_lbl.setStyleSheet("color: #718096; font-size: 12px;")
        footer.addWidget(self.db_stat_lbl)

        main_layout.addLayout(footer)

    def create_metric_widget(self, label, default_val):
        v = QVBoxLayout()
        v.setSpacing(1)
        l = QLabel(label)
        l.setObjectName("MetricLabel")
        val = QLabel(default_val)
        val.setObjectName("MetricVal")
        v.addWidget(l)
        v.addWidget(val)
        return v

    def set_metric_value(self, metric_layout, val_str):
        metric_layout.itemAt(1).widget().setText(val_str)

    def toggle_debleed(self):
        self.apply_debleed = self.debleed_btn.isChecked()
        if self.current_ref_path:
            self.load_reference(self.current_ref_path)

    def set_modifier(self, mod_name):
        self.current_modifier = mod_name
        for m, btn in self.mod_btns.items():
            btn.setChecked(m == mod_name)
        if self.current_slice_audio is not None:
            self.run_match_on_current_slice()

    def on_type_changed(self):
        if self.current_slice_audio is not None:
            self.run_match_on_current_slice()

    def on_category_changed(self):
        if self.current_slice_audio is not None:
            self.run_match_on_current_slice()


    def browse_reference_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Drum Hit or Loop", "",
            "Audio Files (*.wav *.flac *.aif *.aiff *.mp3);;All Files (*.*)"
        )
        if path:
            self.load_reference(path)

    def browse_and_index_kit(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Drum Kit Folder to Index")
        if folder:
            self.status_lbl.setText(f"Indexing '{os.path.basename(folder)}' in background...")
            self.add_kit_btn.setEnabled(False)
            self.indexer_thread = BackgroundIndexer(folder)
            self.indexer_thread.finished.connect(self.on_indexing_finished)
            self.indexer_thread.start()

    def on_indexing_finished(self, count):
        self.add_kit_btn.setEnabled(True)
        self.status_lbl.setText(f"Successfully indexed {count} samples!")
        self.load_default_stats()
        if self.current_slice_audio is not None:
            self.run_match_on_current_slice()

    def load_default_stats(self):
        if not os.path.exists(DB_NAME):
            self.db_stat_lbl.setText("Library: Not found. Run indexer.")
            return
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM samples")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT kit_name) FROM samples")
        kits = cur.fetchone()[0]
        conn.close()
        self.db_stat_lbl.setText(f"Library: {total} samples across {kits} drum kits")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(urls[0].toLocalFile().lower().endswith(('.wav', '.aif', '.aiff', '.flac', '.mp3')) for _ in [1]):
                event.acceptProposedAction()
                self.waveform.is_drag_active = True
                self.waveform.update()

    def dragLeaveEvent(self, event):
        self.waveform.is_drag_active = False
        self.waveform.update()

    def dropEvent(self, event):
        self.waveform.is_drag_active = False
        self.waveform.update()
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.wav', '.aif', '.aiff', '.flac', '.mp3')):
                self.load_reference(path)
                break

    def load_reference(self, path):
        try:
            self.status_lbl.setText(f"Analyzing {os.path.basename(path)}...")
            QApplication.processEvents()

            data, sr = load_audio(path)
            if self.apply_debleed:
                data = debleed_audio(data, sr)

            self.full_audio_data = data
            self.sr = sr
            self.current_ref_path = path

            # Detect all drum onsets in loop
            slices = detect_all_onsets(data, sr)
            self.slices = slices
            self.current_slice_idx = 0

            self.waveform.set_audio_and_slices(data, sr=sr, slices=slices)
            self.play_slice_btn.setEnabled(True)
            self.play_full_btn.setEnabled(True)
            self.match_slice_btn.setEnabled(True)
            self.prev_slice_btn.setEnabled(len(slices) > 1)
            self.next_slice_btn.setEnabled(len(slices) > 1)

            self.select_slice(0)
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Audio", f"Could not load audio file:\n{e}")
            self.status_lbl.setText("Error loading file.")

    def on_waveform_slice_clicked(self, idx):
        self.select_slice(idx)

    def prev_slice(self):
        if self.slices:
            idx = (self.current_slice_idx - 1) % len(self.slices)
            self.select_slice(idx)

    def next_slice(self):
        if self.slices:
            idx = (self.current_slice_idx + 1) % len(self.slices)
            self.select_slice(idx)

    def select_slice(self, idx):
        if not self.slices or idx < 0 or idx >= len(self.slices):
            return

        self.current_slice_idx = idx
        self.waveform.set_selected_slice(idx)

        start, end = self.slices[idx]
        self.current_slice_audio = self.full_audio_data[start:end]

        t_start_s = round(start / self.sr, 2)
        t_end_s = round(end / self.sr, 2)
        self.slice_info_lbl.setText(f"Hit #{idx + 1} of {len(self.slices)} ({t_start_s}s - {t_end_s}s)")

        self.run_match_on_current_slice()

    def run_match_on_current_slice(self):
        if self.current_slice_audio is None or len(self.current_slice_audio) == 0:
            return

        cat_choice = self.cat_combo.currentText().lower()
        category = None if cat_choice == "auto detect" else cat_choice

        type_choice = self.type_combo.currentText().lower()
        if "one-shot" in type_choice:
            sample_type = 'oneshot'
        elif "loop" in type_choice:
            sample_type = 'loop'
        else:
            sample_type = 'all'

        matches, ref_feats = match_sample(
            self.current_slice_audio,
            category=category,
            top_k=25,
            modifier=self.current_modifier,
            apply_debleed=False,
            sample_type=sample_type
        )


        # Update metric cards
        self.set_metric_value(self.m_punch, f"{ref_feats['crest_db']} dB")
        self.set_metric_value(self.m_rise, f"{ref_feats['attack_time_ms']} ms")
        if ref_feats['category'] in ('kick', '808') and ref_feats['pitch_drop_st'] > 0:
            self.set_metric_value(self.m_body, f"{ref_feats['f_sub']} Hz (-{ref_feats['pitch_drop_st']}st)")
        else:
            self.set_metric_value(self.m_body, f"{ref_feats['f_sub']} Hz")
        self.set_metric_value(self.m_decay, f"{ref_feats['decay_ms']} ms")
        self.set_metric_value(self.m_noise, f"{int(ref_feats['noise_ratio']*100)}%")

        # Populate Results List
        self.populate_results(matches)
        self.status_lbl.setText(f"Found {len(matches)} matching samples for Hit #{self.current_slice_idx + 1}.")

    def populate_results(self, matches):
        while self.results_layout.count() > 1:
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for rank, (score, cand, sims) in enumerate(matches, 1):
            item = MatchItemWidget(rank, score, cand, sims, self.current_ref_path, self)
            self.results_layout.insertWidget(rank - 1, item)

    def play_current_slice(self):
        if self.current_slice_audio is not None and len(self.current_slice_audio) > 0:
            try:
                import soundfile as sf
                # Write to temp WAV for zero-latency winsound playback
                if not self.temp_slice_wav:
                    temp_dir = tempfile.gettempdir()
                    self.temp_slice_wav = os.path.join(temp_dir, "drum_matcher_slice.wav")
                sf.write(self.temp_slice_wav, self.current_slice_audio, self.sr)
                winsound.PlaySound(self.temp_slice_wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception as e:
                print(f"Error playing slice: {e}")

    def play_full_audio(self):
        if self.current_ref_path and os.path.exists(self.current_ref_path):
            try:
                winsound.PlaySound(self.current_ref_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception as e:
                print(f"Error playing full audio: {e}")

    def keyPressEvent(self, event):
        key = event.key()
        if Qt.Key_1 <= key <= Qt.Key_9:
            idx = key - Qt.Key_1
            if idx < self.results_layout.count() - 1:
                item = self.results_layout.itemAt(idx).widget()
                if item and hasattr(item, 'play_sample'):
                    item.play_sample()
        elif key == Qt.Key_Space:
            self.play_current_slice()
        elif key == Qt.Key_R:
            self.play_full_audio()
        elif key == Qt.Key_Left:
            self.prev_slice()
        elif key == Qt.Key_Right:
            self.next_slice()
        else:
            super().keyPressEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
