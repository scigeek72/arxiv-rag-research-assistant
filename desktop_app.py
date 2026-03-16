"""
ArXiv Research Paper RAG  —  PyQt6 Desktop Application
=======================================================
Generic desktop app — topic is driven by Config.TOPIC_NAME.
Change TOPIC_NAME in config.py to switch research domains.

Features:
  • Chat interface with full conversation history
  • Year filter & query mode selector
  • Live source panel (papers cited per response)
  • Settings dialog  — flip OpenAI / LM Studio, set API key, tweak parameters
  • System-status tab (DB size, cache entries, model info)
  • Non-blocking QThread worker (UI never freezes during LLM calls)
  • Menu bar (File › Settings, Help)
  • Status bar with timing info

Run:
    python desktop_app.py
"""

import sys
import json
import os
import threading
from typing import Optional

# ── PyQt6 ──────────────────────────────────────────────────────────────────
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSplitter, QTextBrowser, QTextEdit, QLineEdit, QPushButton,
        QComboBox, QLabel, QStatusBar, QTabWidget, QGroupBox,
        QFormLayout, QCheckBox, QSpinBox, QDoubleSpinBox, QDialog,
        QDialogButtonBox, QScrollArea, QFrame, QSizePolicy, QMessageBox,
        QProgressBar,
    )
    from PyQt6.QtCore  import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui   import (
        QFont, QColor, QPalette, QTextCursor, QIcon, QAction,
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


# ── Colour palette (single place to change the theme) ─────────────────────
C = {
    "bg":          "#F0F2F5",   # main window background
    "surface":     "#FFFFFF",   # cards / panels
    "border":      "#CDD3DC",   # input borders
    "text":        "#1A2332",   # primary text  (dark navy)
    "text_muted":  "#5A6A7A",   # secondary text
    "accent":      "#2563EB",   # blue accent  (buttons, links)
    "accent_dark": "#1D4ED8",   # hover state
    "accent_light":"#DBEAFE",   # light blue tint
    "success":     "#16A34A",   # green (ready states)
    "warning":     "#D97706",   # amber (warnings)
    "error":       "#DC2626",   # red (errors)
    "user_bg":     "#DBEAFE",   # user message bubble
    "user_text":   "#1E3A5F",
    "bot_bg":      "#DCFCE7",   # assistant message bubble
    "bot_text":    "#14532D",
    "sys_bg":      "#FEF9C3",   # system/info message bubble
    "sys_text":    "#713F12",
    "tab_active":  "#FFFFFF",
    "tab_inactive":"#E5E9F0",
    "status_bg":   "#1A2332",   # status bar
    "status_text": "#E2E8F0",
}

APP_STYLESHEET = f"""
/* ── Global ──────────────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background: {C['bg']};
    color: {C['text']};
    font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 13px;
}}

/* ── Labels ──────────────────────────────────────────────────────── */
QLabel {{
    color: {C['text']};
    background: transparent;
}}

/* ── Inputs ──────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    color: {C['text']};
    background: {C['surface']};
    border: 1.5px solid {C['border']};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {C['accent']};
    selection-color: white;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {C['accent']};
}}
QLineEdit::placeholder {{
    color: {C['text_muted']};
}}

/* ── ComboBox ────────────────────────────────────────────────────── */
QComboBox {{
    color: {C['text']};
    background: {C['surface']};
    border: 1.5px solid {C['border']};
    border-radius: 6px;
    padding: 5px 8px;
    min-width: 80px;
}}
QComboBox:focus {{ border-color: {C['accent']}; }}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    width: 10px;
    height: 10px;
}}
QComboBox QAbstractItemView {{
    color: {C['text']};
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 4px;
    selection-background-color: {C['accent']};
    selection-color: white;
    outline: none;
}}

/* ── Buttons ─────────────────────────────────────────────────────── */
QPushButton {{
    color: {C['text']};
    background: {C['surface']};
    border: 1.5px solid {C['border']};
    border-radius: 6px;
    padding: 5px 12px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {C['accent_light']};
    border-color: {C['accent']};
    color: {C['accent_dark']};
}}
QPushButton:pressed {{
    background: {C['accent']};
    color: white;
    border-color: {C['accent_dark']};
}}
QPushButton:disabled {{
    color: {C['text_muted']};
    background: {C['bg']};
    border-color: {C['border']};
}}

/* ── Text browsers ───────────────────────────────────────────────── */
QTextBrowser, QTextEdit {{
    color: {C['text']};
    background: {C['surface']};
    border: 1.5px solid {C['border']};
    border-radius: 6px;
    selection-background-color: {C['accent']};
    selection-color: white;
}}

/* ── Group boxes ─────────────────────────────────────────────────── */
QGroupBox {{
    color: {C['text']};
    font-weight: bold;
    font-size: 12px;
    border: 1.5px solid {C['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {C['text']};
    background: {C['bg']};
}}

/* ── Tabs ────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1.5px solid {C['border']};
    border-radius: 8px;
    background: {C['surface']};
}}
QTabBar::tab {{
    color: {C['text_muted']};
    background: {C['tab_inactive']};
    border: 1px solid {C['border']};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 7px 20px;
    font-size: 13px;
    font-weight: 500;
    min-width: 100px;
}}
QTabBar::tab:selected {{
    color: {C['accent']};
    background: {C['tab_active']};
    font-weight: bold;
    border-bottom: none;
}}
QTabBar::tab:hover:!selected {{
    color: {C['text']};
    background: {C['accent_light']};
}}

/* ── Check boxes ─────────────────────────────────────────────────── */
QCheckBox {{
    color: {C['text']};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1.5px solid {C['border']};
    border-radius: 4px;
    background: {C['surface']};
}}
QCheckBox::indicator:checked {{
    background: {C['accent']};
    border-color: {C['accent']};
}}

/* ── Scroll bars ─────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {C['bg']};
    width: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background: {C['border']};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {C['text_muted']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── Status bar ──────────────────────────────────────────────────── */
QStatusBar {{
    background: {C['status_bg']};
    color: {C['status_text']};
    font-size: 12px;
    padding: 3px 8px;
}}
QStatusBar QLabel {{
    color: {C['status_text']};
    background: transparent;
}}

/* ── Progress bar ────────────────────────────────────────────────── */
QProgressBar {{
    background: {C['bg']};
    border: none;
    border-radius: 2px;
    height: 4px;
}}
QProgressBar::chunk {{
    background: {C['accent']};
    border-radius: 2px;
}}

/* ── Splitter handle ─────────────────────────────────────────────── */
QSplitter::handle {{
    background: {C['border']};
    width: 2px;
    height: 2px;
}}

/* ── Menu bar ────────────────────────────────────────────────────── */
QMenuBar {{
    background: {C['bg']};
    color: {C['text']};
    border-bottom: 1px solid {C['border']};
    padding: 2px;
}}
QMenuBar::item:selected {{
    background: {C['accent_light']};
    color: {C['accent_dark']};
    border-radius: 4px;
}}
QMenu {{
    background: {C['surface']};
    color: {C['text']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item:selected {{
    background: {C['accent_light']};
    color: {C['accent_dark']};
    border-radius: 4px;
}}

/* ── Dialog ──────────────────────────────────────────────────────── */
QDialog {{
    background: {C['bg']};
    color: {C['text']};
}}
QDialogButtonBox QPushButton {{
    min-width: 80px;
    padding: 6px 14px;
}}
"""


# ─── Settings Dialog ─────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """Modal settings dialog that writes back to Config."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(500)
        self._build_ui()

    def _build_ui(self):
        from config import Config
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── LLM Provider ──────────────────────────────────────────────────
        llm_box = QGroupBox("LLM Provider")
        llm_form = QFormLayout(llm_box)
        llm_form.setSpacing(8)

        self.chk_use_openai = QCheckBox("Use OpenAI API  (instead of LM Studio)")
        self.chk_use_openai.setChecked(Config.USE_OPENAI)
        llm_form.addRow(self.chk_use_openai)

        self.txt_openai_key = QLineEdit(Config.OPENAI_API_KEY or "")
        self.txt_openai_key.setPlaceholderText("sk-…")
        self.txt_openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        llm_form.addRow("OpenAI API Key:", self.txt_openai_key)

        self.cmb_openai_model = QComboBox()
        self.cmb_openai_model.addItems(
            ["gpt-4o-mini", "gpt-4o", "o3-mini", "gpt-3.5-turbo"]
        )
        self.cmb_openai_model.setCurrentText(Config.OPENAI_CHAT_MODEL)
        llm_form.addRow("OpenAI Chat Model:", self.cmb_openai_model)

        self.txt_lmstudio_url = QLineEdit(Config.LM_STUDIO_API_BASE)
        llm_form.addRow("LM Studio URL:", self.txt_lmstudio_url)
        layout.addWidget(llm_box)

        # ── Embeddings ────────────────────────────────────────────────────
        emb_box = QGroupBox("Embeddings")
        emb_form = QFormLayout(emb_box)
        emb_form.setSpacing(8)

        self.chk_use_openai_emb = QCheckBox(
            "Use OpenAI embeddings  (text-embedding-3-small)"
        )
        self.chk_use_openai_emb.setChecked(Config.USE_OPENAI_EMBEDDINGS)
        emb_form.addRow(self.chk_use_openai_emb)

        self.cmb_local_emb = QComboBox()
        self.cmb_local_emb.addItems([
            "BAAI/bge-large-en-v1.5",
            "all-mpnet-base-v2",
            "all-MiniLM-L6-v2",
        ])
        self.cmb_local_emb.setCurrentText(Config.EMBEDDING_MODEL_NAME)
        emb_form.addRow("Local Embedding Model:", self.cmb_local_emb)

        warn = QLabel("Changing the embedding model requires a full index rebuild.")
        warn.setWordWrap(True)
        warn.setStyleSheet(f"color: {C['warning']}; font-size: 11px;")
        emb_form.addRow(warn)
        layout.addWidget(emb_box)

        # ── RAG Parameters ────────────────────────────────────────────────
        rag_box = QGroupBox("RAG Parameters")
        rag_form = QFormLayout(rag_box)
        rag_form.setSpacing(8)

        self.spn_retrieval_k = QSpinBox()
        self.spn_retrieval_k.setRange(3, 30)
        self.spn_retrieval_k.setValue(Config.RETRIEVAL_K)
        rag_form.addRow("Retrieval K (candidates):", self.spn_retrieval_k)

        self.spn_rerank_k = QSpinBox()
        self.spn_rerank_k.setRange(1, 15)
        self.spn_rerank_k.setValue(Config.RERANK_TOP_K)
        rag_form.addRow("Rerank Top-K:", self.spn_rerank_k)

        self.spn_temp = QDoubleSpinBox()
        self.spn_temp.setRange(0.0, 2.0)
        self.spn_temp.setSingleStep(0.05)
        self.spn_temp.setValue(Config.LLM_TEMPERATURE)
        rag_form.addRow("LLM Temperature:", self.spn_temp)

        self.spn_max_tokens = QSpinBox()
        self.spn_max_tokens.setRange(128, 4096)
        self.spn_max_tokens.setSingleStep(128)
        self.spn_max_tokens.setValue(Config.LLM_MAX_TOKENS)
        rag_form.addRow("Max Tokens:", self.spn_max_tokens)
        layout.addWidget(rag_box)

        # ── Feature Toggles ───────────────────────────────────────────────
        feat_box = QGroupBox("Feature Toggles")
        feat_form = QFormLayout(feat_box)
        feat_form.setSpacing(8)

        self.chk_hybrid   = QCheckBox("Hybrid Search  (BM25 + Semantic)")
        self.chk_hybrid.setChecked(Config.USE_HYBRID_SEARCH)
        feat_form.addRow(self.chk_hybrid)

        self.chk_reranker = QCheckBox("Cross-Encoder Re-ranking")
        self.chk_reranker.setChecked(Config.USE_RERANKER)
        feat_form.addRow(self.chk_reranker)

        self.chk_hyde     = QCheckBox("HyDE  (Hypothetical Document Embeddings)")
        self.chk_hyde.setChecked(Config.USE_HYDE)
        feat_form.addRow(self.chk_hyde)

        self.chk_multihop = QCheckBox("Multi-hop Chain-of-Thought")
        self.chk_multihop.setChecked(Config.USE_MULTIHOP)
        feat_form.addRow(self.chk_multihop)

        self.chk_cache    = QCheckBox("Persistent SQLite Cache")
        self.chk_cache.setChecked(Config.USE_PERSISTENT_CACHE)
        feat_form.addRow(self.chk_cache)
        layout.addWidget(feat_box)

        # ── Buttons ───────────────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._apply)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _apply(self):
        from config import Config
        Config.USE_OPENAI            = self.chk_use_openai.isChecked()
        Config.OPENAI_API_KEY        = self.txt_openai_key.text().strip()
        Config.OPENAI_CHAT_MODEL     = self.cmb_openai_model.currentText()
        Config.LM_STUDIO_API_BASE    = self.txt_lmstudio_url.text().strip()
        Config.USE_OPENAI_EMBEDDINGS = self.chk_use_openai_emb.isChecked()
        Config.EMBEDDING_MODEL_NAME  = self.cmb_local_emb.currentText()
        Config.RETRIEVAL_K           = self.spn_retrieval_k.value()
        Config.RERANK_TOP_K          = self.spn_rerank_k.value()
        Config.LLM_TEMPERATURE       = self.spn_temp.value()
        Config.LLM_MAX_TOKENS        = self.spn_max_tokens.value()
        Config.USE_HYBRID_SEARCH     = self.chk_hybrid.isChecked()
        Config.USE_RERANKER          = self.chk_reranker.isChecked()
        Config.USE_HYDE              = self.chk_hyde.isChecked()
        Config.USE_MULTIHOP          = self.chk_multihop.isChecked()
        Config.USE_PERSISTENT_CACHE  = self.chk_cache.isChecked()
        self.accept()


# ─── Worker Threads ───────────────────────────────────────────────────────────

class QueryWorker(QThread):
    """Runs the RAG pipeline in a background thread to keep UI responsive."""
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, query: str, year_filter: Optional[int], mode: str):
        super().__init__()
        self.query       = query
        self.year_filter = year_filter
        self.mode        = mode

    def run(self):
        try:
            from query_rag_v3 import rag_query
            result = rag_query(self.query, self.year_filter, self.mode)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class InitWorker(QThread):
    """Initialises the RAG system in background on app startup."""
    success = pyqtSignal(str)
    failure = pyqtSignal(str)

    def run(self):
        try:
            from query_rag_v3 import initialize_rag
            ok = initialize_rag()
            if ok:
                from config import Config
                info = Config.get_active_llm_info()
                self.success.emit(
                    f"RAG ready  |  {info['provider']}  |  {info['model']}"
                )
            else:
                self.failure.emit("RAG initialization failed — check logs.")
        except Exception as e:
            self.failure.emit(str(e))


# ─── Status Tab ───────────────────────────────────────────────────────────────

class StatusTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.status_browser = QTextBrowser()
        self.status_browser.setFont(QFont("Menlo, Courier New, monospace", 12))
        layout.addWidget(self.status_browser)

        refresh_btn = QPushButton("Refresh Status")
        refresh_btn.setFixedWidth(140)
        refresh_btn.setStyleSheet(
            f"QPushButton {{ background: {C['accent']}; color: white; "
            f"border: none; border-radius: 6px; padding: 6px 14px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: {C['accent_dark']}; }}"
        )
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(refresh_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        self.refresh()

    def refresh(self):
        from config import Config
        s   = Config.get_system_status()
        llm = Config.get_active_llm_info()
        emb = Config.get_active_embedding_info()

        def badge(on: bool) -> str:
            if on:
                return (f'<span style="background:{C["success"]};color:white;'
                        f'border-radius:3px;padding:1px 6px;font-size:11px;">ON</span>')
            return (f'<span style="background:{C["border"]};color:{C["text_muted"]};'
                    f'border-radius:3px;padding:1px 6px;font-size:11px;">OFF</span>')

        def row(label: str, value: str, warn: bool = False) -> str:
            vc = C["warning"] if warn else C["text"]
            return (f'<tr>'
                    f'<td style="color:{C["text_muted"]};padding:4px 12px 4px 0;'
                    f'font-weight:600;white-space:nowrap;">{label}</td>'
                    f'<td style="color:{vc};padding:4px 0;">{value}</td>'
                    f'</tr>')

        db_ok = s.get("vector_db_exists", False)

        html = f"""
        <style>
            body {{ font-family: -apple-system,"Segoe UI",sans-serif;
                   font-size:13px; color:{C['text']}; margin:0; padding:8px; }}
            h4   {{ color:{C['accent']}; margin:16px 0 6px 0;
                   font-size:13px; letter-spacing:0.5px; text-transform:uppercase; }}
            table {{ border-collapse:collapse; width:100%; }}
        </style>

        <h4>System</h4>
        <table>
          {row("Topic",       s.get("topic","—"))}
          {row("PDFs",        str(s.get("pdfs_count", 0)))}
          {row("Text files",  str(s.get("text_files_count", 0)))}
          {row("Vector DB",
               '<span style="color:'+C['success']+'">Ready</span>' if db_ok
               else '<span style="color:'+C['error']+'">Missing — run build script</span>',
               warn=not db_ok)}
          {row("DB size",     f"{s.get('vector_db_size_mb', 0)} MB")}
          {row("Cache",       f"{s.get('cache_entries', 0)} entries")}
        </table>

        <h4>Active LLM</h4>
        <table>
          {row("Provider",  llm["provider"])}
          {row("Model",     llm["model"])}
          {row("API base",  llm["api_base"])}
        </table>

        <h4>Embeddings</h4>
        <table>
          {row("Provider",  emb["provider"])}
          {row("Model",     emb["model"])}
        </table>

        <h4>Features</h4>
        <table>
          <tr><td style="color:{C['text_muted']};padding:4px 12px 4px 0;
              font-weight:600;">Hybrid Search</td>
              <td style="padding:4px 0;">{badge(Config.USE_HYBRID_SEARCH)}</td></tr>
          <tr><td style="color:{C['text_muted']};padding:4px 12px 4px 0;
              font-weight:600;">HyDE</td>
              <td style="padding:4px 0;">{badge(Config.USE_HYDE)}</td></tr>
          <tr><td style="color:{C['text_muted']};padding:4px 12px 4px 0;
              font-weight:600;">Re-ranking</td>
              <td style="padding:4px 0;">{badge(Config.USE_RERANKER)}</td></tr>
          <tr><td style="color:{C['text_muted']};padding:4px 12px 4px 0;
              font-weight:600;">Multi-hop</td>
              <td style="padding:4px 0;">{badge(Config.USE_MULTIHOP)}</td></tr>
          <tr><td style="color:{C['text_muted']};padding:4px 12px 4px 0;
              font-weight:600;">Cache</td>
              <td style="padding:4px 0;">{badge(Config.USE_PERSISTENT_CACHE)}</td></tr>
        </table>
        """
        self.status_browser.setHtml(html)


# ─── Sources Panel ────────────────────────────────────────────────────────────

class SourcesPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header
        header = QLabel("Sources")
        header.setFont(QFont("-apple-system", 13, QFont.Weight.Bold))
        header.setStyleSheet(
            f"color: {C['text']}; padding: 6px 0 4px 2px; "
            f"border-bottom: 2px solid {C['accent']}; margin-bottom: 4px;"
        )
        layout.addWidget(header)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setFont(QFont("-apple-system", 12))
        self.browser.setStyleSheet(
            f"QTextBrowser {{ border: none; background: {C['bg']}; padding: 4px; }}"
        )
        layout.addWidget(self.browser)

    def update_sources(self, sources: list):
        if not sources:
            self.browser.setHtml(
                f'<p style="color:{C["text_muted"]};font-size:12px;'
                f'padding:8px;">No sources for this response.</p>'
            )
            return

        html = f"""<style>
            body {{ font-family:-apple-system,"Segoe UI",sans-serif;
                   font-size:12px; color:{C['text']}; margin:0; padding:4px; }}
            .card {{ background:{C['surface']}; border:1px solid {C['border']};
                    border-radius:8px; padding:10px 12px; margin-bottom:10px; }}
            .pid  {{ font-weight:700; color:{C['text']}; font-size:12px; }}
            .ttl  {{ font-style:italic; color:{C['text']}; font-size:12px;
                    margin-top:2px; }}
            .aut  {{ color:{C['text_muted']}; font-size:11px; margin-top:2px; }}
            .snp  {{ color:{C['text_muted']}; font-size:11px; margin-top:6px;
                    border-top:1px solid {C['border']}; padding-top:6px; }}
            a     {{ color:{C['accent']}; text-decoration:none; font-size:11px; }}
            a:hover {{ text-decoration:underline; }}
        </style>"""

        seen = set()
        for src in sources:
            pid = src.get("paper_id", "?")
            if pid in seen:
                continue
            seen.add(pid)

            yr  = src.get("paper_year", "")
            ttl = src.get("title", "")
            aut = src.get("authors", "")
            url = src.get("paper_url", "")
            snp = src.get("snippet", "")

            html += '<div class="card">'
            html += f'<div class="pid">{pid}'
            if yr:
                html += f' <span style="font-weight:normal;color:{C["text_muted"]}">({yr})</span>'
            html += '</div>'
            if ttl:
                html += f'<div class="ttl">{ttl[:120]}</div>'
            if aut:
                html += f'<div class="aut">{aut[:100]}</div>'
            if url:
                html += f'<div style="margin-top:6px;"><a href="{url}">Open on arXiv &rarr;</a></div>'
            if snp:
                html += f'<div class="snp">{snp[:220]}&hellip;</div>'
            html += '</div>'

        self.browser.setHtml(html)

    def clear(self):
        self.browser.setHtml(
            f'<p style="color:{C["text_muted"]};font-size:12px;padding:8px;">'
            f'Sources will appear here after a query.</p>'
        )


# ─── Chat Tab ─────────────────────────────────────────────────────────────────

class ChatTab(QWidget):
    """Main chat interface."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window  = main_window
        self._worker: Optional[QueryWorker] = None
        self._history: list = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Toolbar ───────────────────────────────────────────────────────
        toolbar = QFrame()
        toolbar.setStyleSheet(
            f"QFrame {{ background: {C['surface']}; border: 1px solid {C['border']}; "
            f"border-radius: 8px; }}"
        )
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(10, 6, 10, 6)
        tb_layout.setSpacing(10)

        # Year filter
        year_lbl = QLabel("Year:")
        year_lbl.setStyleSheet(f"color:{C['text_muted']};font-weight:600;font-size:12px;")
        tb_layout.addWidget(year_lbl)

        self.year_combo = QComboBox()
        self.year_combo.addItems(
            ["All Years", "2025", "2024", "2023", "2022", "2021", "2020"]
        )
        self.year_combo.setFixedWidth(105)
        self.year_combo.setToolTip("Filter papers by publication year")
        tb_layout.addWidget(self.year_combo)

        # Separator
        sep = QLabel("|")
        sep.setStyleSheet(f"color:{C['border']};font-size:16px;")
        tb_layout.addWidget(sep)

        # Mode selector
        mode_lbl = QLabel("Mode:")
        mode_lbl.setStyleSheet(f"color:{C['text_muted']};font-weight:600;font-size:12px;")
        tb_layout.addWidget(mode_lbl)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["auto", "standard", "hyde", "multihop"])
        self.mode_combo.setFixedWidth(105)
        self.mode_combo.setToolTip(
            "auto     — system picks the best mode\n"
            "standard — direct semantic search (fastest)\n"
            "hyde     — hypothetical doc embedding (better recall)\n"
            "multihop — decompose into sub-queries (best for comparisons)"
        )
        tb_layout.addWidget(self.mode_combo)

        tb_layout.addStretch()

        # Action buttons
        self.clear_btn = QPushButton("Clear Chat")
        self.clear_btn.setFixedWidth(100)
        self.clear_btn.setToolTip("Clear the chat history")
        self.clear_btn.clicked.connect(self._clear_chat)
        tb_layout.addWidget(self.clear_btn)

        self.cache_btn = QPushButton("Clear Cache")
        self.cache_btn.setFixedWidth(100)
        self.cache_btn.setToolTip("Clear the persistent query cache")
        self.cache_btn.clicked.connect(self._clear_cache)
        tb_layout.addWidget(self.cache_btn)

        root.addWidget(toolbar)

        # ── Main splitter: chat | sources ─────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)

        # Left: chat display
        chat_frame = QFrame()
        chat_layout = QVBoxLayout(chat_frame)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(6)

        self.chat_display = QTextBrowser()
        self.chat_display.setFont(QFont("-apple-system", 13))
        self.chat_display.setOpenExternalLinks(True)
        self.chat_display.setStyleSheet(
            f"QTextBrowser {{ background: {C['bg']}; border: none; padding: 6px; }}"
        )
        chat_layout.addWidget(self.chat_display)

        # Input row
        input_row = QHBoxLayout()
        input_row.setSpacing(8)

        self.query_input = QLineEdit()
        self.query_input.setFont(QFont("-apple-system", 13))
        self.query_input.setFixedHeight(38)
        from config import Config
        self.query_input.setPlaceholderText(
            f"Ask anything about {Config.TOPIC_NAME} research…"
        )
        self.query_input.returnPressed.connect(self._send_query)
        input_row.addWidget(self.query_input, stretch=1)

        self.send_btn = QPushButton("Ask")
        self.send_btn.setFont(QFont("-apple-system", 13, QFont.Weight.Bold))
        self.send_btn.setFixedSize(70, 38)
        self.send_btn.clicked.connect(self._send_query)
        self.send_btn.setStyleSheet(
            f"QPushButton {{ background:{C['accent']}; color:white; border:none; "
            f"border-radius:6px; font-weight:bold; }}"
            f"QPushButton:hover {{ background:{C['accent_dark']}; }}"
            f"QPushButton:pressed {{ background:#1E40AF; }}"
            f"QPushButton:disabled {{ background:{C['border']}; color:{C['text_muted']}; }}"
        )
        input_row.addWidget(self.send_btn)
        chat_layout.addLayout(input_row)

        # Progress bar (indeterminate, hidden by default)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        chat_layout.addWidget(self.progress)

        splitter.addWidget(chat_frame)

        # Right: sources panel in a frame
        sources_frame = QFrame()
        sources_frame.setStyleSheet(
            f"QFrame {{ background:{C['bg']}; border-left:1px solid {C['border']}; "
            f"padding: 8px; }}"
        )
        sf_layout = QVBoxLayout(sources_frame)
        sf_layout.setContentsMargins(8, 4, 4, 4)
        self.sources_panel = SourcesPanel()
        sf_layout.addWidget(self.sources_panel)
        splitter.addWidget(sources_frame)
        splitter.setSizes([720, 320])

        root.addWidget(splitter, stretch=1)

        # ── Example questions bar ─────────────────────────────────────────
        ex_frame = QFrame()
        ex_frame.setStyleSheet(
            f"QFrame {{ background:{C['surface']}; border:1px solid {C['border']}; "
            f"border-radius:8px; }}"
        )
        ex_layout = QHBoxLayout(ex_frame)
        ex_layout.setContentsMargins(10, 6, 10, 6)
        ex_layout.setSpacing(8)

        hint = QLabel("Examples:")
        hint.setStyleSheet(
            f"color:{C['text_muted']};font-size:11px;font-weight:600;"
        )
        ex_layout.addWidget(hint)

        from config import Config
        for ex in Config.EXAMPLE_QUESTIONS[:3]:
            btn = QPushButton(ex)
            btn.setStyleSheet(
                f"QPushButton {{ color:{C['accent']}; background:{C['accent_light']}; "
                f"border:1px solid {C['accent']}; border-radius:12px; "
                f"padding:3px 10px; font-size:11px; }}"
                f"QPushButton:hover {{ background:{C['accent']}; color:white; }}"
            )
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, t=ex: self._use_example(t))
            ex_layout.addWidget(btn)
        ex_layout.addStretch()
        root.addWidget(ex_frame)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _use_example(self, text: str):
        self.query_input.setText(text)
        self.query_input.setFocus()

    def _clear_chat(self):
        self.chat_display.clear()
        self._history.clear()
        self.sources_panel.clear()
        self.main_window.set_status("Chat cleared.")

    def _clear_cache(self):
        try:
            from query_rag_v3 import _cache
            if _cache:
                _cache.clear()
                self.main_window.set_status("Cache cleared.")
            else:
                self.main_window.set_status("Cache not enabled.")
        except Exception as e:
            self.main_window.set_status(f"Cache clear error: {e}")

    @staticmethod
    def _format_text(text: str) -> str:
        """
        Convert LLM plain-text output to readable HTML:
          • **word**  → <b>word</b>
          • blank line → paragraph break  (most important for readability)
          • single \n  → <br>
          • numbered list items get a little top margin
        """
        import re
        # Escape any existing HTML special chars first (< > &) that aren't
        # already HTML tags coming from the system prompt.
        # We only escape bare < and > that are NOT part of a tag.
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;").replace(">", "&gt;")

        # **bold** → <b>bold</b>
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)

        # Paragraph breaks (blank line = \n\n or \r\n\r\n)
        paragraphs = re.split(r'\n{2,}', text)
        parts = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # Single newlines within a paragraph → <br>
            para = para.replace('\n', '<br>')
            # Indent numbered list items slightly
            para = re.sub(
                r'^(\d+\.\s)',
                r'<span style="display:inline-block;width:20px;"></span>\1',
                para,
                flags=re.MULTILINE,
            )
            parts.append(para)

        # Join paragraphs with visible vertical space
        return '</p><p style="margin:0 0 10px 0;">'.join(parts)

    def _append_message(self, role: str, text: str):
        """Append a styled message bubble to the chat display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)

        if role == "user":
            html = (
                f'<div style="margin:6px 0 6px 40px; padding:10px 14px; '
                f'background:{C["user_bg"]}; border-radius:12px 12px 4px 12px; '
                f'border:1px solid #BFDBFE;">'
                f'<span style="color:{C["text_muted"]};font-size:11px;'
                f'font-weight:600;">You</span>'
                f'<div style="color:{C["user_text"]};font-size:13px;'
                f'line-height:1.6;margin-top:4px;">{text}</div></div>'
            )
        elif role == "assistant":
            formatted = self._format_text(text)
            html = (
                f'<div style="margin:8px 40px 8px 0; padding:12px 16px; '
                f'background:{C["bot_bg"]}; border-radius:12px 12px 12px 4px; '
                f'border:1px solid #BBF7D0;">'
                f'<span style="color:{C["text_muted"]};font-size:11px;'
                f'font-weight:600;letter-spacing:0.3px;">Assistant</span>'
                f'<div style="color:{C["bot_text"]};font-size:13px;'
                f'line-height:1.75;margin-top:6px;">'
                f'<p style="margin:0 0 10px 0;">{formatted}</p>'
                f'</div></div>'
            )
        else:   # system / info
            html = (
                f'<div style="margin:4px 20px; padding:6px 12px; '
                f'background:{C["sys_bg"]}; border-radius:6px; '
                f'border:1px solid #FDE68A; text-align:center;">'
                f'<span style="color:{C["sys_text"]};font-size:11px;">{text}</span>'
                f'</div>'
            )

        self.chat_display.insertHtml(html + "<br>")
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _get_year_filter(self) -> Optional[int]:
        yr = self.year_combo.currentText()
        return int(yr) if yr != "All Years" else None

    # ── Query dispatch ────────────────────────────────────────────────────────

    def _send_query(self):
        query = self.query_input.text().strip()
        if not query:
            return

        self.query_input.clear()
        self.send_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.sources_panel.clear()
        self._append_message("user", query)

        year = self._get_year_filter()
        mode = self.mode_combo.currentText()
        if year:
            self._append_message("system", f"Filtering for year {year}")

        self.main_window.set_status("Thinking…")

        self._worker = QueryWorker(query, year, mode)
        self._worker.finished.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_result(self, result: dict):
        self.send_btn.setEnabled(True)
        self.progress.setVisible(False)

        answer = result.get("answer", "No answer returned.")
        mode   = result.get("mode_used", "?")
        t      = result.get("response_time", 0.0)
        cache  = result.get("cache_hit", False)
        sub_qs = result.get("sub_questions", [])

        info_parts = [
            f"mode = {mode}",
            f"time = {t}s",
            "cache = HIT (instant)" if cache else "cache = MISS",
        ]
        if len(sub_qs) > 1:
            info_parts.append(f"hops = {len(sub_qs)}")
        footer = "  |  ".join(info_parts)

        self._append_message("assistant", answer)
        self._append_message("system", footer)
        self.sources_panel.update_sources(result.get("sources", []))
        self.main_window.set_status(footer)

    def _on_error(self, error_str: str):
        self.send_btn.setEnabled(True)
        self.progress.setVisible(False)
        self._append_message("system", f"Error: {error_str}")
        self.main_window.set_status(f"Error: {error_str[:80]}")


# ─── Main Window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from config import Config
        self.setWindowTitle(f"{Config.TOPIC_NAME} Research Assistant  (RAG v3)")
        self.setMinimumSize(1100, 740)
        self._rag_ready = False
        self._build_menu()
        self._build_ui()
        self._start_init()

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        settings_act = QAction("Settings…", self)
        settings_act.setShortcut("Ctrl+,")
        settings_act.triggered.connect(self._open_settings)
        file_menu.addAction(settings_act)
        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        help_menu = menu.addMenu("Help")
        about_act = QAction("About", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # ── Header band ───────────────────────────────────────────────────
        header_frame = QFrame()
        header_frame.setStyleSheet(
            f"QFrame {{ background: {C['surface']}; border: 1px solid {C['border']}; "
            f"border-radius: 10px; }}"
        )
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(14, 10, 14, 10)

        from config import Config
        title = QLabel(f"{Config.TOPIC_NAME} Research Assistant")
        title.setFont(QFont("-apple-system", 17, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C['text']}; background: transparent;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.init_label = QLabel("Initialising RAG system…")
        self.init_label.setStyleSheet(
            f"color: {C['warning']}; font-size: 12px; background: transparent;"
        )
        header_layout.addWidget(self.init_label)
        layout.addWidget(header_frame)

        # ── Tabs ──────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.chat_tab   = ChatTab(self)
        self.status_tab = StatusTab()
        self.tabs.addTab(self.chat_tab,   "Chat")
        self.tabs.addTab(self.status_tab, "Status")
        layout.addWidget(self.tabs, stretch=1)

        # ── Status bar ────────────────────────────────────────────────────
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initialising…")

    # ── Initialisation ────────────────────────────────────────────────────────

    def _start_init(self):
        self.chat_tab.send_btn.setEnabled(False)
        self._init_worker = InitWorker()
        self._init_worker.success.connect(self._on_init_success)
        self._init_worker.failure.connect(self._on_init_failure)
        self._init_worker.start()

    def _on_init_success(self, message: str):
        self._rag_ready = True
        self.init_label.setText(f"  {message}")
        self.init_label.setStyleSheet(
            f"color: {C['success']}; font-size: 12px; font-weight: 600; "
            f"background: transparent;"
        )
        self.chat_tab.send_btn.setEnabled(True)
        self.status_bar.showMessage(message)
        self.status_tab.refresh()
        from config import Config
        self.chat_tab._append_message(
            "system",
            f"RAG system ready! &nbsp; {message}<br>"
            f"Ask me anything about <b>{Config.TOPIC_NAME}</b> research"
        )

    def _on_init_failure(self, error: str):
        self.init_label.setText(f"Init failed: {error[:60]}")
        self.init_label.setStyleSheet(
            f"color: {C['error']}; font-size: 12px; background: transparent;"
        )
        self.status_bar.showMessage(f"Init failed: {error}")
        from config import Config
        self.chat_tab._append_message(
            "system",
            f"Initialisation failed: {error}<br><br>"
            "Possible fixes:<br>"
            f"&bull; Build the index first: <tt>python build_rag_index_v4.py</tt><br>"
            "&bull; If using OpenAI: check API key in Settings (File > Settings)<br>"
            "&bull; If using LM Studio: ensure server is running on localhost:1234<br>"
            f"&bull; Topic: <tt>{Config.TOPIC_NAME}</tt>"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def set_status(self, msg: str):
        self.status_bar.showMessage(msg)

    def _open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            QMessageBox.information(
                self, "Settings Saved",
                "Settings applied.\n\n"
                "Note: Changes to LLM provider or API key take effect\n"
                "after restarting the app.\n\n"
                "If you changed the embedding model, rebuild the index:\n"
                "  python build_rag_index_v4.py --rebuild"
            )
            self.status_tab.refresh()

    def _show_about(self):
        from config import Config
        QMessageBox.about(
            self, "About",
            f"<b>{Config.TOPIC_NAME} Research Assistant — RAG v3</b><br><br>"
            "A desktop application for querying arXiv research papers<br>"
            f"on <i>{Config.TOPIC_NAME}</i> using Retrieval-Augmented Generation.<br><br>"
            f"<b>Topic:</b> {Config.TOPIC_NAME}<br>"
            f"<b>Description:</b> {Config.TOPIC_DESCRIPTION}<br><br>"
            "<b>Features:</b><br>"
            "  Hybrid BM25 + Semantic search<br>"
            "  HyDE hypothetical document embeddings<br>"
            "  Cross-encoder re-ranking<br>"
            "  Multi-hop chain-of-thought reasoning<br>"
            "  Persistent SQLite query cache<br>"
            "  OpenAI API and LM Studio support<br><br>"
            "Built with PyQt6 · LangChain · ChromaDB"
        )


# ─── Application Entry-point ─────────────────────────────────────────────────

def main():
    if not PYQT_AVAILABLE:
        print("PyQt6 is not installed.  Run: pip install PyQt6")
        print("Falling back to Tkinter…")
        _run_tkinter_fallback()
        return

    app = QApplication(sys.argv)
    from config import Config
    app.setApplicationName(f"{Config.TOPIC_NAME} RAG")
    app.setApplicationVersion("3.0")

    # Force light-mode palette so text always reads dark on light backgrounds
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(C["bg"]))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(C["text"]))
    palette.setColor(QPalette.ColorRole.Base,            QColor(C["surface"]))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(C["bg"]))
    palette.setColor(QPalette.ColorRole.Text,            QColor(C["text"]))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(C["text"]))
    palette.setColor(QPalette.ColorRole.Button,          QColor(C["surface"]))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(C["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(C["text_muted"]))
    app.setPalette(palette)

    app.setStyleSheet(APP_STYLESHEET)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


# ─── Tkinter Fallback ────────────────────────────────────────────────────────

def _run_tkinter_fallback():
    """Minimal Tkinter app for systems without PyQt6."""
    import tkinter as tk
    from tkinter import ttk, scrolledtext

    class TkRAGApp:
        def __init__(self, root):
            self.root = root
            from config import Config
            root.title(f"{Config.TOPIC_NAME} RAG (Tkinter)")
            root.geometry("900x650")
            self._build_ui()
            threading.Thread(target=self._init_rag, daemon=True).start()

        def _build_ui(self):
            top = ttk.Frame(self.root)
            top.pack(fill=tk.X, padx=10, pady=6)

            ttk.Label(top, text="Year:").pack(side=tk.LEFT)
            self.year_var = tk.StringVar(value="All Years")
            ttk.Combobox(
                top, textvariable=self.year_var,
                values=["All Years","2025","2024","2023","2022","2021"],
                width=10, state="readonly"
            ).pack(side=tk.LEFT, padx=4)

            ttk.Label(top, text="  Mode:").pack(side=tk.LEFT)
            self.mode_var = tk.StringVar(value="auto")
            ttk.Combobox(
                top, textvariable=self.mode_var,
                values=["auto","standard","hyde","multihop"],
                width=10, state="readonly"
            ).pack(side=tk.LEFT, padx=4)

            self.chat = scrolledtext.ScrolledText(
                self.root, wrap=tk.WORD, state=tk.DISABLED, height=28,
                font=("Helvetica", 11)
            )
            self.chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

            bottom = ttk.Frame(self.root)
            bottom.pack(fill=tk.X, padx=10, pady=6)
            self.entry = ttk.Entry(bottom, font=("Helvetica", 11))
            self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.entry.bind("<Return>", lambda _e: self._send())
            self.btn = ttk.Button(bottom, text="Ask", command=self._send)
            self.btn.pack(side=tk.LEFT, padx=6)

        def _init_rag(self):
            from query_rag_v3 import initialize_rag
            ok = initialize_rag()
            msg = "RAG ready." if ok else "Init failed — see console."
            self._append(f"System: {msg}\n")

        def _send(self):
            q = self.entry.get().strip()
            if not q:
                return
            self.entry.delete(0, tk.END)
            self._append(f"\nYou: {q}\n")
            year_str = self.year_var.get()
            year = int(year_str) if year_str != "All Years" else None
            mode = self.mode_var.get()
            threading.Thread(
                target=self._query_thread, args=(q, year, mode), daemon=True
            ).start()

        def _query_thread(self, q, year, mode):
            try:
                from query_rag_v3 import rag_query
                res = rag_query(q, year, mode)
                ans  = res.get("answer", "")
                srcs = {s["paper_id"] for s in res.get("sources", [])}
                text = f"Assistant: {ans}\n"
                if srcs:
                    text += f"Sources: {', '.join(srcs)}\n"
                self.root.after(0, self._append, text + "\n")
            except Exception as e:
                self.root.after(0, self._append, f"Error: {e}\n")

        def _append(self, text):
            self.chat.config(state=tk.NORMAL)
            self.chat.insert(tk.END, text)
            self.chat.see(tk.END)
            self.chat.config(state=tk.DISABLED)

    root = tk.Tk()
    TkRAGApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
