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


# ─── Settings Dialog ─────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    """Modal settings dialog that writes back to Config."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)
        self._build_ui()

    def _build_ui(self):
        from config import Config
        layout = QVBoxLayout(self)

        # ── LLM Provider ──────────────────────────────────────────────────
        llm_box = QGroupBox("LLM Provider")
        llm_form = QFormLayout(llm_box)

        self.chk_use_openai = QCheckBox("Use OpenAI API (instead of LM Studio)")
        self.chk_use_openai.setChecked(Config.USE_OPENAI)
        llm_form.addRow(self.chk_use_openai)

        self.txt_openai_key = QLineEdit(Config.OPENAI_API_KEY or "")
        self.txt_openai_key.setPlaceholderText("sk-...")
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

        self.chk_use_openai_emb = QCheckBox(
            "Use OpenAI embeddings (text-embedding-3-small)"
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

        warn = QLabel("⚠  Changing the embedding model requires a full index rebuild.")
        warn.setWordWrap(True)
        warn.setStyleSheet("color: #e67e22; font-size: 11px;")
        emb_form.addRow(warn)

        layout.addWidget(emb_box)

        # ── RAG Parameters ────────────────────────────────────────────────
        rag_box = QGroupBox("RAG Parameters")
        rag_form = QFormLayout(rag_box)

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

        self.chk_hybrid   = QCheckBox("Hybrid Search (BM25 + Semantic)")
        self.chk_hybrid.setChecked(Config.USE_HYBRID_SEARCH)
        feat_form.addRow(self.chk_hybrid)

        self.chk_reranker = QCheckBox("Cross-Encoder Re-ranking")
        self.chk_reranker.setChecked(Config.USE_RERANKER)
        feat_form.addRow(self.chk_reranker)

        self.chk_hyde     = QCheckBox("HyDE (Hypothetical Document Embeddings)")
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


# ─── Worker Thread ────────────────────────────────────────────────────────────

class QueryWorker(QThread):
    """Runs the RAG pipeline in a background thread to keep UI responsive."""

    finished  = pyqtSignal(dict)   # emits full result dict
    error     = pyqtSignal(str)    # emits error string

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

        self.status_browser = QTextBrowser()
        self.status_browser.setFont(QFont("Courier", 11))
        layout.addWidget(self.status_browser)

        refresh_btn = QPushButton("🔄 Refresh Status")
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(refresh_btn)

        self.refresh()

    def refresh(self):
        from config import Config
        s = Config.get_system_status()
        llm = Config.get_active_llm_info()
        emb = Config.get_active_embedding_info()

        html = """<style>
            body  { font-family: Courier New, monospace; font-size: 13px; }
            h3    { color: #2c3e50; }
            .key  { color: #2980b9; font-weight: bold; }
            .val  { color: #27ae60; }
            .warn { color: #e67e22; }
        </style>"""

        def row(k, v, warn=False):
            cls = "warn" if warn else "val"
            return f'<tr><td class="key">{k}</td><td class="{ cls }">{v}</td></tr>'

        html += "<h3>📊 System Status</h3><table cellpadding='4'>"
        html += row("PDFs downloaded",    s.get("pdfs_count", 0))
        html += row("Text files",         s.get("text_files_count", 0))
        db_ok = s.get("vector_db_exists", False)
        html += row("Vector DB",          "✅ Ready" if db_ok else "❌ Missing",
                    warn=not db_ok)
        html += row("DB size",            f"{s.get('vector_db_size_mb', 0)} MB")
        html += row("Cache entries",      s.get("cache_entries", 0))
        html += "</table>"

        html += "<h3>🤖 Active LLM</h3><table cellpadding='4'>"
        html += row("Provider", llm["provider"])
        html += row("Model",    llm["model"])
        html += row("API base", llm["api_base"])
        html += "</table>"

        html += "<h3>🔢 Embeddings</h3><table cellpadding='4'>"
        html += row("Provider", emb["provider"])
        html += row("Model",    emb["model"])
        html += "</table>"

        from config import Config as C
        html += "<h3>⚙️ Feature Toggles</h3><table cellpadding='4'>"
        html += row("Hybrid Search",  "ON ✅" if C.USE_HYBRID_SEARCH else "OFF")
        html += row("HyDE",           "ON ✅" if C.USE_HYDE          else "OFF")
        html += row("Re-ranking",     "ON ✅" if C.USE_RERANKER      else "OFF")
        html += row("Multi-hop",      "ON ✅" if C.USE_MULTIHOP      else "OFF")
        html += row("Persist Cache",  "ON ✅" if C.USE_PERSISTENT_CACHE else "OFF")
        html += "</table>"

        self.status_browser.setHtml(html)


# ─── Sources Panel ────────────────────────────────────────────────────────────

class SourcesPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel("📚 Sources")
        lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(lbl)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setFont(QFont("Segoe UI", 10))
        self.browser.setPlaceholderText("Sources will appear here after a query.")
        layout.addWidget(self.browser)

    def update_sources(self, sources: list):
        if not sources:
            self.browser.setPlainText("No sources for this response.")
            return

        html = "<style>a{color:#2980b9;} .pid{font-weight:bold;} .ttl{font-style:italic;}</style>"
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

            html += f'<p><span class="pid">📄 {pid}'
            if yr:
                html += f" ({yr})"
            html += "</span>"
            if ttl:
                html += f'<br><span class="ttl">{ttl[:100]}</span>'
            if aut:
                html += f"<br><small>{aut[:100]}</small>"
            if url:
                html += f'<br><a href="{url}">Open on arXiv ↗</a>'
            if snp:
                html += f"<br><small style='color:#555'>{snp[:200]}…</small>"
            html += "<hr></p>"

        self.browser.setHtml(html)

    def clear(self):
        self.browser.clear()


# ─── Chat Tab ─────────────────────────────────────────────────────────────────

class ChatTab(QWidget):
    """Main chat interface."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window  = main_window
        self._worker: Optional[QueryWorker] = None
        self._history: list = []      # [(user, assistant), …]
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Top controls bar ─────────────────────────────────────────────
        ctrl_bar = QHBoxLayout()

        ctrl_bar.addWidget(QLabel("📅 Year:"))
        self.year_combo = QComboBox()
        self.year_combo.addItems(
            ["All Years", "2025", "2024", "2023", "2022", "2021", "2020"]
        )
        self.year_combo.setFixedWidth(110)
        ctrl_bar.addWidget(self.year_combo)

        ctrl_bar.addWidget(QLabel("  🔧 Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["auto", "standard", "hyde", "multihop"])
        self.mode_combo.setFixedWidth(100)
        self.mode_combo.setToolTip(
            "auto     — system picks best mode\n"
            "standard — direct semantic search\n"
            "hyde     — hypothetical doc embedding\n"
            "multihop — decompose → sub-queries → synthesise"
        )
        ctrl_bar.addWidget(self.mode_combo)

        ctrl_bar.addStretch()

        self.clear_btn = QPushButton("🗑 Clear Chat")
        self.clear_btn.setFixedWidth(110)
        self.clear_btn.clicked.connect(self._clear_chat)
        ctrl_bar.addWidget(self.clear_btn)

        self.cache_btn = QPushButton("⚡ Clear Cache")
        self.cache_btn.setFixedWidth(115)
        self.cache_btn.clicked.connect(self._clear_cache)
        ctrl_bar.addWidget(self.cache_btn)

        root.addLayout(ctrl_bar)

        # ── Main splitter: chat | sources ─────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Chat display
        chat_frame = QFrame()
        chat_layout = QVBoxLayout(chat_frame)
        chat_layout.setContentsMargins(0, 0, 0, 0)

        self.chat_display = QTextBrowser()
        self.chat_display.setFont(QFont("Segoe UI", 12))
        self.chat_display.setOpenExternalLinks(True)
        chat_layout.addWidget(self.chat_display)

        # Input row
        input_row = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setFont(QFont("Segoe UI", 12))
        from config import Config
        self.query_input.setPlaceholderText(
            f"Ask anything about {Config.TOPIC_NAME} research…"
        )
        self.query_input.returnPressed.connect(self._send_query)
        input_row.addWidget(self.query_input, stretch=1)

        self.send_btn = QPushButton("Ask ➤")
        self.send_btn.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.send_btn.setFixedWidth(80)
        self.send_btn.clicked.connect(self._send_query)
        self.send_btn.setStyleSheet(
            "QPushButton { background:#2980b9; color:white; border-radius:5px; padding:6px; }"
            "QPushButton:hover { background:#3498db; }"
            "QPushButton:disabled { background:#bdc3c7; }"
        )
        input_row.addWidget(self.send_btn)
        chat_layout.addLayout(input_row)

        # Progress bar (hidden by default)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # indeterminate
        self.progress.setVisible(False)
        self.progress.setFixedHeight(5)
        chat_layout.addWidget(self.progress)

        splitter.addWidget(chat_frame)

        # Sources panel
        self.sources_panel = SourcesPanel()
        splitter.addWidget(self.sources_panel)
        splitter.setSizes([700, 300])

        root.addWidget(splitter, stretch=1)

        # ── Quick example questions ───────────────────────────────────────
        from config import Config
        ex_bar = QHBoxLayout()
        ex_bar.addWidget(QLabel("💡 Examples:"))
        examples = Config.EXAMPLE_QUESTIONS[:3]
        for ex in examples:
            btn = QPushButton(ex)
            btn.setStyleSheet(
                "QPushButton { border:1px solid #bdc3c7; border-radius:4px; "
                "padding:3px 8px; background:#ecf0f1; font-size:11px; }"
                "QPushButton:hover { background:#d5dbdb; }"
            )
            btn.clicked.connect(lambda _, t=ex: self._use_example(t))
            ex_bar.addWidget(btn)
        ex_bar.addStretch()
        root.addLayout(ex_bar)

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
                self.main_window.set_status("✅ Cache cleared.")
            else:
                self.main_window.set_status("Cache not enabled.")
        except Exception as e:
            self.main_window.set_status(f"Cache clear error: {e}")

    def _append_message(self, role: str, text: str):
        """Append a styled message bubble to the chat display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)

        if role == "user":
            html = (
                f'<div style="margin:8px 0; padding:8px 12px; '
                f'background:#d6eaf8; border-radius:8px; '
                f'font-size:13px;">'
                f'<b style="color:#1a5276;">You</b><br>{text}</div>'
            )
        elif role == "assistant":
            html = (
                f'<div style="margin:8px 0; padding:8px 12px; '
                f'background:#eafaf1; border-radius:8px; '
                f'font-size:13px;">'
                f'<b style="color:#1e8449;">Assistant</b><br>{text}</div>'
            )
        else:   # system / info
            html = (
                f'<div style="margin:4px 0; padding:6px 10px; '
                f'background:#fef9e7; border-radius:6px; '
                f'font-size:11px; color:#7d6608;">{text}</div>'
            )

        self.chat_display.insertHtml(html)
        # scroll to bottom
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

        self.main_window.set_status("⏳ Thinking…")

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

        # Build info footer
        info_parts = [
            f"mode={mode}",
            f"time={t}s",
            "cache=HIT ⚡" if cache else "cache=MISS",
        ]
        if len(sub_qs) > 1:
            info_parts.append(f"hops={len(sub_qs)}")
        footer = " | ".join(info_parts)

        self._append_message("assistant", answer)
        self._append_message("system", footer)

        # Update sources panel
        self.sources_panel.update_sources(result.get("sources", []))
        self.main_window.set_status(f"Done  |  {footer}")

    def _on_error(self, error_str: str):
        self.send_btn.setEnabled(True)
        self.progress.setVisible(False)
        self._append_message("system", f"❌ Error: {error_str}")
        self.main_window.set_status(f"Error: {error_str[:80]}")


# ─── Main Window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from config import Config
        self.setWindowTitle(f"{Config.TOPIC_NAME} Research Assistant  (RAG v3)")
        self.setMinimumSize(1100, 720)
        self._rag_ready = False
        self._build_menu()
        self._build_ui()
        self._start_init()

    # ── Menu bar ──────────────────────────────────────────────────────────────

    def _build_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        settings_act = QAction("⚙  Settings…", self)
        settings_act.triggered.connect(self._open_settings)
        file_menu.addAction(settings_act)
        file_menu.addSeparator()
        quit_act = QAction("Quit", self)
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
        layout.setContentsMargins(8, 8, 8, 8)

        # ── Header ────────────────────────────────────────────────────────
        from config import Config
        header = QLabel(f"🤖  {Config.TOPIC_NAME} Research Assistant")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setStyleSheet("color: #2c3e50; padding: 4px 0;")
        layout.addWidget(header)

        self.init_label = QLabel("⏳ Initialising RAG system…")
        self.init_label.setStyleSheet("color: #e67e22; font-size: 12px;")
        layout.addWidget(self.init_label)

        # ── Tabs ─────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.chat_tab   = ChatTab(self)
        self.status_tab = StatusTab()
        self.tabs.addTab(self.chat_tab,   "💬 Chat")
        self.tabs.addTab(self.status_tab, "📊 Status")
        layout.addWidget(self.tabs, stretch=1)

        # ── Status bar ───────────────────────────────────────────────────
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
        self.init_label.setText(f"✅ {message}")
        self.init_label.setStyleSheet("color: #1e8449; font-size: 12px;")
        self.chat_tab.send_btn.setEnabled(True)
        self.status_bar.showMessage(message)
        self.status_tab.refresh()
        from config import Config
        self.chat_tab._append_message(
            "system",
            f"RAG system ready!  {message}<br>"
            f"Ask me anything about {Config.TOPIC_NAME} research 👇"
        )

    def _on_init_failure(self, error: str):
        self.init_label.setText(f"❌ {error}")
        self.init_label.setStyleSheet("color: #c0392b; font-size: 12px;")
        self.status_bar.showMessage(f"Init failed: {error}")
        from config import Config
        self.chat_tab._append_message(
            "system",
            f"❌ Initialisation failed: {error}<br><br>"
            "Possible fixes:<br>"
            f"• Build the index first: <tt>python build_rag_index_v4.py</tt><br>"
            "• If using OpenAI: set <tt>export OPENAI_API_KEY=sk-…</tt> and enable in Settings<br>"
            "• If using LM Studio: ensure it is running on <tt>localhost:1234</tt><br>"
            f"• Topic configured: <tt>{Config.TOPIC_NAME}</tt>"
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
                "⚠  Note: Changes take effect after restarting the app\n"
                "    (or re-initialising via File › Settings).\n\n"
                "If you changed the embedding model, rebuild the index:\n"
                "  python build_rag_index_v4.py --rebuild"
            )
            self.status_tab.refresh()

    def _show_about(self):
        from config import Config
        QMessageBox.about(
            self, "About",
            f"<b>{Config.TOPIC_NAME} Research Assistant — RAG v3</b><br><br>"
            "A generic desktop application for querying arXiv research papers<br>"
            f"on <i>{Config.TOPIC_NAME}</i> using Retrieval-Augmented Generation.<br><br>"
            f"<b>Topic:</b> {Config.TOPIC_NAME}<br>"
            f"<b>Description:</b> {Config.TOPIC_DESCRIPTION}<br><br>"
            "<b>Features:</b><br>"
            "• Hybrid BM25 + Semantic search<br>"
            "• HyDE hypothetical document embeddings<br>"
            "• Cross-encoder re-ranking<br>"
            "• Multi-hop chain-of-thought reasoning<br>"
            "• Persistent SQLite query cache<br>"
            "• OpenAI API & LM Studio support<br><br>"
            "Built with PyQt6 · LangChain · ChromaDB · sentence-transformers"
        )


# ─── Application Entry-point ─────────────────────────────────────────────────

def main():
    if not PYQT_AVAILABLE:
        print("PyQt6 is not installed.")
        print("Run: pip install PyQt6")
        print()
        print("Falling back to Tkinter desktop app…")
        _run_tkinter_fallback()
        return

    app = QApplication(sys.argv)
    from config import Config
    app.setApplicationName(f"{Config.TOPIC_NAME} RAG")
    app.setApplicationVersion("3.0")

    # ── Light application style ───────────────────────────────────────────
    app.setStyleSheet("""
        QMainWindow, QWidget { background: #f5f6fa; }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #bdc3c7;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
            color: #2c3e50;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            border: 1px solid #bdc3c7;
            border-radius: 4px;
            padding: 4px 6px;
            background: white;
        }
        QLineEdit:focus, QComboBox:focus {
            border-color: #2980b9;
        }
        QTextBrowser, QTextEdit {
            border: 1px solid #dfe6e9;
            border-radius: 4px;
            background: white;
        }
        QTabWidget::pane {
            border: 1px solid #dfe6e9;
            border-radius: 4px;
        }
        QTabBar::tab {
            padding: 6px 18px;
            background: #ecf0f1;
            border: 1px solid #dfe6e9;
        }
        QTabBar::tab:selected {
            background: white;
            font-weight: bold;
        }
        QStatusBar { background: #2c3e50; color: white; padding: 2px 6px; }
        QProgressBar::chunk { background: #2980b9; }
    """)

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
                font=("Segoe UI", 11)
            )
            self.chat.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

            bottom = ttk.Frame(self.root)
            bottom.pack(fill=tk.X, padx=10, pady=6)
            self.entry = ttk.Entry(bottom, font=("Segoe UI", 11))
            self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.entry.bind("<Return>", lambda _e: self._send())
            self.btn = ttk.Button(bottom, text="Ask ➤", command=self._send)
            self.btn.pack(side=tk.LEFT, padx=6)

        def _init_rag(self):
            from query_rag_v3 import initialize_rag
            ok = initialize_rag()
            msg = "✅ RAG ready." if ok else "❌ Init failed — see console."
            self._append(f"System: {msg}\n")
            if ok:
                self.btn.configure(state=tk.NORMAL)

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
