"""Session page — camera feed (left) + status sidebar (right)."""

from __future__ import annotations

import tkinter as tk
from typing import Callable

from app.config import DISPLAY_HEIGHT, DISPLAY_WIDTH

# Catppuccin Mocha palette
_BG = "#1e1e2e"
_SURFACE = "#313244"
_TEXT = "#cdd6f4"
_SUBTEXT = "#a6adc8"
_GREEN = "#a6e3a1"
_RED = "#f38ba8"
_YELLOW = "#f9e2af"
_BLUE = "#89b4fa"

SIDEBAR_WIDTH = 240


class SessionPage(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        on_end: Callable[[], None],
        on_connect_muse: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, bg=_BG)
        self._on_end = on_end
        self._on_connect_muse = on_connect_muse
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        # ---- Left: camera feed ----
        self.camera_canvas = tk.Canvas(
            self,
            width=DISPLAY_WIDTH,
            height=DISPLAY_HEIGHT,
            bg="#000000",
            highlightthickness=0,
        )
        self.camera_canvas.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)

        # ---- Right: sidebar ----
        sidebar = tk.Frame(self, bg=_SURFACE, width=SIDEBAR_WIDTH)
        sidebar.grid(row=0, column=1, sticky="ns", padx=(0, 16), pady=16)
        sidebar.grid_propagate(False)

        # -- Status header --
        tk.Label(
            sidebar, text="Status", font=("Helvetica", 16, "bold"),
            fg=_TEXT, bg=_SURFACE,
        ).pack(anchor="w", padx=16, pady=(16, 12))

        # -- State row --
        self._state_label = self._make_row(sidebar, "State", "Landing")

        # -- Device rows --
        self._xarm_label = self._make_row(sidebar, "xArm", "Disconnected")
        self._muse_label = self._make_row(sidebar, "Muse", "Disconnected")

        # -- EEG row --
        self._eeg_label = self._make_row(sidebar, "EEG", "Off")

        # -- Connect Muse button + status --
        muse_frame = tk.Frame(sidebar, bg=_SURFACE)
        muse_frame.pack(fill="x", padx=16, pady=(8, 0))

        self._muse_btn = tk.Button(
            muse_frame,
            text="Connect Muse 2",
            font=("Helvetica", 10),
            fg=_TEXT,
            bg="#45475a",
            activebackground="#585b70",
            activeforeground=_TEXT,
            relief="flat",
            padx=8,
            pady=3,
            cursor="hand2",
            command=self._on_connect_muse or (lambda: None),
        )
        self._muse_btn.pack(anchor="w")

        self._muse_status_label = tk.Label(
            muse_frame,
            text="",
            font=("Helvetica", 9),
            fg=_SUBTEXT,
            bg=_SURFACE,
            anchor="w",
        )
        self._muse_status_label.pack(anchor="w", pady=(2, 0))

        # -- Separator --
        sep = tk.Frame(sidebar, bg=_SUBTEXT, height=1)
        sep.pack(fill="x", padx=16, pady=12)

        # -- Object info --
        tk.Label(
            sidebar, text="Selection", font=("Helvetica", 13, "bold"),
            fg=_TEXT, bg=_SURFACE,
        ).pack(anchor="w", padx=16, pady=(0, 8))

        self._highlight_label = self._make_row(sidebar, "Gaze", "—")
        self._selected_label = self._make_row(sidebar, "Selected", "—")

        # -- Spacer --
        spacer = tk.Frame(sidebar, bg=_SURFACE)
        spacer.pack(fill="both", expand=True)

        # -- End Session button --
        end_btn = tk.Button(
            sidebar,
            text="End Session",
            font=("Helvetica", 13),
            fg=_BG,
            bg=_RED,
            activebackground="#eba0ac",
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._on_end,
        )
        end_btn.pack(fill="x", padx=16, pady=(0, 16))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_row(self, parent: tk.Widget, label: str, initial: str) -> tk.Label:
        row = tk.Frame(parent, bg=_SURFACE)
        row.pack(fill="x", padx=16, pady=2)
        tk.Label(
            row, text=f"{label}:", font=("Helvetica", 11),
            fg=_SUBTEXT, bg=_SURFACE, width=8, anchor="w",
        ).pack(side="left")
        value = tk.Label(
            row, text=initial, font=("Helvetica", 11),
            fg=_TEXT, bg=_SURFACE, anchor="w",
        )
        value.pack(side="left", fill="x", expand=True)
        return value

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def update_state(self, state_name: str) -> None:
        self._state_label.configure(text=state_name)

    def update_xarm(self, connected: bool) -> None:
        text = "Connected" if connected else "Disconnected"
        color = _GREEN if connected else _RED
        self._xarm_label.configure(text=text, fg=color)

    def update_muse(self, connected: bool) -> None:
        text = "Connected" if connected else "Disconnected"
        color = _GREEN if connected else _RED
        self._muse_label.configure(text=text, fg=color)

    def update_eeg(self, status: str) -> None:
        color = _GREEN if status == "Active" else _YELLOW if "Warmup" in status else _TEXT
        self._eeg_label.configure(text=status, fg=color)

    def set_muse_status(self, text: str, color: str = _SUBTEXT) -> None:
        self._muse_status_label.configure(text=text, fg=color)

    def set_muse_button_enabled(self, enabled: bool) -> None:
        self._muse_btn.configure(state="normal" if enabled else "disabled")

    def update_highlight(self, name: str | None) -> None:
        self._highlight_label.configure(text=name or "—")

    def update_selected(self, name: str | None) -> None:
        self._selected_label.configure(text=name or "—")
