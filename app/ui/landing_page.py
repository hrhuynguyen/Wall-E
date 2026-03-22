"""Landing page — Start / End session controls."""

from __future__ import annotations

import tkinter as tk
from typing import Callable


class LandingPage(tk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        on_start: Callable[[], None],
        on_connect_muse: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent, bg="#1e1e2e")
        self._on_start = on_start
        self._on_connect_muse = on_connect_muse
        self._build()

    def _build(self) -> None:
        # Center content vertically
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Title
        tk.Label(
            self,
            text="WallE Assistive Arm",
            font=("Helvetica", 28, "bold"),
            fg="#cdd6f4",
            bg="#1e1e2e",
        ).grid(row=1, column=0, pady=(0, 8))

        # Subtitle
        tk.Label(
            self,
            text="Gaze + BCI controlled robotic arm",
            font=("Helvetica", 14),
            fg="#a6adc8",
            bg="#1e1e2e",
        ).grid(row=2, column=0, pady=(0, 32))

        # Start button
        self._start_btn = tk.Button(
            self,
            text="Start Session",
            font=("Helvetica", 16, "bold"),
            fg="#1e1e2e",
            bg="#a6e3a1",
            activebackground="#94e2d5",
            activeforeground="#1e1e2e",
            relief="flat",
            padx=32,
            pady=12,
            cursor="hand2",
            command=self._on_start,
        )
        self._start_btn.grid(row=3, column=0)

        # Connect Muse button + status
        muse_frame = tk.Frame(self, bg="#1e1e2e")
        muse_frame.grid(row=4, column=0, pady=(16, 0))

        self._muse_btn = tk.Button(
            muse_frame,
            text="Connect to Muse 2",
            font=("Helvetica", 12),
            fg="#cdd6f4",
            bg="#45475a",
            activebackground="#585b70",
            activeforeground="#cdd6f4",
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
            command=self._on_connect_muse or (lambda: None),
        )
        self._muse_btn.pack(side="left")

        self._muse_status = tk.Label(
            muse_frame,
            text="",
            font=("Helvetica", 11),
            fg="#a6adc8",
            bg="#1e1e2e",
        )
        self._muse_status.pack(side="left", padx=(10, 0))

    def set_muse_status(self, text: str, color: str = "#a6adc8") -> None:
        """Update the Muse connection status text."""
        self._muse_status.configure(text=text, fg=color)

    def set_muse_button_enabled(self, enabled: bool) -> None:
        self._muse_btn.configure(state="normal" if enabled else "disabled")

    def set_status(self, text: str) -> None:
        """Show a status message below the buttons (e.g. error feedback)."""
        if hasattr(self, "_status_label"):
            self._status_label.configure(text=text)
        else:
            self._status_label = tk.Label(
                self,
                text=text,
                font=("Helvetica", 12),
                fg="#f38ba8",
                bg="#1e1e2e",
            )
            self._status_label.grid(row=5, column=0, sticky="n", pady=(16, 0))
