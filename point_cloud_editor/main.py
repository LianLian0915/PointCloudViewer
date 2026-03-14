#!/usr/bin/env python3

import sys

try:
    # Prefer GTK3 implementation if available
    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk
    from app.gtk_editor import Editor as EditorGTK

    def main() -> None:
        win = EditorGTK()
        win.set_default_size(1440, 900)
        win.show_all()
        Gtk.main()

except Exception:
    # Fallback to PySide6 implementation if GTK isn't available
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication
    from app.editor import Editor as EditorQt

    def main() -> None:
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGL)
        fmt.setVersion(2, 1)
        fmt.setProfile(QSurfaceFormat.NoProfile)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
        QSurfaceFormat.setDefaultFormat(fmt)

        app = QApplication(sys.argv)
        w = EditorQt()
        w.resize(1440, 900)
        w.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
