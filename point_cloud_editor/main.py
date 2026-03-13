#!/usr/bin/env python3

import sys
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication
from app.editor import Editor


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
    w = Editor()
    w.resize(1360, 860)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
