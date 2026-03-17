#!/usr/bin/env python3

import sys
import os

# Ensure the current directory (where this script is) is in the path
# so that the 'app' module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from app.gtk_editor import Editor as EditorGTK


def main() -> None:
    win = EditorGTK()
    win.set_default_size(1440, 900)
    # Connect close event to cleanly exit
    win.connect("delete-event", Gtk.main_quit)
    win.show_all()
    # Auto-load demo point cloud on startup
    win.demo()
    Gtk.main()


if __name__ == "__main__":
    main()
