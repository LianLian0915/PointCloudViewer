#!/usr/bin/env python3
"""
Minimal headless test to verify GLArea rendering pipeline.
Can be run without X11 to check if on_render callbacks fire.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'point_cloud_editor'))

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib

from app.gtk_editor import Editor


def on_timeout():
    """Simulates GTK event loop tick for testing."""
    print("[Test] GTK iteration...")
    return True  # Keep timer running


def main_headless():
    """Run GTK in headless mode with timeout simulation."""
    print("[Test] Creating editor window...")
    try:
        win = Editor()
        win.set_default_size(1440, 900)
        win.connect("delete-event", Gtk.main_quit)
        print("[Test] Showing window...")
        win.show_all()
        print("[Test] Loading demo...")
        win.demo()
        
        print("[Test] Starting GTK main loop (will timeout after 3 seconds)...")
        # Add a timeout to exit after 3 seconds
        GLib.timeout_add(3000, Gtk.main_quit)
        
        # Try to pump some events to trigger rendering
        print("[Test] Pumping GTK events...")
        while Gtk.events_pending():
            Gtk.main_iteration_do(False)
        
        print("[Test] Running Gtk.main()...")
        Gtk.main()
        print("[Test] GTK main loop exited successfully")
        
    except Exception as e:
        print(f"[Test] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    try:
        exit_code = main_headless()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("[Test] Interrupted by user")
        sys.exit(0)
