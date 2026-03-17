#!/usr/bin/env python3
"""
Simple test to check if GTK editor starts without errors.
This runs in headless mode (no X11 display expected) for debugging.
"""
import sys
import os

# Add the point_cloud_editor directory to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'point_cloud_editor'))

try:
    print("Importing GTK...")
    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk
    print("GTK imported successfully")
    
    print("Importing app modules...")
    from app.model import PointCloudModel
    from app.gtk_viewer import GLViewer
    from app.gtk_editor import Editor
    print("App modules imported successfully")
    
    print("Creating editor instance...")
    # Try to create the editor (this will fail in headless mode, but we can catch the error)
    try:
        editor = Editor()
        print("Editor created successfully")
        print("Attempting to load demo point cloud...")
        editor.demo()
        print(f"Demo loaded: {editor.model.count} points")
    except Exception as e:
        print(f"Error creating/running editor: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nAll import tests passed!")
    
except Exception as e:
    print(f"Import or initialization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
