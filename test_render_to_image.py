#!/usr/bin/env python3
"""
Test script to render point cloud to image file (headless)
"""
import sys
sys.path.insert(0, 'point_cloud_editor')

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib
from OpenGL.GL import *
import time

from app.gtk_editor import Editor

def main_render_to_image():
    print("[Test] Creating editor window...")
    win = Editor()
    win.set_default_size(800, 600)
    win.show_all()
    
    print("[Test] Loading demo...")
    win.demo()
    
    print("[Test] Triggering render...")
    
    # Process pending events to ensure render
    while Gtk.events_pending():
        Gtk.main_iteration()
    
    # Now try to get the GL surface and save it
    print("[Test] Getting GL surface...")
    gl_area = win.viewer
    
    # Trigger a render
    gl_area.queue_draw()
    
    # Process events to trigger render
    for i in range(10):
        while Gtk.events_pending():
            Gtk.main_iteration()
        time.sleep(0.1)
    
    print("[Test] Attempting to read framebuffer...")
    try:
        # Try to read pixels from OpenGL
        gl_area.make_current()
        
        width = gl_area.get_allocated_width()
        height = gl_area.get_allocated_height()
        
        print(f"[Test] Framebuffer size: {width}x{height}")
        
        # Read pixels from framebuffer
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        print(f"[Test] Pixels read: {len(pixels)} bytes")
        
        # Check if pixels are all black (no render) or have colors
        import numpy as np
        pixels_array = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
        
        # Count non-black pixels
        non_black = np.sum(np.any(pixels_array > 10, axis=2))
        print(f"[Test] Non-black pixels: {non_black} out of {width*height}")
        
        if non_black > 0:
            print("[Test] SUCCESS: Framebuffer contains rendered data!")
            
            # Save to file
            try:
                from PIL import Image
                # Flip vertically because OpenGL coordinates are bottom-left
                img_data = pixels_array[::-1, :, :]
                img = Image.fromarray(img_data, 'RGB')
                img.save('/tmp/render_test.png')
                print("[Test] Image saved to /tmp/render_test.png")
            except Exception as e:
                print(f"[Test] Could not save image: {e}")
        else:
            print("[Test] WARNING: Framebuffer appears to be all black")
        
    except Exception as e:
        print(f"[Test] Error reading framebuffer: {e}")
        import traceback
        traceback.print_exc()
    
    print("[Test] Done")
    win.destroy()

if __name__ == "__main__":
    main_render_to_image()
