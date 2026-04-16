from __future__ import annotations

import math
import numpy as np
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk
import ctypes

# Configure PyOpenGL to work better with GTK GLArea
import OpenGL
OpenGL.ERROR_CHECKING = False  # Disable error checking 
OpenGL.ALLOW_NUMPY_FUNCTIONS = False  # Don't use numpy arrays which might trigger context checks

from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
from numpy.typing import NDArray
import moderngl

from app.camera import Camera
from app.lod import LODBuilder
from app.model import PointCloudModel
from app.selection import box_select_indices, brush_select_indices

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None


def create_projection_matrix(fovy: float, aspect: float, znear: float, zfar: float) -> NDArray:
    """Create a perspective projection matrix (CPU-side)."""
    # Standard perspective projection matrix
    # fovy is in degrees
    f = 1.0 / math.tan(math.radians(fovy) * 0.5)
    result = np.zeros((4, 4), dtype=np.float32)
    result[0, 0] = f / aspect
    result[1, 1] = f
    result[2, 2] = (zfar + znear) / (znear - zfar)
    result[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    result[3, 2] = -1.0
    return result


def create_view_matrix(camera: 'Camera') -> NDArray:
    """Create a view matrix from camera (CPU-side)."""
    # Build view matrix: translate to -center, rotate, then move back by camera distance
    # Standard order: V = T(0,0,-dist) * Ry(yaw) * Rx(pitch) * T(-center)
    
    # First translate to remove center
    view = np.eye(4, dtype=np.float32)
    view[0, 3] = -camera.center[0]
    view[1, 3] = -camera.center[1]
    view[2, 3] = -camera.center[2]
    
    # Rotate around X axis (pitch)
    pitch_rad = camera.pitch
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    pitch_mat = np.eye(4, dtype=np.float32)
    pitch_mat[1, 1] = cp
    pitch_mat[1, 2] = -sp
    pitch_mat[2, 1] = sp
    pitch_mat[2, 2] = cp
    view = pitch_mat @ view
    
    # Rotate around Y axis (yaw)
    yaw_rad = camera.yaw
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    yaw_mat = np.eye(4, dtype=np.float32)
    yaw_mat[0, 0] = cy
    yaw_mat[0, 2] = sy
    yaw_mat[2, 0] = -sy
    yaw_mat[2, 2] = cy
    view = yaw_mat @ view
    
    # Finally, translate along -Z by camera distance
    dist_mat = np.eye(4, dtype=np.float32)
    dist_mat[2, 3] = -camera.dist
    view = dist_mat @ view
    
    return view


class GLViewer(Gtk.GLArea):
    def __init__(self, model: PointCloudModel, status_callback) -> None:
        super().__init__()
        self.set_has_depth_buffer(True)
        self.set_has_stencil_buffer(False)
        
        # Try to set OpenGL version to request a compatible profile
        # This may help avoid core-profile-only contexts
        try:
            # Use OpenGL 3.2 for better compatibility
            self.set_required_version(3, 2)
        except Exception:
            # Method might not exist in older PyGObject
            pass
        
        self.connect("realize", self.on_realize)
        self.connect("unrealize", self.on_unrealize)
        self.connect("render", self.on_render)

        # enable mouse events
        self.set_events(
            Gdk.EventMask.BUTTON_PRESS_MASK
            | Gdk.EventMask.BUTTON_RELEASE_MASK
            | Gdk.EventMask.POINTER_MOTION_MASK
            | Gdk.EventMask.SCROLL_MASK
        )
        self.connect("button-press-event", self.on_button_press)
        self.connect("button-release-event", self.on_button_release)
        self.connect("motion-notify-event", self.on_motion_notify)
        self.connect("scroll-event", self.on_scroll)

        self.model = model
        self.status_callback = status_callback
        self.camera = Camera()
        self.scene_radius = 1.0
        self.point_size = 3.0
        self.last_pos = None
        self.interactive = False  # when True, use lightweight LOD for smooth interaction
        self.set_size_request(640, 480)

        self.lod_levels: dict[str, np.ndarray] = {
            "full": np.zeros((0,), dtype=np.int32),
            "medium": np.zeros((0,), dtype=np.int32),
            "preview": np.zeros((0,), dtype=np.int32),
        }
        self.render_indices = np.zeros((0,), dtype=np.int32)
        self.render_positions = np.zeros((0, 3), dtype=np.float32)
        self.render_gpu_positions = np.zeros((0, 3), dtype=np.float32)
        self.render_colors = np.zeros((0, 3), dtype=np.float32)
        self.render_tree = None
        self.use_preview = True

        self.pos_vbo = None
        self.col_vbo = None
        self.gpu_dirty = True
        self.cache_dirty = True

        self.selection_mode = "point"
        self.dragging_box = False
        self.box_start = None
        self.box_end = None
        self.brush_radius = 20.0
        self.brush_active = False

        self._gl_inited = False
        # modern GL objects
        self.shader_program = None
        self.vao = None
        self.vao_setup_deferred = False
        self.u_mvp = None
        self.u_point_size = None
        # ModernGL context and objects
        self.mgl_ctx = None
        self.mgl_prog = None
        self.mgl_pos_vbo = None
        self.mgl_col_vbo = None
        self.mgl_vao = None
        self.moderngl_enabled = True

    # --- GL lifecycle ---
    def on_realize(self, widget) -> None:
        # GL context available after realize
        self.make_current()
        print("[GLViewer] on_realize called - GL context available")
        # nothing heavy here; actual GL init in first render

    def on_unrealize(self, widget) -> None:
        print("[GLViewer] on_unrealize called")
        self.make_current()
        try:
            if self.pos_vbo:
                glDeleteBuffers(1, [self.pos_vbo])
                self.pos_vbo = None
            if self.col_vbo:
                glDeleteBuffers(1, [self.col_vbo])
                self.col_vbo = None
        except Exception:
            pass

    def on_render(self, area, ctx) -> bool:
        # called with valid GL context
        print(f"[GLViewer] on_render called - _gl_inited={self._gl_inited}")
        if not self._gl_inited:
            print("[GLViewer] Initializing GL...")
            try:
                self.initializeGL()
                self._gl_inited = True
                print("[GLViewer] GL initialized successfully")
            except Exception as e:
                print(f"[GLViewer] GL initialization failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # Set up VAO if deferred
        if self.vao_setup_deferred and self.vao and self.vao > 0:
            print(f"[GLViewer] Setting up deferred VAO {self.vao} in on_render")
            try:
                glBindVertexArray(self.vao)
                glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
                glBindBuffer(GL_ARRAY_BUFFER, self.col_vbo)
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glBindVertexArray(0)
                self.vao_setup_deferred = False
                print(f"[GLViewer] VAO setup complete in on_render")
            except Exception as e:
                print(f"[GLViewer] Deferred VAO setup failed in on_render: {e}")
                import traceback
                traceback.print_exc()
                self.vao = None
                self.vao_setup_deferred = False
        
        # Now call paintGL
        self.paintGL()
        return True

        # paint equivalent
        try:
            self.paintGL()
        except Exception as e:
            print(f"[GLViewer] paintGL failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        return True

    def initializeGL(self) -> None:
        print("[GLViewer.initializeGL] Starting GL initialization")
        try:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            # GL_POINT_SMOOTH may be unsupported in some GL contexts; skip if invalid
            try:
                glDisable(GL_POINT_SMOOTH)
            except Exception:
                pass
            self.pos_vbo = glGenBuffers(1)
            self.col_vbo = glGenBuffers(1)
            print(f"[GLViewer.initializeGL] Created VBOs: pos={self.pos_vbo}, col={self.col_vbo}")
            
            version = glGetString(GL_VERSION)
            renderer = glGetString(GL_RENDERER)
            v = version.decode("utf-8", errors="ignore") if version else "unknown"
            r = renderer.decode("utf-8", errors="ignore") if renderer else "unknown"
            print(f"[GLViewer.initializeGL] OpenGL: {v} | Renderer: {r}")
            try:
                self.status_callback(f"OpenGL: {v} | Renderer: {r}")
            except Exception:
                pass
            # Try to create a simple shader program for modern GL contexts
            # Note: Positions are pre-transformed on CPU side, so shader just uses them directly
            vert_src = '''#version 330
        layout(location = 0) in vec3 in_pos;  // pre-transformed position in clip space
        layout(location = 1) in vec3 in_col;
        uniform mat4 u_mvp;
        uniform float u_point_size;
        out vec3 v_col;
        void main() {
            gl_Position = u_mvp * vec4(in_pos, 1.0);
            v_col = in_col;
            gl_PointSize = u_point_size;
        }
        '''
            frag_src = '''#version 330
        in vec3 v_col;
        out vec4 out_col;
        void main() {
            out_col = vec4(v_col, 1.0);
        }
        '''
            try:
                # Initialize ModernGL context from the current GL context
                try:
                    self.make_current()
                except Exception:
                    pass
                try:
                    self.mgl_ctx = moderngl.create_context()
                    print(f"[GLViewer.initializeGL] ModernGL context created: {self.mgl_ctx}")
                except Exception as e:
                    print(f"[GLViewer.initializeGL] ModernGL context creation failed: {e}")
                    self.mgl_ctx = None

                if self.mgl_ctx is not None:
                    # Create ModernGL program (shader)
                    try:
                        self.mgl_prog = self.mgl_ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
                        print("[GLViewer.initializeGL] ModernGL program created")

                        # Create buffers with minimal size (VAO will be created in upload_gpu_buffers when data is available)
                        self.mgl_pos_vbo = self.mgl_ctx.buffer(reserve=12)  # Minimum size for 1 point
                        self.mgl_col_vbo = self.mgl_ctx.buffer(reserve=12)  # Minimum size for 1 point
                        print("[GLViewer.initializeGL] ModernGL buffers created")

                        # VAO will be created in upload_gpu_buffers when we have actual data
                        self.mgl_vao = None

                        # Enable program point size if supported
                        try:
                            glEnable(GL_PROGRAM_POINT_SIZE)
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"[GLViewer.initializeGL] ModernGL program creation failed: {e}")
                        self.mgl_prog = None
                else:
                    print("[GLViewer.initializeGL] ModernGL unavailable, will attempt PyOpenGL fallback")
                    self.mgl_prog = None
            except Exception as e:
                print(f"[GLViewer.initializeGL] Shader creation failed: {e}, will use fixed-function or fallback")
                self.shader_program = None
        except Exception as e:
            print(f"[GLViewer.initializeGL] Critical error: {e}")
            import traceback
            traceback.print_exc()
            raise

    # --- GL data & drawing ---
    def mark_model_dirty(self) -> None:
        self.cache_dirty = True
        self.gpu_dirty = True
        self.queue_draw()

    def update_moved_points(self, indices: np.ndarray) -> None:
        if indices.size == 0:
            return
        if self.cache_dirty or self.render_indices.size == 0:
            self.mark_model_dirty()
            return
        moved_visible = np.isin(self.render_indices, indices)
        if not np.any(moved_visible):
            self.queue_draw()
            return
        self.render_positions[moved_visible] = self.model.positions[self.render_indices[moved_visible]]
        self.render_gpu_positions[moved_visible] = (
            self.render_positions[moved_visible] - self.camera.center[None, :]
        )
        self.gpu_dirty = True
        self.queue_draw()

    def set_selection_mode(self, mode: str) -> None:
        self.selection_mode = mode

    def set_preview_enabled(self, enabled: bool) -> None:
        self.use_preview = enabled
        self.mark_model_dirty()

    def update_after_soft_delete(self) -> None:
        """Lightweight cache update after soft-delete (mark as deleted but not yet compacted)"""
        self.cache_dirty = True
        self.gpu_dirty = True
        self.queue_draw()

    def rebuild_render_cache(self) -> None:
        alive_count = self.model.alive_count
        if alive_count == 0:
            self.render_indices = np.zeros((0,), dtype=np.int32)
            self.render_positions = np.zeros((0, 3), dtype=np.float32)
            self.render_gpu_positions = np.zeros((0, 3), dtype=np.float32)
            self.render_colors = np.zeros((0, 3), dtype=np.float32)
            self.render_tree = None
            self.cache_dirty = False
            return

        self.lod_levels = LODBuilder.build_lod_levels_for_count(alive_count)
        # If we're in an interactive drag (rotating) prefer a lightweight preview LOD
        if self.interactive and self.use_preview:
            chosen_alive_local = self.lod_levels["preview"]
        else:
            chosen_alive_local = self.lod_levels["full"] if not self.use_preview else LODBuilder.choose_level(
                self.lod_levels, alive_count, self.camera.dist
            )
        if self.model.deleted_mask.size and np.any(self.model.deleted_mask):
            alive_idx = self.model.alive_indices()
            self.render_indices = np.ascontiguousarray(alive_idx[chosen_alive_local], dtype=np.int32)
            self.render_positions = np.ascontiguousarray(self.model.positions[self.render_indices], dtype=np.float32)
            self.render_colors = np.ascontiguousarray(self.model.colors[self.render_indices], dtype=np.float32)
        else:
            self.render_indices = np.ascontiguousarray(chosen_alive_local, dtype=np.int32)
            if self.render_indices.size == self.model.count:
                self.render_positions = self.model.positions
                self.render_colors = self.model.colors
            else:
                self.render_positions = np.ascontiguousarray(self.model.positions[self.render_indices], dtype=np.float32)
                self.render_colors = np.ascontiguousarray(self.model.colors[self.render_indices], dtype=np.float32)
        self.render_gpu_positions = np.ascontiguousarray(
            self.render_positions - self.camera.center[None, :],
            dtype=np.float32,
        )

        # build spatial tree only when not in interactive mode (costly)
        if not self.interactive and cKDTree is not None and self.render_positions.shape[0] > 0:
            try:
                self.render_tree = cKDTree(self.render_positions)
            except Exception:
                self.render_tree = None
        else:
            self.render_tree = None

        self.cache_dirty = False
        self.gpu_dirty = True

    def upload_gpu_buffers(self) -> None:
        if self.cache_dirty:
            self.rebuild_render_cache()
        colors = self.render_colors.copy()
        if self.model.selected_count() > 0 and self.render_indices.size > 0:
            sel_mask = self.model.selected[self.render_indices]
            colors[sel_mask] = np.array([1.0, 0.2, 0.1], dtype=np.float32)
        colors = np.ascontiguousarray(colors, dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.render_gpu_positions.nbytes, self.render_gpu_positions, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.col_vbo)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        # Also update ModernGL buffers if available
        try:
            if self.mgl_ctx is not None and self.mgl_pos_vbo is not None and self.mgl_prog is not None:
                # ensure contiguous float32
                pos_bytes = np.ascontiguousarray(self.render_gpu_positions, dtype=np.float32).tobytes()
                col_bytes = np.ascontiguousarray(colors, dtype=np.float32).tobytes()
                # Recreate or write depending on buffer size
                try:
                    if getattr(self.mgl_pos_vbo, "size", len(pos_bytes)) != len(pos_bytes):
                        self.mgl_pos_vbo.release()
                        self.mgl_pos_vbo = self.mgl_ctx.buffer(pos_bytes)
                        self.mgl_vao = None
                    else:
                        self.mgl_pos_vbo.orphan(len(pos_bytes))
                except Exception:
                    pass
                try:
                    if getattr(self.mgl_col_vbo, "size", len(col_bytes)) != len(col_bytes):
                        self.mgl_col_vbo.release()
                        self.mgl_col_vbo = self.mgl_ctx.buffer(col_bytes)
                        self.mgl_vao = None
                    else:
                        self.mgl_col_vbo.orphan(len(col_bytes))
                except Exception:
                    pass
                try:
                    if getattr(self.mgl_pos_vbo, "size", len(pos_bytes)) == len(pos_bytes):
                        self.mgl_pos_vbo.write(pos_bytes)
                    if getattr(self.mgl_col_vbo, "size", len(col_bytes)) == len(col_bytes):
                        self.mgl_col_vbo.write(col_bytes)
                except Exception as e:
                    print(f"[GLViewer.upload_gpu_buffers] ModernGL buffer write failed: {e}")
                
                # Create VAO if not already created (after we have data in buffers)
                if self.mgl_vao is None and len(pos_bytes) > 0:
                    try:
                        self.mgl_vao = self.mgl_ctx.vertex_array(
                            self.mgl_prog,
                            [(self.mgl_pos_vbo, '3f', 'in_pos'), (self.mgl_col_vbo, '3f', 'in_col')],
                        )
                        print(f"[GLViewer.upload_gpu_buffers] ModernGL VAO created on first data upload: {self.mgl_vao}")
                    except Exception as e:
                        print(f"[GLViewer.upload_gpu_buffers] ModernGL VAO creation failed: {e}")
                        self.mgl_vao = None
        except Exception:
            pass
        self.gpu_dirty = False

    def reset_view(self) -> None:
        self.camera.yaw = 0.0
        self.camera.pitch = 0.0
        self.fit_camera_to_model()
        self.queue_draw()

    def fit_camera_to_model(self) -> None:
        if self.model.alive_count == 0:
            self.camera.center = np.zeros((3,), dtype=np.float32)
            self.camera.dist = 3.0
            return
        mn, mx = self.model.bounds()
        self.camera.center = ((mn + mx) * 0.5).astype(np.float32)
        extent = (mx - mn).astype(np.float32)
        radius = max(0.5, float(np.linalg.norm(extent)) * 0.5)
        self.scene_radius = radius
        self.camera.dist = max(2.0, radius * 3.0)

    def get_mvp_matrix(self) -> NDArray:
        """Compute Model-View-Projection matrix on CPU side."""
        w = max(1, self.get_allocated_width())
        h = max(1, self.get_allocated_height())
        aspect = w / h
        znear = max(0.001, min(self.scene_radius / 1000.0, self.camera.dist * 0.1))
        zfar = max(1000.0, self.camera.dist + self.scene_radius * 4.0)
        proj = create_projection_matrix(45.0, aspect, znear, zfar)
        camera = Camera(
            yaw=self.camera.yaw,
            pitch=self.camera.pitch,
            dist=self.camera.dist,
            center=np.zeros((3,), dtype=np.float32),
        )
        view = create_view_matrix(camera)
        mvp = proj @ view  # P * V * M (where M is identity for now)
        return mvp
    
    def get_transformed_positions(self) -> NDArray:
        """Get positions transformed by MVP matrix on CPU side."""
        if self.render_positions.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)
        
        # Get MVP matrix
        mvp = self.get_mvp_matrix()
        
        # Transform positions: convert to homogeneous coordinates, apply MVP, perspective divide
        ones = np.ones((self.render_positions.shape[0], 1), dtype=np.float32)
        positions_h = np.hstack([self.render_positions, ones])  # (N, 4)
        
        # Apply MVP transformation
        transformed = positions_h @ mvp.T  # (N, 4) @ (4, 4)^T = (N, 4)
        
        # Perspective division
        transformed_xyz = transformed[:, :3] / np.maximum(transformed[:, 3:4], 1e-6)
        
        return transformed_xyz.astype(np.float32)

    def render_with_shader(self) -> None:
        """Render points using modern OpenGL shader pipeline (simplified version)."""
        if not self.shader_program or self.render_positions.shape[0] == 0:
            return
        
        try:
            # For now, skip shader rendering and fall back to simple point drawing
            # This avoids context issues with glUniform calls
            print("[render_with_shader] Shader pipeline not fully implemented, skipping")
            return
        except Exception as e:
            print(f"[render_with_shader] Exception: {e}")

    def paintGL(self) -> None:
        self.make_current()
        
        w = self.get_allocated_width()
        h = self.get_allocated_height()
        glViewport(0, 0, w, h)
        glClearColor(0.1, 0.1, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if self.model.alive_count == 0:
            return

        if self.gpu_dirty or self.cache_dirty:
            self.upload_gpu_buffers()

        try:
            if self.mgl_ctx is not None and self.mgl_vao is not None:
                try:
                    mvp = self.get_mvp_matrix()
                    if 'u_mvp' in self.mgl_prog:
                        self.mgl_prog['u_mvp'].write(np.ascontiguousarray(mvp.T, dtype=np.float32).tobytes())
                    try:
                        if 'u_point_size' in self.mgl_prog:
                            self.mgl_prog['u_point_size'].value = float(self.point_size)
                    except Exception:
                        pass

                    self.mgl_vao.render(mode=moderngl.POINTS)
                    return
                except Exception as e:
                    print(f"[GLViewer.paintGL] ModernGL render failed: {e}")
                    import traceback
                    traceback.print_exc()

            # Fallback: attempt to use existing PyOpenGL shader path
            if not self.shader_program:
                return
            
            print("[GLViewer.paintGL] Using PyOpenGL shader fallback")
            glUseProgram(self.shader_program)
            point_size_loc = glGetUniformLocation(self.shader_program, "u_point_size")
            glUniform1f(point_size_loc, float(self.point_size))
            # Draw with existing VAO if possible
            if self.vao and self.vao > 0:
                glBindVertexArray(self.vao)
                glDrawArrays(GL_POINTS, 0, int(self.render_positions.shape[0]))
                glBindVertexArray(0)
            else:
                glDrawArrays(GL_POINTS, 0, int(self.render_positions.shape[0]))
            glUseProgram(0)
            print("[GLViewer.paintGL] Fallback render complete")

            if self.dragging_box and self.box_start is not None and self.box_end is not None:
                self.draw_overlay_box()
        except Exception as e:
            print(f"[GLViewer.paintGL] Rendering failed: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.status_callback(f"警告: GL 渲染错误: {str(e)}")
            except Exception:
                pass
            return

    def draw_overlay_box(self) -> None:
        """Draw selection box overlay (disabled in Core Profile GL - needs shader implementation)."""
        # TODO: Implement selection box drawing with shaders
        # For now, skip this feature as it requires glBegin/glEnd which are not available in Core Profile
        pass

    # --- input handling ---
    def on_button_press(self, widget, event) -> bool:
        self.last_pos = (event.x, event.y)
        if event.button == 1:
            if self.selection_mode == "point":
                self.pick_point(event.x, event.y)
            elif self.selection_mode == "box":
                self.dragging_box = True
                self.box_start = (event.x, event.y)
                self.box_end = (event.x, event.y)
            elif self.selection_mode == "brush":
                self.brush_active = True
                self.brush_select(event.x, event.y, additive=False)
        elif event.button == 3:
            # start interactive camera drag mode: use preview LOD to keep interaction smooth
            self.interactive = True
            if self.use_preview and self.model.alive_count > 150_000:
                self.cache_dirty = True
                self.gpu_dirty = True
        return True

    def on_motion_notify(self, widget, event) -> bool:
        if self.selection_mode == "box" and self.dragging_box:
            self.box_end = (event.x, event.y)
            self.queue_draw()
            return True
        if self.selection_mode == "brush" and self.brush_active:
            self.brush_select(event.x, event.y, additive=True)
            return True
        if self.last_pos is None:
            return False
        state = event.get_state()
        # check right button held
        # EventMotion doesn't have 'button'; check modifier state for right-button
        if state & Gdk.ModifierType.BUTTON3_MASK:
            dx = event.x - self.last_pos[0]
            dy = event.y - self.last_pos[1]
            self.camera.yaw += dx * 0.01
            self.camera.pitch += dy * 0.01
            self.camera.pitch = max(-1.5, min(1.5, self.camera.pitch))
            # rotation doesn't change model LOD selection (dist unchanged);
            # avoid expensive cache rebuild during drag — only redraw the view
            self.queue_draw()
        self.last_pos = (event.x, event.y)
        return True

    def on_button_release(self, widget, event) -> bool:
        if self.selection_mode == "box" and self.dragging_box and self.box_start is not None and self.box_end is not None:
            x1, y1 = self.box_start
            x2, y2 = self.box_end
            self.box_select(x1, y1, x2, y2)
        self.dragging_box = False
        self.brush_active = False
        self.box_start = None
        self.box_end = None
        # end interactive camera drag mode
        if event.button == 3:
            self.interactive = False
            # request a full rebuild (may re-evaluate LOD based on camera)
            self.mark_model_dirty()
        self.last_pos = None
        return True

    def on_scroll(self, widget, event) -> bool:
        delta = 0.0
        try:
            ok, _, dy = event.get_scroll_deltas()
            if ok:
                delta = float(dy)
        except Exception:
            pass
        if delta == 0.0:
            if event.direction == Gdk.ScrollDirection.UP:
                delta = -1.0
            elif event.direction == Gdk.ScrollDirection.DOWN:
                delta = 1.0
        if delta == 0.0:
            return True

        self.camera.dist *= float(1.15 ** delta)
        min_dist = max(0.001, self.scene_radius / 1000.0)
        max_dist = max(100000.0, self.scene_radius * 50.0)
        self.camera.dist = max(min_dist, min(max_dist, self.camera.dist))
        self.queue_draw()
        return True

    # --- selection / projection helpers ---
    def project_render_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.render_positions.shape[0] == 0:
            z = np.zeros((0,), dtype=np.float32)
            return z, z, z
        pts = self.render_positions - self.camera.center[None, :]
        cp = math.cos(self.camera.pitch)
        sp = math.sin(self.camera.pitch)
        cy = math.cos(self.camera.yaw)
        sy = math.sin(self.camera.yaw)

        x1 = cy * pts[:, 0] + sy * pts[:, 2]
        y1 = pts[:, 1]
        z1 = -sy * pts[:, 0] + cy * pts[:, 2]
        x2 = x1
        y2 = cp * y1 - sp * z1
        z2 = sp * y1 + cp * z1 - self.camera.dist

        valid = z2 < -1e-5
        aspect = max(1.0, self.get_allocated_width() / max(1, self.get_allocated_height()))
        f = 1.0 / math.tan(math.radians(45.0) * 0.5)
        ndc_x = np.zeros_like(x2)
        ndc_y = np.zeros_like(y2)
        depth = np.full_like(z2, np.inf)
        ndc_x[valid] = (x2[valid] * f / aspect) / (-z2[valid])
        ndc_y[valid] = (y2[valid] * f) / (-z2[valid])
        depth[valid] = -z2[valid]
        sx = (ndc_x * 0.5 + 0.5) * self.get_allocated_width()
        sy_scr = (1.0 - (ndc_y * 0.5 + 0.5)) * self.get_allocated_height()
        return sx, sy_scr, depth

    def pick_point(self, x: float, y: float) -> None:
        if self.cache_dirty:
            self.rebuild_render_cache()
        sx, sy, depth = self.project_render_points()
        if sx.size == 0:
            return
        d = np.sqrt((sx - x) ** 2 + (sy - y) ** 2)
        threshold = max(8.0, self.point_size + 4.0)
        candidates = np.where(d < threshold)[0]
        if candidates.size == 0:
            self.model.clear_selection()
            self.status_callback("未命中点")
        else:
            best_render = int(candidates[np.argmin(depth[candidates])])
            raw_idx = int(self.render_indices[best_render])
            self.model.select_raw_index(raw_idx)
            p = self.model.positions[raw_idx]
            self.status_callback(f"已选中点 #{raw_idx}: ({p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f})")
        self.gpu_dirty = True
        self.queue_draw()

    def box_select(self, x1: float, y1: float, x2: float, y2: float) -> None:
        if self.cache_dirty:
            self.rebuild_render_cache()
        sx, sy, _ = self.project_render_points()
        idx = box_select_indices(sx, sy, x1, y1, x2, y2)
        raw_idx = self.render_indices[idx] if idx.size > 0 else np.zeros((0,), dtype=np.int32)
        count = self.model.select_raw_indices(raw_idx, additive=False)
        self.status_callback(f"框选点数: {count}")
        self.gpu_dirty = True
        self.queue_draw()

    def brush_select(self, x: float, y: float, additive: bool = True) -> None:
        if self.cache_dirty:
            self.rebuild_render_cache()
        sx, sy, _ = self.project_render_points()
        idx = brush_select_indices(sx, sy, x, y, self.brush_radius)
        raw_idx = self.render_indices[idx] if idx.size > 0 else np.zeros((0,), dtype=np.int32)
        count = self.model.select_raw_indices(raw_idx, additive=additive)
        self.status_callback(f"刷选新增点数: {count}")
        self.gpu_dirty = True
        self.queue_draw()
