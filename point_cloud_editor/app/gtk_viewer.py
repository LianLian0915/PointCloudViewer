from __future__ import annotations

import math
import numpy as np
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram
import ctypes
import moderngl

from app.camera import Camera
from app.lod import LODBuilder
from app.model import PointCloudModel
from app.selection import box_select_indices, brush_select_indices

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None


class GLViewer(Gtk.GLArea):
    def __init__(self, model: PointCloudModel, status_callback) -> None:
        super().__init__()
        self.set_has_depth_buffer(True)
        self.set_has_stencil_buffer(False)
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
        self.point_size = 3.0
        self.last_pos = None
        self.set_size_request(640, 480)

        self.lod_levels: dict[str, np.ndarray] = {
            "full": np.zeros((0,), dtype=np.int32),
            "medium": np.zeros((0,), dtype=np.int32),
            "preview": np.zeros((0,), dtype=np.int32),
        }
        self.render_indices = np.zeros((0,), dtype=np.int32)
        self.render_positions = np.zeros((0, 3), dtype=np.float32)
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
        self.u_mvp = None
        self.u_point_size = None
    # moderngl context and resources
    self.mgl_ctx = None
    self.mgl_prog = None
    self.mgl_vbo_pos = None
    self.mgl_vbo_col = None
    self.mgl_vao = None

    # --- GL lifecycle ---
    def on_realize(self, widget) -> None:
        # GL context available after realize
        self.make_current()
        # nothing heavy here; actual GL init in first render

    def on_unrealize(self, widget) -> None:
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
        # ensure context is current for PyOpenGL
        try:
            self.make_current()
        except Exception:
            pass

        if not self._gl_inited:
            self.initializeGL()
            self._gl_inited = True

        # paint equivalent
        self.paintGL()
        return True

    def initializeGL(self) -> None:
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
        version = glGetString(GL_VERSION)
        renderer = glGetString(GL_RENDERER)
        v = version.decode("utf-8", errors="ignore") if version else "unknown"
        r = renderer.decode("utf-8", errors="ignore") if renderer else "unknown"
        try:
            self.status_callback(f"OpenGL: {v} | Renderer: {r}")
        except Exception:
            pass
        print(f"OpenGL: {v} | Renderer: {r}")
        # Try to create a simple shader program for modern GL contexts
        vert_src = '''#version 330
        layout(location = 0) in vec3 in_pos;
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
        # Try moderngl first (more robust with Gtk.GLArea contexts)
        try:
            try:
                self.mgl_ctx = moderngl.create_context()
            except Exception:
                try:
                    # some versions accept require=False
                    self.mgl_ctx = moderngl.create_context(require=False)
                except Exception as e:
                    print("moderngl 创建上下文失败:", e)
                    self.mgl_ctx = None

            if self.mgl_ctx is not None:
                try:
                    m_vert = """#version 330
                    in vec3 in_pos;
                    in vec3 in_col;
                    uniform mat4 u_mvp;
                    uniform float u_point_size;
                    out vec3 v_col;
                    void main() {
                        gl_Position = u_mvp * vec4(in_pos, 1.0);
                        v_col = in_col;
                        gl_PointSize = u_point_size;
                    }
                    """
                    m_frag = """#version 330
                    in vec3 v_col;
                    out vec4 out_col;
                    void main() {
                        out_col = vec4(v_col, 1.0);
                    }
                    """
                    self.mgl_prog = self.mgl_ctx.program(vertex_shader=m_vert, fragment_shader=m_frag)
                    print("moderngl: program 创建成功")
                except Exception as e:
                    print("moderngl program 创建失败:", e)
                    self.mgl_prog = None
            else:
                self.mgl_prog = None
        except Exception:
            self.mgl_ctx = None
            self.mgl_prog = None

        try:
            if self.mgl_prog is None:
                try:
                    vs = compileShader(vert_src, GL_VERTEX_SHADER)
                    fs = compileShader(frag_src, GL_FRAGMENT_SHADER)
                    self.shader_program = compileProgram(vs, fs)
                except Exception as e:
                    # log shader compile/link errors for diagnosis
                    print("着色器(330) 编译/链接失败:", e)
                    self.shader_program = None
            # create VAO
            try:
                self.vao = glGenVertexArrays(1)
                glBindVertexArray(self.vao)
                # bind buffers and enable attribs (no data yet)
                glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                glBindBuffer(GL_ARRAY_BUFFER, self.col_vbo)
                glEnableVertexAttribArray(1)
                glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glBindVertexArray(0)
            except Exception:
                # VAO may not be available; continue without it
                import traceback
                print("VAO creation failed:")
                traceback.print_exc()
                self.vao = None
            # get uniform locations (PyOpenGL path)
            if self.shader_program is not None:
                self.u_mvp = glGetUniformLocation(self.shader_program, b"u_mvp")
                self.u_point_size = glGetUniformLocation(self.shader_program, b"u_point_size")
            else:
                self.u_mvp = None
                self.u_point_size = None
            # enable program point size if supported
            try:
                glEnable(GL_PROGRAM_POINT_SIZE)
            except Exception:
                pass
            # report shader usage (only if compiled)
            if self.shader_program is not None:
                try:
                    self.status_callback("着色器: 使用 GLSL 330")
                except Exception:
                    pass
                print("着色器: 使用 GLSL 330")
                print(f"VAO={self.vao} pos_vbo={self.pos_vbo} col_vbo={self.col_vbo}")
        except Exception:
            # shader creation failed; we'll fall back to legacy path or clear-only
            self.shader_program = None
            # try an alternative GLSL version without layout qualifiers (e.g. 130)
            try:
                vert130 = '''#version 130
                in vec3 in_pos;
                in vec3 in_col;
                uniform mat4 u_mvp;
                uniform float u_point_size;
                out vec3 v_col;
                void main() {
                    gl_Position = u_mvp * vec4(in_pos, 1.0);
                    v_col = in_col;
                    gl_PointSize = u_point_size;
                }
                '''
                frag130 = '''#version 130
                in vec3 v_col;
                out vec4 out_col;
                void main() {
                    out_col = vec4(v_col, 1.0);
                }
                '''
                try:
                    vs130 = compileShader(vert130, GL_VERTEX_SHADER)
                    fs130 = compileShader(frag130, GL_FRAGMENT_SHADER)
                    prog = glCreateProgram()
                    glAttachShader(prog, vs130)
                    glAttachShader(prog, fs130)
                except Exception as e:
                    print("着色器(130) 编译失败:", e)
                    raise
                # bind attrib locations for older GLSL without layout qualifiers
                try:
                    glBindAttribLocation(prog, 0, b"in_pos")
                    glBindAttribLocation(prog, 1, b"in_col")
                except Exception:
                    pass
                glLinkProgram(prog)
                # check link status
                try:
                    status = glGetProgramiv(prog, GL_LINK_STATUS)
                except Exception:
                    status = 0
                if status == GL_TRUE:
                    self.shader_program = prog
                    # get uniform locations
                    self.u_mvp = glGetUniformLocation(self.shader_program, b"u_mvp")
                    self.u_point_size = glGetUniformLocation(self.shader_program, b"u_point_size")
                    try:
                        glEnable(GL_PROGRAM_POINT_SIZE)
                    except Exception:
                        pass
                    try:
                        self.status_callback("着色器: 使用 GLSL 130 兼容模式")
                    except Exception:
                        pass
                    print("着色器: 使用 GLSL 130 兼容模式")
                else:
                    # leave shader_program as None
                    self.shader_program = None
            except Exception:
                self.shader_program = None

    # --- GL data & drawing ---
    def mark_model_dirty(self) -> None:
        self.cache_dirty = True
        self.gpu_dirty = True
        self.queue_draw()

    def set_selection_mode(self, mode: str) -> None:
        self.selection_mode = mode

    def set_preview_enabled(self, enabled: bool) -> None:
        self.use_preview = enabled
        self.mark_model_dirty()

    def rebuild_render_cache(self) -> None:
        alive_idx = self.model.alive_indices()
        if alive_idx.size == 0:
            self.render_indices = np.zeros((0,), dtype=np.int32)
            self.render_positions = np.zeros((0, 3), dtype=np.float32)
            self.render_colors = np.zeros((0, 3), dtype=np.float32)
            self.render_tree = None
            self.cache_dirty = False
            return

        alive_points = self.model.positions[alive_idx]
        self.lod_levels = LODBuilder.build_lod_levels(alive_points)
        chosen_alive_local = self.lod_levels["full"] if not self.use_preview else LODBuilder.choose_level(
            self.lod_levels, alive_points.shape[0], self.camera.dist
        )
        self.render_indices = np.ascontiguousarray(alive_idx[chosen_alive_local], dtype=np.int32)
        self.render_positions = np.ascontiguousarray(self.model.positions[self.render_indices], dtype=np.float32)
        self.render_colors = np.ascontiguousarray(self.model.colors[self.render_indices], dtype=np.float32)

        if cKDTree is not None and self.render_positions.shape[0] > 0:
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
        glBufferData(GL_ARRAY_BUFFER, self.render_positions.nbytes, self.render_positions, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.col_vbo)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.gpu_dirty = False
        try:
            self.status_callback(f"上传到 GPU: {self.render_positions.shape[0]} 个点")
        except Exception:
            pass
        print(f"上传到 GPU: {self.render_positions.shape[0]} 个点")

        # also push to moderngl buffers if available
        if self.mgl_ctx is not None and self.mgl_prog is not None:
            try:
                # release previous buffers
                if self.mgl_vbo_pos is not None:
                    try:
                        self.mgl_vbo_pos.release()
                    except Exception:
                        pass
                if self.mgl_vbo_col is not None:
                    try:
                        self.mgl_vbo_col.release()
                    except Exception:
                        pass
                # create new buffers
                self.mgl_vbo_pos = self.mgl_ctx.buffer(self.render_positions.tobytes())
                self.mgl_vbo_col = self.mgl_ctx.buffer(colors.tobytes())
                # release previous vao
                if self.mgl_vao is not None:
                    try:
                        self.mgl_vao.release()
                    except Exception:
                        pass
                # create vertex array object binding program inputs to buffers
                # program attributes names must match shader: in_pos, in_col
                self.mgl_vao = self.mgl_ctx.vertex_array(
                    self.mgl_prog,
                    [
                        (self.mgl_vbo_pos, '3f', 'in_pos'),
                        (self.mgl_vbo_col, '3f', 'in_col'),
                    ],
                )
                print(f"moderngl: 上传到 GPU buffers (n={self.render_positions.shape[0]})")
            except Exception as e:
                print("moderngl 上传失败:", e)

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
        self.camera.dist = max(2.0, radius * 3.0)

    def setup_projection(self) -> None:
        w = max(1, self.get_allocated_width())
        h = max(1, self.get_allocated_height())
        aspect = w / h
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        znear = 0.01
        zfar = 100000.0
        fovy = 45.0
        top = znear * math.tan(math.radians(fovy) * 0.5)
        bottom = -top
        right = top * aspect
        left = -right
        glFrustum(left, right, bottom, top, znear, zfar)

    def setup_modelview(self) -> None:
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -float(self.camera.dist))
        glRotatef(math.degrees(self.camera.pitch), 1.0, 0.0, 0.0)
        glRotatef(math.degrees(self.camera.yaw), 0.0, 1.0, 0.0)
        glTranslatef(-float(self.camera.center[0]), -float(self.camera.center[1]), -float(self.camera.center[2]))

    def paintGL(self) -> None:
        w = self.get_allocated_width()
        h = self.get_allocated_height()
        glViewport(0, 0, w, h)
        glClearColor(0.1, 0.1, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if self.model.alive_count == 0:
            return
        # Many systems provide a core-profile GL context where legacy fixed-function
        # pipeline calls (glMatrixMode, glVertexPointer, etc.) are invalid. Wrap
        # the original rendering sequence and fall back to a simple clear if
        # the calls are not supported yet. Later we should implement modern
        # shader-based rendering for full functionality.
        if self.gpu_dirty or self.cache_dirty:
            self.upload_gpu_buffers()

        try:
            # compute MVP matrix
            w = max(1, self.get_allocated_width())
            h = max(1, self.get_allocated_height())
            aspect = w / h
            znear = 0.01
            zfar = 100000.0
            fovy = 45.0
            f = 1.0 / math.tan(math.radians(fovy) * 0.5)
            # perspective matrix
            proj = np.zeros((4, 4), dtype=np.float32)
            proj[0, 0] = f / aspect
            proj[1, 1] = f
            proj[2, 2] = (zfar + znear) / (znear - zfar)
            proj[2, 3] = (2 * zfar * znear) / (znear - zfar)
            proj[3, 2] = -1.0

            # modelview: translate and rotate similar to previous fixed pipeline
            mv = np.eye(4, dtype=np.float32)
            # translate -dist on z
            mv = mv @ np.array([[1,0,0,0],[0,1,0,0],[0,0,1,-self.camera.dist],[0,0,0,1]], dtype=np.float32)
            # rotate pitch around X
            cp = math.cos(self.camera.pitch)
            sp = math.sin(self.camera.pitch)
            rx = np.array([[1,0,0,0],[0,cp,-sp,0],[0,sp,cp,0],[0,0,0,1]], dtype=np.float32)
            # rotate yaw around Y
            cy = math.cos(self.camera.yaw)
            sy = math.sin(self.camera.yaw)
            ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=np.float32)
            mv = mv @ rx @ ry
            # translate by -center
            trans = np.eye(4, dtype=np.float32)
            trans[0,3] = -float(self.camera.center[0])
            trans[1,3] = -float(self.camera.center[1])
            trans[2,3] = -float(self.camera.center[2])
            mv = mv @ trans

            mvp = proj @ mv

            try:
                self.status_callback(f"绘制: n={self.render_positions.shape[0]} center={self.camera.center.tolist()} dist={self.camera.dist:.3f}")
            except Exception:
                pass
            print(f"绘制: n={self.render_positions.shape[0]} center={self.camera.center.tolist()} dist={self.camera.dist:.3f}")

            # Prefer moderngl rendering when available (more robust than PyOpenGL context handling)
            if self.mgl_prog is not None:
                try:
                    if self.gpu_dirty:
                        self.upload_gpu_buffers()
                    data = np.ascontiguousarray(mvp, dtype=np.float32).tobytes()
                    try:
                        # write uniform (moderngl Program exposes uniforms by name)
                        self.mgl_prog["u_mvp"].write(data)
                    except Exception:
                        # older moderngl may use different access; try attribute-style
                        try:
                            self.mgl_prog["u_mvp"].value = tuple(np.ascontiguousarray(mvp, dtype=np.float32).flatten())
                        except Exception:
                            pass
                    try:
                        if "u_point_size" in self.mgl_prog:
                            self.mgl_prog["u_point_size"].value = float(self.point_size)
                    except Exception:
                        pass
                    if self.mgl_vao is not None:
                        self.mgl_vao.render(mode=moderngl.POINTS)
                    else:
                        print("moderngl: 没有可用的 VAO")
                except Exception as e:
                    print("moderngl 渲染失败:", e)
                # draw overlay if needed (overlay uses legacy GL calls; try but ignore failures)
                if self.dragging_box and self.box_start is not None and self.box_end is not None:
                    try:
                        self.draw_overlay_box()
                    except Exception:
                        pass
                return

            # Prefer shader-based rendering when available
            if self.shader_program is not None:
                try:
                    # ensure buffers uploaded
                    if self.gpu_dirty:
                        self.upload_gpu_buffers()

                    # set uniforms (MVP computed above)
                    data = np.ascontiguousarray(mvp, dtype=np.float32).flatten()
                    glUseProgram(self.shader_program)
                    if self.u_mvp is not None:
                        glUniformMatrix4fv(self.u_mvp, 1, GL_FALSE, data)
                    if self.u_point_size is not None:
                        glUniform1f(self.u_point_size, float(self.point_size))

                    # bind and draw
                    if self.vao:
                        glBindVertexArray(self.vao)
                        glDrawArrays(GL_POINTS, 0, int(self.render_positions.shape[0]))
                        glBindVertexArray(0)
                    else:
                        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
                        glEnableVertexAttribArray(0)
                        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                        glBindBuffer(GL_ARRAY_BUFFER, self.col_vbo)
                        glEnableVertexAttribArray(1)
                        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
                        glDrawArrays(GL_POINTS, 0, int(self.render_positions.shape[0]))
                        glBindBuffer(GL_ARRAY_BUFFER, 0)

                    glUseProgram(0)
                except Exception:
                    try:
                        self.status_callback("警告: 着色器渲染失败，尝试固定管线")
                    except Exception:
                        pass
                    import traceback
                    import sys

                    exc = sys.exc_info()
                    print("警告: 着色器渲染失败，尝试固定管线", exc[1])
                    traceback.print_exc()
                    try:
                        err = glGetError()
                        print("glGetError:", err)
                    except Exception:
                        pass
                    # fall back to fixed-function below
                # If shader succeeded we drawn and can draw overlay (skip legacy matrices)
                if self.dragging_box and self.box_start is not None and self.box_end is not None:
                    # overlay drawing relies on legacy matrix calls; try but ignore failures
                    try:
                        self.draw_overlay_box()
                    except Exception:
                        pass
                # shader path done
                return

            # Fixed-function fallback (may not be supported on core-profile contexts)
            # setup projection/modelview using legacy calls for fixed-pipeline rendering
            self.setup_projection()
            self.setup_modelview()
            try:
                glPointSize(float(self.point_size))
                glEnableClientState(GL_VERTEX_ARRAY)
                glEnableClientState(GL_COLOR_ARRAY)
                glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
                glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
                glBindBuffer(GL_ARRAY_BUFFER, self.col_vbo)
                glColorPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
                glDrawArrays(GL_POINTS, 0, int(self.render_positions.shape[0]))
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                glDisableClientState(GL_COLOR_ARRAY)
                glDisableClientState(GL_VERTEX_ARRAY)
            except Exception:
                # if even fallback fails, warn and return
                try:
                    self.status_callback("警告: 固定管线渲染不可用，无法显示点云")
                except Exception:
                    pass
                print("警告: 固定管线渲染不可用，无法显示点云")

            if self.dragging_box and self.box_start is not None and self.box_end is not None:
                self.draw_overlay_box()
        except Exception:
            # Fixed-function not available; fall back to a basic clear-only view
            try:
                self.status_callback("警告: 当前 GL 上下文不支持固定管线渲染，已启用降级视图")
            except Exception:
                pass
            print("警告: 当前 GL 上下文不支持固定管线渲染，已启用降级视图")
            return

    def draw_overlay_box(self) -> None:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.get_allocated_width(), self.get_allocated_height(), 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glColor3f(0.2, 0.8, 1.0)
        x1, y1 = self.box_start
        x2, y2 = self.box_end
        glBegin(GL_LINE_LOOP)
        glVertex2f(x1, y1)
        glVertex2f(x2, y1)
        glVertex2f(x2, y2)
        glVertex2f(x1, y2)
        glEnd()
        glEnable(GL_DEPTH_TEST)

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
            self.cache_dirty = True
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
        self.last_pos = None
        return True

    def on_scroll(self, widget, event) -> bool:
        # event.delta_y may be present, but use event.direction for discrete scroll
        if hasattr(event, "delta_y"):
            delta = event.delta_y
        else:
            # fallback: use event.direction
            delta = -1.0 if event.direction == Gdk.ScrollDirection.UP else 1.0
        self.camera.dist += -delta * 0.2
        self.camera.dist = max(0.2, min(100000.0, self.camera.dist))
        self.cache_dirty = True
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
