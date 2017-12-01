import os
import sys
from ctypes import c_double
from multiprocessing import Value
from uvc import get_time_monotonic
from multiprocessing import freeze_support

from OpenGL.GL import *
from pyglui import ui
from pyglui.cygl.utils import Named_Texture
from pyglui.cygl.utils import init

base_dir = os.path.abspath(__file__).rsplit('src', 1)[0]
sys.path.append(os.path.join(base_dir, 'src', 'modules'))

import glfw
import platform

import gl_utils
from methods import normalize, denormalize, timer
from plugin import Plugin_List
from plugins.surface_tracker import Surface_Tracker
from video_capture import UVC_Source, UVC_Manager, Default_Manager, Default_Source

user_dir = os.path.expanduser(os.path.join('~', 'configuracoes_controlador_ssc32'))
if not os.path.exists(user_dir):
    os.makedirs(user_dir)

timebase = Value(c_double, 0)


class Global_Container(object):
    pass


def main():
    # UI Platform tweaks
    if platform.system() == 'Linux':
        scroll_factor = 10.0
        window_position_default = (30, 30)
    elif platform.system() == 'Windows':
        scroll_factor = 10.0
        window_position_default = (8, 31)
    else:
        scroll_factor = 1.0
        window_position_default = (0, 0)

    icon_bar_width = 50
    window_size = None
    camera_render_size = None
    hdpi_factor = 1.0

    g_pool = Global_Container()
    g_pool.app = 'capture'
    g_pool.process = 'main'
    g_pool.timebase = timebase
    g_pool.user_dir = user_dir

    def get_timestamp():
        return get_time_monotonic() - timebase.value

    g_pool.get_timestamp = get_timestamp
    g_pool.get_now = get_time_monotonic

    default_capture_settings = {
        'frame_rate': 30,
        'frame_size': (1280, 720),
        'preferred_names': ['USB2.0 PC CAMERA']
    }

    default_plugins = [("Default_Source", default_capture_settings),
                       ('Default_Manager', {}),
                       ('Surface_Tracker', {})]

    plugins = [Default_Source, Default_Manager, Surface_Tracker]
    g_pool.plugin_by_name = {p.__name__: p for p in plugins}

    g_pool.capture = None
    g_pool.surface_tracker = None

    # Callbacks
    def on_resize(janela, largura, altura):
        nonlocal camera_render_size
        altura, largura = max(altura, 1), max(largura, 1)

        fator_escala = float(glfw.get_framebuffer_size(janela)[0] / glfw.get_window_size(janela)[0])
        largura, altura = int(largura * fator_escala), int(altura * fator_escala)

        camera_render_size = largura - int(icon_bar_width * g_pool.gui.scale), altura
        if camera_render_size[0] < 0:
            camera_render_size = (0, camera_render_size[1])

        glfw.make_context_current(janela)
        gl_utils.adjust_gl_view(largura, altura)

        g_pool.gui.update_window(largura, altura)
        g_pool.gui.collect_menus()

        for p in g_pool.plugins:
            p.on_window_resize(janela, *camera_render_size)

    def on_key(janela, tecla, codigo, acao, mods):
        g_pool.gui.update_key(tecla, codigo, acao, mods)

    def on_char(janela, char):
        g_pool.gui.update_char(char)

    def on_button(janela, botao, acao, mods):
        g_pool.gui.update_button(botao, acao, mods)

    def on_pos(janela, x, y):
        x, y = x * hdpi_factor, y * hdpi_factor
        g_pool.gui.update_mouse(x, y)

        pos = x, y
        pos = normalize(pos, camera_render_size)
        pos = denormalize(pos, g_pool.capture.frame_size)
        for p in g_pool.plugins:
            p.on_pos(pos)

    def on_scroll(janela, x, y):
        g_pool.gui.update_scroll(x, y * scroll_factor)

    if not glfw.init():
        return

    janela = glfw.create_window(1360, 720, "Controle de braço robótico", None, None)

    glfw.set_window_pos(janela, *window_position_default)

    if not janela:
        glfw.terminate()
        return

    glfw.make_context_current(janela)
    glfw.set_window_size_callback(janela, on_resize)
    glfw.set_mouse_button_callback(janela, on_button)
    glfw.set_cursor_pos_callback(janela, on_pos)
    glfw.set_key_callback(janela, on_key)
    glfw.set_char_callback(janela, on_char)
    glfw.set_scroll_callback(janela, on_scroll)

    init()
    gl_utils.basic_gl_setup()
    g_pool.image_tex = Named_Texture()
    g_pool.main_window = janela

    g_pool.gui = ui.UI()
    g_pool.gui_user_scale = 1.0
    g_pool.gui.scale = 1.0
    g_pool.menubar = ui.Scrolling_Menu("Settings", pos=(-400, 0), size=(-icon_bar_width, 0), header_pos='left')
    g_pool.iconbar = ui.Scrolling_Menu("Icons", pos=(-icon_bar_width, 0), size=(0, 0), header_pos='hidden')
    g_pool.quickbar = ui.Stretching_Menu('Quick Bar', (0, 100), (120, -100))
    g_pool.gui.append(g_pool.menubar)
    g_pool.gui.append(g_pool.iconbar)
    g_pool.gui.append(g_pool.quickbar)

    def set_scale(new_scale):
        g_pool.gui_user_scale = new_scale
        window_size = camera_render_size[0] + \
                      int(icon_bar_width * g_pool.gui_user_scale * hdpi_factor), \
                      glfw.get_framebuffer_size(janela)[1]
        glfw.set_window_size(janela, *window_size)

    def toggle_general_settings(collapsed):
        g_pool.menubar.collapsed = collapsed
        for m in g_pool.menubar.elements:
            m.collapsed = True
        general_settings.collapsed = collapsed

    general_settings = ui.Growing_Menu('Geral', header_pos='headline')

    def set_window_size():
        f_width, f_height = g_pool.capture.frame_size

        f_width += int(icon_bar_width * g_pool.gui.scale)
        glfw.set_window_size(janela, f_width, f_height)

    general_settings.append(ui.Button('Resetar tamanho da janela', set_window_size))
    general_settings.append(ui.Selector('gui_user_scale', g_pool, setter=set_scale, selection=[.6, .8, 1., 1.2, 1.4],
                                        label='Tamanho da interface'))

    g_pool.menubar.append(general_settings)
    icon = ui.Icon('collapsed', general_settings, label=chr(0xe8b8), on_val=False, off_val=True,
                   setter=toggle_general_settings, label_font='pupil_icons')
    icon.tooltip = 'Configurações gerais'
    g_pool.iconbar.append(icon)

    g_pool.plugins = Plugin_List(g_pool, default_plugins)

    # Garante que o tamanho da janela seja ajustado
    on_resize(janela, *glfw.get_window_size(janela))
    set_window_size()

    # create a timer to control window update frequency
    window_update_timer = timer(1 / 60)

    def window_should_update():
        return next(window_update_timer)

    while not glfw.window_should_close(janela):
        glClearColor(0.2, 0.2, 0.2, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        events = {}

        for p in g_pool.plugins:
            p.recent_events(events)

        g_pool.plugins.clean()

        for p in g_pool.plugins:
            p.gl_display()

        glfw.make_context_current(janela)
        # render visual feedback from loaded plugins
        if window_should_update() and gl_utils.is_window_visible(janela):

            glViewport(0, 0, *glfw.get_window_size(janela))
            unused_elements = g_pool.gui.update()

            for button, action, mods in unused_elements.buttons:
                pos = glfw.get_cursor_pos(janela)
                pos = normalize(pos, camera_render_size)
                # Position in img pixels
                pos = denormalize(pos, g_pool.capture.frame_size)

                for p in g_pool.plugins:
                    p.on_click(pos, button, action)

            glfw.swap_buffers(janela)

        glfw.poll_events()

    # de-init all running plugins
    for p in g_pool.plugins:
        p.alive = False
    g_pool.plugins.clean()

    g_pool.gui.terminate()
    glfw.destroy_window(janela)
    glfw.terminate()


if __name__ == '__main__':
    freeze_support()
    main()
