# -*- coding: utf-8 -*-
import logging
import cv2

from _glfw import *
from OpenGL.GL import *
from pyglui import ui
from pyglui.cygl.utils import init
from pyglui.cygl.utils import RGBA
from pyglui import cygl

logger = logging.getLogger(__name__)

width, height = (1280, 720)


def inicializar():
    glEnable(GL_POINT_SPRITE)
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)
    glClearColor(1., 1., 1., 1.)
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_POINT_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_POLYGON_SMOOTH)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)


def basic_gl_setup():
    glEnable(GL_POINT_SPRITE)
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE) # overwrite pointsize
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_BLEND)
    glClearColor(.8,.8,.8,1.)
    glEnable(GL_LINE_SMOOTH)
    # glEnable(GL_POINT_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_POLYGON_SMOOTH)
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)


def ajustar_janela(w, h, janela):
    """
    Ajustar matriz de projeção para janela
    """

    # Ajusta o tamanho da janela destinada ao desenho
    glViewport(0, 0, int(w), int(h))
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, w, h, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def adjust_gl_view(w, h, window):
    """
    adjust view onto our scene.
    """
    glViewport(0, 0, int(w), int(h))
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, w, h, 0, -1, 1)
    # glOrtho(0, 1, 0, 1, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def main():
    global quit
    quit = False

    def on_resize(janela, w, h):
        h = max(h, 1)
        w = max(w, 1)

        hdpi_factor = glfwGetFramebufferSize(janela)[0] / glfwGetWindowSize(janela)[0]
        w, h = w * hdpi_factor, h * hdpi_factor
        gui.update_window(w, h)

        janela_ativa = glfwGetCurrentContext()
        glfwMakeContextCurrent(janela_ativa)

        ajustar_janela(w, h, janela)
        glfwMakeContextCurrent(janela_ativa)

    def on_key(janela, tecla, codigo, acao, mods):
        gui.update_key(tecla, codigo, acao, mods)

    def on_char(janela, char):
        gui.update_char(char)

    def on_button(janela, botao, acao, mods):
        gui.update_button(botao, acao, mods)

    def on_pos(janela, x, y):
        hdpi_factor = float(glfwGetFramebufferSize(janela)[0]/glfwGetWindowSize(janela)[0])
        x, y = x * hdpi_factor, y * hdpi_factor
        print(x, y)
        gui.update_mouse(x, y)

    def on_scroll(janela, x, y):
        gui.update_scroll(x, y)

    def on_close(window):
        global quit
        quit = True

    glfwInit()

    janela = glfwCreateWindow(width, height, "Teste", None, None)

    glfwSetWindowPos(janela, 0, 0)
    # Register callbacks for the window
    glfwSetWindowSizeCallback(janela, on_resize)
    glfwSetWindowCloseCallback(janela, on_close)
    # glfwSetWindowIconifyCallback(janela, on_iconify)
    glfwSetKeyCallback(janela, on_key)
    glfwSetCharCallback(janela, on_char)
    glfwSetMouseButtonCallback(janela, on_button)
    glfwSetCursorPosCallback(janela, on_pos)
    glfwSetScrollCallback(janela, on_scroll)
    # test out new paste function

    glfwMakeContextCurrent(janela)
    init()
    inicializar()

    class Temp(object):
        """Temp class to make objects"""
        def __init__(self):
            pass

    foo = Temp()
    foo.calibrar = False

    gui = ui.UI()
    gui.scale = 1.0

    rightbar = ui.Stretching_Menu('Right Bar', (0, 100), (150, -100))
    rightbar.color = RGBA(0, 0, 0, 0.2)
    rightbar.append(ui.Thumb("calibrar", foo, label="C"))
    gui.append(rightbar)

    if not janela:
        exit()

    cap = cv2.VideoCapture(0)

    on_resize(janela, *glfwGetWindowSize(janela))

    while not quit:
        ret, img = cap.read()
        glfwSetWindowSize(janela, img.shape[1], img.shape[0])
        glfwMakeContextCurrent(janela)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, 1, 0, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        cygl.utils.draw_gl_texture(img, interpolation=True)

        on_resize(janela, *glfwGetWindowSize(janela))
        gui.update()

        glfwSwapBuffers(janela)
        glfwPollEvents()

        glClearColor(0.2, 0.2, 0.2, 1)
        glClear(GL_COLOR_BUFFER_BIT)

    gui.terminate()
    glfwTerminate()


# def demo():
#     global quit
#     quit = False
#
#     # Callback functions
#     def on_resize(window, w, h):
#         h = max(h, 1)
#         w = max(w, 1)
#         hdpi_factor = glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0]
#         w, h = w * hdpi_factor, h*hdpi_factor
#         gui.update_window(w, h)
#         active_window = glfwGetCurrentContext()
#         glfwMakeContextCurrent(active_window)
#         # norm_size = normalize((w,h),glfwGetWindowSize(window))
#         # fb_size = denormalize(norm_size,glfwGetFramebufferSize(window))
#         adjust_gl_view(w,h,window)
#         glfwMakeContextCurrent(active_window)
#
#
#     def on_iconify(window,iconfied):
#         pass
#
#     def on_key(window, key, scancode, action, mods):
#         gui.update_key(key,scancode,action,mods)
#
#         if action == GLFW_PRESS:
#             if key == GLFW_KEY_ESCAPE:
#                 on_close(window)
#             if mods == GLFW_MOD_SUPER:
#                 if key == 67:
#                     # copy value to system clipboard
#                     # ideally copy what is in our text input area
#                     test_val = "copied text input"
#                     glfwSetClipboardString(window,test_val)
#                     print("set clipboard to: %s" %(test_val))
#                 if key == 86:
#                     # copy from system clipboard
#                     clipboard = glfwGetClipboardString(window)
#                     print("pasting from clipboard: %s" %(clipboard))
#
#
#     def on_char(window,char):
#         gui.update_char(char)
#
#     def on_button(window,button, action, mods):
#         gui.update_button(button,action,mods)
#         # pos = normalize(pos,glfwGetWindowSize(window))
#         # pos = denormalize(pos,(frame.img.shape[1],frame.img.shape[0]) ) # Position in img pixels
#
#     def on_pos(window,x, y):
#         hdpi_factor = float(glfwGetFramebufferSize(window)[0]/glfwGetWindowSize(window)[0])
#         x,y = x*hdpi_factor,y*hdpi_factor
#         gui.update_mouse(x,y)
#
#     def on_scroll(window,x,y):
#         gui.update_scroll(x,y)
#
#     def on_close(window):
#         global quit
#         quit = True
#         logger.info('Process closing from window')
#
#     # get glfw started
#     glfwInit()
#
#     window = glfwCreateWindow(width, height, "pyglui demo", None, None)
#     if not window:
#         exit()
#
#     glfwSetWindowPos(window,0,0)
#     # Register callbacks for the window
#     glfwSetWindowSizeCallback(window,on_resize)
#     glfwSetWindowCloseCallback(window,on_close)
#     glfwSetWindowIconifyCallback(window,on_iconify)
#     glfwSetKeyCallback(window,on_key)
#     glfwSetCharCallback(window,on_char)
#     glfwSetMouseButtonCallback(window,on_button)
#     glfwSetCursorPosCallback(window,on_pos)
#     glfwSetScrollCallback(window,on_scroll)
#     # test out new paste function
#
#     glfwMakeContextCurrent(window)
#     init()
#     basic_gl_setup()
#
#     print(glGetString(GL_VERSION))
#
#
#     class Temp(object):
#         """Temp class to make objects"""
#         def __init__(self):
#             pass
#
#     foo = Temp()
#     foo.bar = 34
#     foo.sel = 'mi'
#     foo.selection = ['€','mi', u"re"]
#
#     foo.mytext = "some text"
#     foo.calibrar = False
#
#     def set_text_val(val):
#         foo.mytext = val
#         # print 'setting to :',val
#
#     def pr():
#         print("pyglui version: %s" %(ui.__version__))
#
#     gui = ui.UI()
#     gui.scale = 1.0
#     rightbar = ui.Stretching_Menu('Right Bar', (0, 100), (150, -100))
#     rightbar.color = RGBA(0, 0, 0, 0.2)
#     rightbar.append(ui.Thumb("calibrar", foo, label="C"))
#     gui.append(rightbar)
#
#     teste = ui.Container((0, 0), (100, 100))
#     gui.append(teste)
#
#     on_resize(window,*glfwGetWindowSize(window))
#
#     cap = cv2.VideoCapture(0)
#
#     while not quit:
#         gui.update()
#         # ret, img = cap.read()
#         #
#         # glfwMakeContextCurrent(window)
#         #
#         # glMatrixMode(GL_PROJECTION)
#         # glLoadIdentity()
#         # glOrtho(0, 1, 0, 1, -1, 1)
#         # # glOrtho(0, 1, 0, 1, -1, 1)
#         # glMatrixMode(GL_MODELVIEW)
#         # glLoadIdentity()
#         #
#         # cygl.utils.draw_gl_texture(img, interpolation=True)
#         # gui.update()
#         # cpu_g.update()
#         # cpu_g.draw()
#         # print(T.collapsed,L.collapsed,M.collapsed)
#         # T.collapsed = True
#         # glfwMakeContextCurrent(window)
#         glfwSwapBuffers(window)
#         glfwPollEvents()
#         # glfwMakeContextCurrent(window)
#         # adjust_gl_view(1280,720,window)
#         glClearColor(0.2, 0.2, 0.2, 1)
#         glClear(GL_COLOR_BUFFER_BIT)
#
#     gui.terminate()
#     glfwTerminate()
#     logger.debug("Process done")

if __name__ == '__main__':
    main()