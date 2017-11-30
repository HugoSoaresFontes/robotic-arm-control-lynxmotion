"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

# logging
import logging
from time import time, sleep
import uvc
from camera_models import load_intrinsics
import collections
import cv2
from .fake_backend import Frame
import numpy as np

from .base_backend import Base_Source, Base_Manager

logger = logging.getLogger(__name__)


class Default_Source(Base_Source):

    def __init__(self, g_pool, frame_size, frame_rate, name=None, preferred_names=(), id=None):
        super().__init__(g_pool)
        self.capture = None

        assert name or preferred_names or id is not None

        self.devices = uvc.Device_List()
        devices_by_name = collections.OrderedDict([(dev['name'], dev) for dev in self.devices])

        # if uid is supplied we init with that
        if id is not None:
            try:
                self.capture = cv2.VideoCapture(id)
                self._name = self.devices[id]['name']
            except Exception:
                logger.warning("No camera found that matched with id={}".format(id))
        else:
            if name:
                preferred_names = (name,)
            else:
                pass
            assert preferred_names

            for name in preferred_names:
                for i, d_name in enumerate(devices_by_name.keys()):
                    if name in d_name:
                        try:
                            id_for_name = i
                            self._name = d_name
                            self.capture = cv2.VideoCapture(i)
                        except Exception as e:
                            logger.info(e)
                        else:
                            break

        # check if we were sucessfull
        if not self.capture:
            logger.error("Inicialização falhou")
            self.name_backup = preferred_names
            self.frame_size_backup = frame_size
            self.frame_rate_backup = frame_rate
        else:
            self.configure_capture(frame_size, frame_rate)
            self._intrinsics = load_intrinsics(self.g_pool.user_dir, self._name, self.frame_size)
            self.name_backup = (self.name,)
            self.frame_size_backup = frame_size
            self.frame_rate_backup = frame_rate

    def configure_capture(self, frame_size, frame_rate):
        self.ts_offset = 0.0
        self.frame_count = 0
        self.frame_size = frame_size
        self.frame_rate = frame_rate

    def update_menu(self):
        del self.menu[:]
        from pyglui import ui
        ui_elements = []

        # lets define some  helper functions:

        if self.capture is None:
            ui_elements.append(ui.Info_Text('Inicialização da captura falhou'))
            self.menu.extend(ui_elements)
            return

        ui_elements.append(ui.Info_Text('Conectado à câmera {}'.format(self._name)))

        self.menu.extend(ui_elements)

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Imagem de câmera sem suporte UVC"

        from pyglui import ui
        text = ui.Info_Text("Fonte de vídeo")
        self.menu.append(text)
        self.update_menu()

    def deinit_ui(self):
        self.remove_menu()

    def recent_events(self, events):
        # try:
        # print(self.capture)
        if self.capture:
            timestamp = self.g_pool.get_timestamp()
            timestamp_base = self.g_pool.timebase.value

            # if self._recent_frame:
            #     passado = timestamp - timestamp_base - self._recent_frame.timestamp
            #     if passado < 1 / self.frame_rate:
            #         print(passado, 1 / self.frame_rate)
            #         self._recent_frame.timestamp -= passado
            #         return

            ret, img = self.capture.read()

            self.frame_count += 1
            if img is not None:
                frame = Frame(timestamp, img, self.frame_count)
                frame.timestamp -= self.g_pool.timebase.value
                self._recent_frame = frame
                events['frame'] = frame
            else:
                self._recent_frame = None
        else:
            self._recent_frame = None
        # except Exception as e:
        #     self._recent_frame = None

        # now = time()
        # spent = now - self.presentation_time
        # wait = max(0, 1. / self.fps - spent)
        # sleep(wait)
        # self.presentation_time = time()
        # self.frame_count += 1
        # timestamp = self.g_pool.get_timestamp()
        # frame = Frame(timestamp, self._img.copy(), self.frame_count)
        # cv2.putText(frame.img, "Frame %s" % self.frame_count, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100))
        # events['frame'] = frame
        # self._recent_frame = frame

    @property
    def name(self):
        if self.capture:
            return self._name
        else:
            return 'Gost Capture'

    @property
    def intrinsics(self):
        return self._intrinsics

    def cleanup(self):
        if self.devices:
            self.devices.cleanup()
        self.devices = None
        if self.capture:
            self.capture.release()
            self.capture = None
        super().cleanup()

    @property
    def settings(self):
        return {'frame_size': self.frame_size, 'frame_rate': self.frame_rate}

    @settings.setter
    def settings(self, settings):
        self.frame_size = settings.get('frame_size', self.frame_size)
        self.frame_rate = settings.get('frame_rate', self.frame_rate)

    @property
    def frame_rate(self):
        if self.capture:
            return int(self.capture.get(cv2.CAP_PROP_FPS))
        else:
            return self.frame_rate_backup

    @property
    def frame_size(self):
        if self.capture and self.capture.isOpened():
            return (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        else:
            return self.frame_size_backup

    @frame_size.setter
    def frame_size(self, new_size):
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, new_size[0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, new_size[1])

        self.frame_size_backup = new_size

    @property
    def frame_rates(self):
        return (30, )

    @property
    def frame_sizes(self):
        return ((640, 480), (1280, 720), (1920, 1080))

    @property
    def frame_rate(self):
        if self.capture:
            return self.capture.get(cv2.CAP_PROP_FPS)
        else:
            return self.frame_rate_backup

    @frame_rate.setter
    def frame_rate(self, new_rate):
        pass

    @property
    def jpeg_support(self):
        return False

    @property
    def online(self):
        return bool(self.capture)

    def get_init_dict(self):
        d = super().get_init_dict()
        d['frame_size'] = self.frame_size
        d['frame_rate'] = self.frame_rate
        d['name'] = self.name
        return d


class Default_Manager(Base_Manager):
    """Manages local USB sources

    Attributes:
        check_intervall (float): Intervall in which to look for new UVC devices
    """
    gui_name = 'Câmeras (OPENCV)'

    @property
    def pretty_class_name(self):
        return 'Gerenciador de dispositivos'

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.devices = uvc.Device_List()

    def get_init_dict(self):
        return {}

    def init_ui(self):
        self.add_menu()

        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text('Dispositivos locais'))

        def dev_selection_list():
            default = (None, 'Selecione o dispositivo')
            self.devices.update()
            dev_pairs = [default] + [(i, d['name']) for i, d in enumerate(self.devices)]
            return zip(*dev_pairs)

        def activate(id):
            if id is None:
                return
            capture = cv2.VideoCapture(id)
            if not capture.isOpened():
                logger.error("A câmera selecionada está em uso ou bloqueada")
                capture.release()
                return
            capture.release()

            self.g_pool.capture.deinit_ui()
            self.g_pool.capture.cleanup()
            self.g_pool.plugins._plugins.remove(
                self.g_pool.capture
            )

            settings = {
                'frame_size': self.g_pool.capture.frame_size,
                'frame_rate': self.g_pool.capture.frame_rate,
                'id': id
            }

            self.g_pool.plugins.add(
                Default_Source, settings
            )
            # self.g_pool.capture.__init__(self.g_pool, **settings)

        ui_elements.append(ui.Selector(
            'selected_source',
            selection_getter=dev_selection_list,
            getter=lambda: None,
            setter=activate,
            label='Dispositivo ativo'
        ))
        self.menu.extend(ui_elements)

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        if self.devices:
            self.devices.cleanup()
        self.devices = None

    def recent_events(self, events):
        pass
