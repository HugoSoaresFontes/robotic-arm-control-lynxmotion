import logging
import time

from .base_backend import Base_Source, Base_Manager
from .fake_backend import Frame

# logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ts = time.time()


class Surface_Source(Base_Source):
    icon_chr = chr(0xec07)

    @property
    def pretty_class_name(self):
        return 'Configurações de superfície'

    def __init__(self, g_pool, frame_size, frame_rate, name=None, preferred_names=(), uid=None, uvc_controls={}):
        super().__init__(g_pool)
        assert name or preferred_names or uid

        self.surfaces = list(filter(lambda surface: surface.detected,
                                    g_pool.surface_tracker.surfaces)) if g_pool.surface_tracker else []

        # self.surface = None
        self.frame_count = 0

        self.surface = None

        surfaces_by_name = {surface.name: surface for surface in self.surfaces}

        # if uid is supplied we init with that
        if uid:
            for surface in self.surfaces:
                if surface.uid == uid:
                    self.surface = surface
                    break

            if not self.surface:
                logger.warning("No surface found that matched {}".format(preferred_names))
        else:
            if name:
                preferred_names = (name,)
            else:
                pass
            assert preferred_names

            for name in preferred_names:
                for s_name in surfaces_by_name.keys():
                    if name in s_name:
                        uid_for_name = surfaces_by_name[s_name].uid

                        for surface in self.surfaces:
                            if surface.uid == uid_for_name:
                                self.surface = surface
                                break

            if not self.surface:
                logger.warning("No surface found that matched {}".format(preferred_names))

        # check if we were sucessfull
        if not self.surface:
            logger.error("Inicialização falhou")

        self.frame_size_backup = frame_size
        self.frame_rate_backup = frame_rate

    def recent_events(self, events):
        if self.surface:
            frame = self.surface.frame
            if frame is not None:
                dt, ts = time.time() - self.g_pool.timebase.value, time.time()
                self.frame_count += 1
                timestamp = self.g_pool.get_timestamp()
                frame = Frame(timestamp, frame, self.frame_count)
                events['frame'] = frame
                self._recent_frame = frame

    def get_init_dict(self):
        d = super().get_init_dict()
        d['frame_size'] = self.frame_size
        d['frame_rate'] = self.frame_rate
        if self.surface:
            d['name'] = self.name

        return d

    @property
    def name(self):
        if self.surface:
            return self.surface.name
        else:
            return "Superfície não encontrada"

    @property
    def frame_size(self):
        return self.frame_size_backup

    @frame_size.setter
    def frame_size(self, new_size):
        # closest match for size
        sizes = [abs(r[0] - new_size[0]) for r in self.frame_sizes]
        best_size_idx = sizes.index(min(sizes))
        size = self.frame_sizes[best_size_idx]

    @property
    def frame_rates(self):
        return (30, 60, 90, 120)

    @property
    def frame_sizes(self):
        return ((700, 700), (1280, 720), (1920, 1080))

    @property
    def frame_rate(self):
        return self.fps

    @frame_rate.setter
    def frame_rate(self, new_rate):
        rates = [abs(r - new_rate) for r in self.frame_rates]
        best_rate_idx = rates.index(min(rates))
        rate = self.frame_rates[best_rate_idx]
        self.fps = rate

    @property
    def jpeg_support(self):
        return True

    @property
    def online(self):
        return bool(self.surface)

    def deinit_ui(self):
        self.remove_menu()

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Superfície: {}".format(self.name)
        self.update_menu()

    def update_menu(self):
        del self.menu[:]
        from pyglui import ui
        ui_elements = []

        # # lets define some  helper functions:
        # def gui_load_defaults():
        #     for c in self.uvc_capture.controls:
        #         try:
        #             c.value = c.def_val
        #         except:
        #             pass
        #
        # def gui_update_from_device():
        #     for c in self.uvc_capture.controls:
        #         c.refresh()
        #
        # def set_frame_size(new_size):
        #     self.frame_size = new_size

        if self.surface is None:
            ui_elements.append(ui.Info_Text('Inicialização falhou'))
            self.menu.extend(ui_elements)
            return

        ui_elements.append(ui.Info_Text('Controles para {}'.format(self.name)))
        # sensor_control = ui.Growing_Menu(label='Configurações da Câmera')
        # sensor_control.collapsed = False
        # image_processing = ui.Growing_Menu(label='Processamento de Imagem')
        # image_processing.collapsed = True
        #
        # sensor_control.append(ui.Selector(
        #     'frame_size', self,
        #     setter=set_frame_size,
        #     selection=self.uvc_capture.frame_sizes,
        #     label='Resolução'
        # ))
        #
        # def frame_rate_getter():
        #     return (self.uvc_capture.frame_rates, [str(fr) for fr in self.uvc_capture.frame_rates])
        # sensor_control.append(ui.Selector('frame_rate', self, selection_getter=frame_rate_getter, label='Taxa de captura'))
        #
        # for control in self.uvc_capture.controls:
        #     c = None
        #     ctl_name = control.display_name
        #
        #     # now we add controls
        #     if control.d_type == bool:
        #         c = ui.Switch('value', control, label=ctl_name, on_val=control.max_val, off_val=control.min_val)
        #     elif control.d_type == int:
        #         c = ui.Slider('value', control, label=ctl_name, min=control.min_val, max=control.max_val, step=control.step)
        #     elif type(control.d_type) == dict:
        #         selection = [value for name, value in control.d_type.items()]
        #         labels = [name for name, value in control.d_type.items()]
        #         c = ui.Selector('value', control, label=ctl_name, selection=selection, labels=labels)
        #     else:
        #         pass
        #     # if control['disabled']:
        #     #     c.read_only = True
        #     # if ctl_name == 'Exposure, Auto Priority':
        #     #     # the controll should always be off. we set it to 0 on init (see above)
        #     #     c.read_only = True
        #
        #     if c is not None:
        #         if control.unit == 'processing_unit':
        #             image_processing.append(c)
        #         else:
        #             sensor_control.append(c)
        #
        # ui_elements.append(sensor_control)
        # if image_processing.elements:
        #     ui_elements.append(image_processing)
        # ui_elements.append(ui.Button("Atualizar", gui_update_from_device))
        # ui_elements.append(ui.Button("Carregar configurações padrão", gui_load_defaults))
        self.menu.extend(ui_elements)

    def cleanup(self):
        self.surfaces = []

        if self.surface:
            self.surface = None
        super().cleanup()


class Surface_Manager(Base_Manager):
    """Manages local USB sources

    Attributes:
        check_intervall (float): Intervall in which to look for new UVC devices
    """
    gui_name = 'Superfícies'

    @property
    def pretty_class_name(self):
        return 'Gerenciador de superfícies'

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.surfaces = self.g_pool.surface_tracker.surfaces

    def get_init_dict(self):
        return {}

    def init_ui(self):
        self.add_menu()

        from pyglui import ui
        ui_elements = []
        ui_elements.append(ui.Info_Text('Surperfícies disponívies'))

        def dev_selection_list():
            default = (None, 'Selecione a superfície')
            dev_pairs = [default] + [(d.uid, d.name) for d in self.surfaces]
            return zip(*dev_pairs)

        def activate(source_uid):
            if not source_uid:
                return

            settings = {
                'frame_size': (500, 500),
                'frame_rate': 30,
                'uid': source_uid
            }

            source = Surface_Source(self.g_pool, **settings)

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
        self.surfaces = None

    def recent_events(self, events):
        pass

    def add_menu(self):
        super(Base_Manager, self).add_menu()
        from pyglui import ui

        self.menu_icon.order = 0.1

        def replace_backend_manager(manager_class):
            self.g_pool.capture_manager.deinit_ui()
            self.g_pool.capture_manager.cleanup()
            self.g_pool.capture_manager = manager_class(self.g_pool)
            self.g_pool.capture_manager.init_ui()

        # We add the capture selection menu
        self.menu.append(ui.Selector(
            'gerenciador_de_video',
            setter=replace_backend_manager,
            getter=lambda: self.__class__,
            selection=[Surface_Manager],
            labels=[b.gui_name for b in [Surface_Manager]],
            label='Gerenciador'
        ))

        # here is where you add all your menu entries.
        self.menu.label = "Gerenciador de vídeo"
