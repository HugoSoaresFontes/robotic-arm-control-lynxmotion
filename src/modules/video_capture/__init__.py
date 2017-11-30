"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs
Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

"""
Video Capture provides the interface to get frames from diffferent backends.
Backends consist of a manager and at least one source class. The manager
is a Pupil plugin that provides an GUI that lists all available sources. The
source provides the stream of image frames.
These backends are available:
- UVC: Local USB sources
- Fake: Fallback, static random image
- File: Loads video from file
"""

import logging

logger = logging.getLogger(__name__)

from .base_backend import InitialisationError, StreamError
from .base_backend import Base_Source, Base_Manager
from .fake_backend import Fake_Source, Fake_Manager
# from .file_backend import FileCaptureError, EndofVideoFileError, FileSeekError
# from .file_backend import File_Source, File_Manager
from .uvc_backend import UVC_Source, UVC_Manager
from .surface_backend import Surface_Source, Surface_Manager
from .default_backend import Default_Source, Default_Manager

source_classes = [UVC_Source, Fake_Source, Surface_Source, Default_Source]
manager_classes = [UVC_Manager, Fake_Manager, Default_Manager]
