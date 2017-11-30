'''
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2017  Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
'''

import platform
import sys, os
import shutil
from subprocess import call

if platform.system() == 'Linux':

    distribtution_dir = 'dist'
    main = os.path.join(distribtution_dir, 'controlador_ssc32')

    old_deb_dir = [d for d in os.listdir('.') if d.startswith('controlador_ssc32')]
    for d in old_deb_dir:
        try:
            shutil.rmtree(d)
            print( 'removed deb structure dir: "%s"' %d)
        except:
            pass

    #lets build the structure for our deb package.
    deb_root = 'controlador_ssc32_linux_os_x64_v0.1'
    DEBIAN_dir = os.path.join(deb_root,'DEBIAN')
    opt_dir = os.path.join(deb_root, 'opt')
    bin_dir = os.path.join(deb_root, 'usr', 'bin')
    app_dir = os.path.join(deb_root, 'usr', 'share', 'applications')
    ico_dir = os.path.join(deb_root, 'usr', 'share', 'icons', 'hicolor', 'scalable', 'apps')
    os.makedirs(DEBIAN_dir,0o755)
    os.makedirs(bin_dir,0o755)
    os.makedirs(app_dir,0o755)
    os.makedirs(ico_dir,0o755)

    #DEBAIN Package description
    with open(os.path.join(DEBIAN_dir,'control'),'w') as f:
        dist_size = sum(os.path.getsize(os.path.join(main,f)) for f in os.listdir(main) if os.path.isfile(os.path.join(main,f)))
        content = '''\
Package: controlador-ssc32
Version: %s
Architecture: amd64
Maintainer: Cefas Rodrigues <cefas.rodrigues04@gmail.com>
Priority: optional
Description: Controlador para módulo SSC32 com mapeamento de câmera
Installed-Size: %s
'''%('0.1',dist_size/1024)
        f.write(content)
    os.chmod(os.path.join(DEBIAN_dir,'control'),0o644)

    #pre install script
    with open(os.path.join(DEBIAN_dir,'preinst'),'w') as f:
        content = '''\
#!/bin/sh
echo 'SUBSYSTEM=="usb",  ENV{DEVTYPE}=="usb_device", GROUP="plugdev", MODE="0664"' > /etc/udev/rules.d/10-libuvc.rules
udevadm trigger'''
        f.write(content)
    os.chmod(os.path.join(DEBIAN_dir,'preinst'),0o755)


    #bin_starter script
    with open(os.path.join(bin_dir,'controlador_ssc32'),'w') as f:
        content = '''\
#!/bin/sh
exec /opt/controlador_ssc32/controlador_ssc32 "$@"'''
        f.write(content)
    os.chmod(os.path.join(bin_dir,'controlador_ssc32'),0o755)


    #.desktop entry
    with open(os.path.join(app_dir,'controlador_ssc32.desktop'),'w') as f:
        content = '''\
[Desktop Entry]
Version=1.0
Type=Application
Name=Controlador SSC32
Comment=Módulo de controle de braço robótico através de webcam
Exec=/opt/controlador_ssc32/controlador_ssc32
Terminal=false
Icon=controlador-ssc32
Categories=Application;
Name[pt_BR]=Controlador SSC32
Actions=Terminal;

[Desktop Action Terminal]
Name=Open in Terminal
Exec=x-terminal-emulator -e controlador_ssc32'''
        f.write(content)
    os.chmod(os.path.join(app_dir,'controlador_ssc32.desktop'),0o644)

    #copy icon:
    shutil.copy('controlador-ssc32.svg',ico_dir)
    os.chmod(os.path.join(ico_dir,'controlador-ssc32.svg'),0o644)

    #copy the actual application
    shutil.copytree(distribtution_dir,opt_dir)
    # set permissions
    for root, dirs, files in os.walk(opt_dir):
        for name in files:
            if name == 'controlador_ssc32':
                os.chmod(os.path.join(root,name),0o755)
            else:
                os.chmod(os.path.join(root,name),0o644)
        for name in dirs:
            os.chmod(os.path.join(root,name),0o755)
    os.chmod(opt_dir,0o755)


    #run dpkg_deb
    call('fakeroot dpkg-deb --build %s' % deb_root, shell=True)

    print( 'DONE!')