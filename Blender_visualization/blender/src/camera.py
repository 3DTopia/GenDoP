import bpy


class Camera:
    def __init__(self, *, first_root, mode, is_mesh):
        camera = bpy.data.objects["Camera"]

        # initial position
        camera.location.x = 4.5  # bmgWabhgTyI-00006_00000
        # camera.location.x = 8.5  # bmgWabhgTyI-00006_00000
        camera.location.y = -10.93  # bmgWabhgTyI-00006_00000
        # camera.location.y = -6.93  # bmgWabhgTyI-00006_00000
        if is_mesh:
            camera.location.z = 5.45  # bmgWabhgTyI-00006_00000
            # camera.location.z = 5.45  # bmgWabhgTyI-00006_00000
            # camera.location.z = 5.6
        else:
            camera.location.z = 5.2

        # wider point of view
        if mode in ["sequence", "raw"]:
            if is_mesh:
                camera.data.lens = 50  # bmgWabhgTyI-00006_00000
                # camera.data.lens = 50  # bmgWabhgTyI-00006_00000
            else:
                camera.data.lens = 85
        elif mode == "frame":
            if is_mesh:
                camera.data.lens = 130
            else:
                camera.data.lens = 140
        elif mode == "video":
            if is_mesh:
                camera.data.lens = 110
            else:
                camera.data.lens = 140

        # camera.location.x += 0.75

        self.mode = mode
        self.camera = camera

        self.camera.location.x += first_root[0]
        self.camera.location.y += first_root[1]

        self._root = first_root

    def update(self, newroot):
        delta_root = newroot - self._root

        self.camera.location.x += delta_root[0]
        self.camera.location.y += delta_root[1]

        self._root = newroot
