import torch
from opt import get_opts
import numpy as np
from einops import rearrange
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
import time

from datasets import dataset_dict
from datasets.ray_utils import get_ray_directions, get_rays
from models.networks import NGP
from models.rendering import render
from utils import load_ckpt

import warnings; warnings.filterwarnings("ignore")


class OrbitCamera:
    def __init__(self, img_wh, r=5, fovy=50):
        self.W, self.H = img_wh
        self.radius = r  # camera distance from center
        self.fovy = fovy  # in degree
        self.center = np.zeros(3, dtype=np.float32)
        self.rot = R.from_quat([0, 1, 0, 0])
        self.up = np.float32([0, 1, 0])

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def orbit(self, dx, dy):
        # TODO: this requires change....
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(0.03 * dx)
        rotvec_y = side * np.radians(-0.03 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 0.0001 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])


class NeRFGUI:
    def __init__(self, hparams, K, img_wh, radius=2.5, fovy=50):
        self.hparams = hparams
        self.model = NGP(scale=hparams.scale).cuda()
        load_ckpt(self.model, hparams.ckpt_path)

        self.W, self.H = img_wh
        self.directions = get_ray_directions(self.H, self.W, K).cuda()

        self.cam = OrbitCamera(img_wh, r=radius, fovy=fovy)
        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)

        self.dt = 0

        self.register_dpg()

    def render_pose(self, pose):
        t = time.time()
        rays_o, rays_d = \
            get_rays(self.directions, torch.cuda.FloatTensor(pose))
        if self.hparams.dataset_name in ['colmap', 'nerfpp']:
            exp_step_factor = 1/256
        else: exp_step_factor = 0

        results = render(self.model, rays_o, rays_d,
                         **{'test_time': True,
                            'T_threshold': 1e-2,
                            'exp_step_factor': exp_step_factor})

        rgb_pred = rearrange(results["rgb"].cpu().numpy(),
                             "(h w) c -> h w c", h=self.H)
        torch.cuda.synchronize()
        self.dt = time.time()-t

        return rgb_pred

    def register_dpg(self):
        dpg.create_context()
        ## register texture ##
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture")

        ## register window ##
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")
        dpg.set_primary_window("_primary_window", True)

        ## control window ##
        with dpg.window(label="Control", tag="_control_window", width=200, height=150):
            with dpg.collapsing_header(label="Info", default_open=True):
                dpg.add_separator()
                dpg.add_text('', tag="_log_time")

        ## register camera handler ##
        def callback_camera_drag_rotate(sender, app_data):
            dx = app_data[1]
            dy = app_data[2]
            self.cam.orbit(dx, dy)

        def callback_camera_wheel_scale(sender, app_data):
            delta = app_data
            self.cam.scale(delta)

        def callback_camera_drag_pan(sender, app_data):
            dx = app_data[1]
            dy = app_data[2]
            self.cam.pan(dx, dy)

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        ## Window name ##
        dpg.create_viewport(title="ngp", width=self.W, height=self.H, resizable=False)

        ## Avoid scroll bar in the window ##
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        ## Launch the gui ##
        dpg.setup_dearpygui()
        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            dpg.set_value("_texture", self.render_pose(self.cam.pose))
            dpg.set_value("_log_time", f'Render time: {1000*self.dt:.2f} ms')
            dpg.render_dearpygui_frame()


if __name__ == "__main__":
    hparams = get_opts()
    kwargs = {'root_dir': hparams.root_dir,
              'downsample': hparams.downsample,
              'read_meta': False}
    dataset = dataset_dict[hparams.dataset_name](**kwargs)

    NeRFGUI(hparams, dataset.K, dataset.img_wh).render()

    dpg.destroy_context()
