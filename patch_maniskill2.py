import os
import mani_skill2
import shutil

# 获取 ManiSkill2 的安装路径
ms2_path = os.path.dirname(mani_skill2.__file__)
env_file = os.path.join(ms2_path, "envs", "sapien_env.py")
base_env_file = os.path.join(ms2_path, "envs", "pick_and_place", "base_env.py")

# 备份原文件
for f in [env_file, base_env_file]:
    backup_file = f + '.backup'
    if not os.path.exists(backup_file):
        shutil.copy2(f, backup_file)

# 修改 sapien_env.py
with open(env_file, 'r') as f:
    content = f.read()

# 修改渲染器初始化部分
renderer_code = """        # Create dummy renderer
        self._renderer_type = "none"
        self._renderer = None
        self._cameras = {}"""

content = content.replace(
    """        # Create SAPIEN renderer
        self._renderer_type = renderer
        if renderer_kwargs is None:
            renderer_kwargs = {}
        if self._renderer_type == "sapien":
            self._renderer = sapien.SapienRenderer(**renderer_kwargs)""",
    renderer_code
)

# 修改相机设置方法
setup_cameras_old = """    def _setup_cameras(self):
        self._cameras = OrderedDict()
        for uid, camera_cfg in self._camera_cfgs.items():
            if uid in self._agent_camera_cfgs:
                articulation = self.agent.robot
            else:
                articulation = None
            if isinstance(camera_cfg, StereoDepthCameraConfig):
                cam_cls = StereoDepthCamera
            else:
                cam_cls = Camera
            self._cameras[uid] = cam_cls(
                camera_cfg,
                self._scene,
                self._renderer_type,
                articulation=articulation,
            )

        # Cameras for rendering only
        self._render_cameras = OrderedDict()
        if self._renderer_type != "client":
            for uid, camera_cfg in self._render_camera_cfgs.items():
                self._render_cameras[uid] = Camera(
                    camera_cfg, self._scene, self._renderer_type
                )"""

setup_cameras_new = """    def _setup_cameras(self):
        # Skip camera setup in headless mode
        self._cameras = OrderedDict()
        self._render_cameras = OrderedDict()"""

content = content.replace(setup_cameras_old, setup_cameras_new)

# 修改 _add_ground 方法
ground_old = """    def _add_ground(self, altitude=0.0, render=True):
        if render:
            rend_mtl = self._renderer.create_material()
            rend_mtl.base_color = [0.06, 0.08, 0.12, 1]
            rend_mtl.metallic = 0.0
            rend_mtl.roughness = 0.9
            rend_mtl.specular = 0.8
        else:
            rend_mtl = None
        return self._scene.add_ground(
            altitude=altitude,
            render=render,
            render_material=rend_mtl,
        )"""

ground_new = """    def _add_ground(self, altitude=0.0, render=False):
        return self._scene.add_ground(
            altitude=altitude,
            render=False,
            render_material=None,
        )"""

content = content.replace(ground_old, ground_new)

with open(env_file, 'w') as f:
    f.write(content)

# 修改 base_env.py
with open(base_env_file, 'r') as f:
    content = f.read()

# 修改 _build_cube 方法
cube_old = """    def _build_cube(
        self,
        half_size,
        color=(1, 0, 0),
        name="cube",
        static=False,
        render_material: sapien.RenderMaterial = None,
    ):
        if render_material is None:
            render_material = self._renderer.create_material()
            render_material.set_base_color(np.hstack([color, 1.0]))

        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=render_material)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)"""

cube_new = """    def _build_cube(
        self,
        half_size,
        color=(1, 0, 0),
        name="cube",
        static=False,
        render_material: sapien.RenderMaterial = None,
    ):
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, color=color)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)"""

content = content.replace(cube_old, cube_new)

with open(base_env_file, 'w') as f:
    f.write(content)

print(f"Successfully patched {env_file} and {base_env_file}")
print(f"Original files backed up to {env_file}.backup and {base_env_file}.backup") 