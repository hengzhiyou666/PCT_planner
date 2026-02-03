from .scene import Scene


class SceneDefault(Scene):
    """使用 pcd 目录下第一个 default_*.pcd 文件的场景，参数与 Plaza 相同"""

    def __init__(self):
        super().__init__()
        # pcd.file_name 在 tomography_node 中根据 default_*.pcd 动态发现
        self.pcd.file_name = None

        self.map.resolution = 0.20
        self.map.ground_h = 0.0
        self.map.slice_dh = 0.5

        self.trav.kernel_size = 7
        self.trav.interval_min = 0.50
        self.trav.interval_free = 0.65
        self.trav.slope_max = 0.36
        self.trav.step_max = 0.17
        self.trav.standable_ratio = 0.2
        self.trav.cost_barrier = 50.0
        self.trav.safe_margin = 0.4
        self.trav.inflation = 0.2
