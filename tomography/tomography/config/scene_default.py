from .scene import Scene


class SceneDefault(Scene):
    """使用 pcd 目录下第一个 default_*.pcd 文件的场景，参数与 Plaza 相同"""

    def __init__(self):
        super().__init__()
        # pcd.file_name 在 tomography_node 中根据 default_*.pcd 动态发现
        self.pcd.file_name = None

        # --- 地图与切片 ---
        self.map.resolution = 0.5   # 平面栅格分辨率 [m/格]，越小越精细、计算量越大
        self.map.ground_h = 0.001      # 地面高度 z 基准，低于此值的点云会被当作地面处理
        self.map.slice_dh = 3.0      # 垂直切片高度间隔 [m]，决定 tomogram 层数与每层厚度
        self.map.flip_xyz_output = True  # 导出到本地 txt 时是否将 x/y/z 全部取反

        # --- 可通行性判定 ---
        self.trav.kernel_size = 5   # 局部窗口大小（邻域核边长），用于平滑/统计可通行性
        self.trav.interval_min = 0.50   # 垂直“可用空间厚度”下限 [m]，低于则判为不可通行(低于该值，机器狗钻不进去)
        self.trav.interval_free = 0.65  # “自由空间”厚度阈值 [m]，高于则视为可通行(高于该值，机器狗可以钻进去)
        self.trav.slope_max = 0.3      # 允许的最大坡度（tan），超过判为不可通行
        self.trav.step_max = 0.005        # 允许的最大台阶/落差 [m]，超过判为不可通行
        self.trav.standable_ratio = 0.2 # 邻域内“可站立”栅格比例阈值，用于判定是否可停留
        self.trav.cost_barrier = 50.0   # 代价图中的障碍墙值，超过则视为完全不可通行
        self.trav.safe_margin = 0.4     # 与障碍的安全边距 [m]，规划时留出的缓冲区
        self.trav.inflation = 0.5       # 障碍膨胀半径 [m]，相当于考虑机器人尺寸的膨胀
