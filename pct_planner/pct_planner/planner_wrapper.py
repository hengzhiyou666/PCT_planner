import os
from re import M
import sys
import pathlib
import pickle
import numpy as np

from .utils import *
from .config import Config

# 保留对源码路径的兼容处理（开发时直接运行）
parent_dir = pathlib.Path(__file__).resolve().parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# 在已安装包环境下，pybind11 模块位于 pct_planner.lib 中
from .lib import a_star, ele_planner, traj_opt

# rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class TomogramPlanner(object):
    def __init__(self, cfg: Config, rsg_root: str):
        self.cfg = cfg

        if rsg_root is None:
            raise ValueError("Missing required parameter: rsg_root")

        self.use_quintic = self.cfg.planner.use_quintic
        self.max_heading_rate = self.cfg.planner.max_heading_rate

        self.tomo_dir = rsg_root + self.cfg.wrapper.tomo_dir

        self.resolution = None
        self.center = None
        self.n_slice = None
        self.slice_h0 = None
        self.slice_dh = None
        self.map_dim = []
        self.offset = None

        self.start_idx = np.zeros(3, dtype=np.int32)
        self.end_idx = np.zeros(3, dtype=np.int32)
        
        # 保存高度数据，用于从 2D 点击位置推断 z 坐标
        self.elev_g = None           # 原始地面高度图（含 nan）
        self.elev_g_clean = None     # 去 nan 的地面高度图（nan -> -100）
        self.elev_c = None           # 天花板高度图

    def loadTomogram(self, tomo_file):
        with open(self.tomo_dir + tomo_file + '.pickle', 'rb') as handle:
            multi_layer_point_cloud_map_and_params = pickle.load(handle)

            tomogram = np.asarray(multi_layer_point_cloud_map_and_params['data'], dtype=np.float32)

            self.resolution = float(multi_layer_point_cloud_map_and_params['resolution'])
            self.center = np.asarray(multi_layer_point_cloud_map_and_params['center'], dtype=np.double)
            self.n_slice = tomogram.shape[1]
            self.slice_h0 = float(multi_layer_point_cloud_map_and_params['slice_h0'])
            self.slice_dh = float(multi_layer_point_cloud_map_and_params['slice_dh'])
            self.map_dim = [tomogram.shape[2], tomogram.shape[3]]
            self.offset = np.array([int(self.map_dim[0] / 2), int(self.map_dim[1] / 2)], dtype=np.int32)

        trav = tomogram[0]
        trav_gx = tomogram[1]
        trav_gy = tomogram[2]
        elev_g = tomogram[3]
        elev_g_clean = np.nan_to_num(elev_g, nan=-100)
        elev_c = tomogram[4]
        elev_c_clean = np.nan_to_num(elev_c, nan=1e6)
        
        # 保存高度数据（用于查询）
        self.elev_g = elev_g              # 原始（含 nan）
        self.elev_g_clean = elev_g_clean  # 去 nan
        self.elev_c = elev_c

        self.initPlanner(trav, trav_gx, trav_gy, elev_g_clean, elev_c_clean)
        
    def initPlanner(self, trav, trav_gx, trav_gy, elev_g, elev_c):
        diff_t = trav[1:] - trav[:-1]
        diff_g = np.abs(elev_g[1:] - elev_g[:-1])

        gateway_up = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t < -8.0
        mask_g = (diff_g < 0.1) & (~np.isnan(elev_g[1:]))
        gateway_up[:-1] = np.logical_and(mask_t, mask_g)

        gateway_dn = np.zeros_like(trav, dtype=bool)
        mask_t = diff_t > 8.0
        mask_g = (diff_g < 0.1) & (~np.isnan(elev_g[:-1]))
        gateway_dn[1:] = np.logical_and(mask_t, mask_g)
        
        gateway = np.zeros_like(trav, dtype=np.int32)
        gateway[gateway_up] = 2
        gateway[gateway_dn] = -2

        self.planner = ele_planner.OfflineElePlanner(
            max_heading_rate=self.max_heading_rate, use_quintic=self.use_quintic
        )
        
        self.planner.init_map(
            20, 15, self.resolution, self.n_slice, 0.2,
            trav.reshape(-1, trav.shape[-1]).astype(np.double),
            elev_g.reshape(-1, elev_g.shape[-1]).astype(np.double),
            elev_c.reshape(-1, elev_c.shape[-1]).astype(np.double),
            gateway.reshape(-1, gateway.shape[-1]),
            trav_gy.reshape(-1, trav_gy.shape[-1]).astype(np.double),
            -trav_gx.reshape(-1, trav_gx.shape[-1]).astype(np.double)
        )

    def plan(self, start_pos, end_pos, start_z=None, end_z=None):
        print("########################### 进入planner_wrapper.py的111行的planner.plan()函数 ###########################", flush=True)
        # 根据 z 坐标计算 slice index，如果没有提供则默认使用 slice 0
        if start_z is not None:
            print(f"########################### 计算起点z坐标对应的start_z值：{start_z:.2f} ###########################", flush=True)
            start_slice = self.z_to_slice_index(start_z)
            self.start_idx[0] = start_slice
            print(f"########################### 计算得到的起点对应的slice index：{start_slice} ###########################", flush=True)
            print(f"[DEBUG] start_z={start_z:.2f}, slice_h0={self.slice_h0:.2f}, slice_dh={self.slice_dh:.2f}, n_slice={self.n_slice}, computed start_slice={start_slice}")
        else:
            self.start_idx[0] = 0  # 默认使用 slice 0
            print("########################### 没有提供起点z坐标，默认使用slice 0 ###########################", flush=True)
        
        if end_z is not None:
            print(f"########################### 计算终点z坐标对应的end_z值：{end_z:.2f} ###########################", flush=True)
            end_slice = self.z_to_slice_index(end_z)
            self.end_idx[0] = end_slice
            print(f"[DEBUG] end_z={end_z:.2f}, slice_h0={self.slice_h0:.2f}, slice_dh={self.slice_dh:.2f}, n_slice={self.n_slice}, computed end_slice={end_slice}")
            print(f"########################### 计算得到的终点对应的slice index：{end_slice} ###########################", flush=True)
        else:
            self.end_idx[0] = 0  # 默认使用 slice 0
            print("########################### 没有提供终点z坐标，默认使用slice 0 ###########################", flush=True)
        
        self.start_idx[1:] = self.pos2idx(start_pos)
        self.end_idx[1:] = self.pos2idx(end_pos)
        print(f"########################### 计算得到的起点索引和终点索引：{self.start_idx} 到 {self.end_idx} ###########################", flush=True)

        print("########################### 调用planner.plan()函数进行路径规划 ###########################", flush=True)
        self.planner.plan(self.start_idx, self.end_idx, True)
        print("########################### planner_wrapper.py的139行的planner.plan()函数进行路径规划完成，暂不知道结果如何 ###########################", flush=True)
        path_finder: a_star.Astar = self.planner.get_path_finder()
        print("########################### planner_wrapper.py的142行的planner.get_path_finder()函数获取路径查找器完成，暂不知道结果如何 ###########################", flush=True)
        path = path_finder.get_result_matrix()
        if len(path) == 0:
            print("########################### planner_wrapper.py的145行的if len(path) == 0: 路径长度为0,路径规划失败 ###########################", flush=True)
            return None
        print("########################### planner_wrapper.py的148行的if len(path) !=0，路径长度不为0,继续执行 ###########################", flush=True)
        optimizer: traj_opt.GPMPOptimizer = (
            self.planner.get_trajectory_optimizer()
            if not self.use_quintic
            else self.planner.get_trajectory_optimizer_wnoj()
        )

        opt_init = optimizer.get_opt_init_value()
        init_layer = optimizer.get_opt_init_layer()
        traj_raw = optimizer.get_result_matrix()
        layers = optimizer.get_layers()
        heights = optimizer.get_heights()

        opt_init = np.concatenate([opt_init.transpose(1, 0), init_layer.reshape(-1, 1)], axis=-1)
        traj = np.concatenate([traj_raw, layers.reshape(-1, 1)], axis=-1)
        y_idx = (traj.shape[-1] - 1) // 2
        traj_3d = np.stack([traj[:, 0], traj[:, y_idx], heights / self.resolution], axis=1)
        traj_3d = transTrajGrid2Map(self.map_dim, self.center, self.resolution, traj_3d)

        return traj_3d
    
    def pos2idx(self, pos):
        pos = pos - self.center
        idx = np.round(pos / self.resolution).astype(np.int32) + self.offset
        idx = np.array([idx[1], idx[0]], dtype=np.float32)
        return idx
    
    def z_to_slice_index(self, z):
        """根据 z 坐标（高度）计算对应的 slice index"""
        print("########################### 进入planner_wrapper.py的176行的z_to_slice_index()函数 ###########################", flush=True)
        if self.slice_h0 is None or self.slice_dh is None:
            return 0  # 默认使用 slice 0
        
        # slice_index = int((z - slice_h0) / slice_dh)
        # 确保在有效范围内 [0, n_slice)
        #print(slice_h0, slice_dh, z)
        #print("计算公式：slice_idx = int((z - self.slice_h0) / self.slice_dh)")
        #slice_idx = int((z - self.slice_h0) / self.slice_dh)
        print("计算公式：slice_idx = int(z/2.25+2.25/2)#四舍五入")
        slice_idx = int(z/2.25+2.25/2)#四舍五入
        
        if slice_idx < 0:
            slice_idx = 0
        elif slice_idx >= self.n_slice:
            slice_idx = self.n_slice - 1
        
        return slice_idx
    
    def is_tomogram_loaded(self):
        """检查 tomogram 是否已加载"""
        return (self.elev_g_clean is not None and 
                self.map_dim is not None and 
                self.center is not None and 
                self.resolution is not None)
    
    def query_height_at_xy(self, x, y):
        """从 tomogram 高度数据中查询指定 (x, y) 位置的高度
        
        Args:
            x, y: 地图坐标（相对于 map frame）
            
        Returns:
            float: 高度值，如果查询失败则返回 None
        """
        if self.elev_g_clean is None:
            return None
        
        try:
            # 转换为网格索引
            idx_xy = self.pos2idx(np.array([x, y]))
            ix = int(round(idx_xy[1]))  # x 索引
            iy = int(round(idx_xy[0]))  # y 索引
            
            # 检查索引是否在有效范围内
            max_x, max_y = self.map_dim
            if ix < 0 or ix >= max_x or iy < 0 or iy >= max_y:
                return None
            
            # 在所有 slice 中查找有效高度
            # 策略：从底层（slice 0）开始，找到第一个有效高度（最接近地面的高度）
            for slice_idx in range(self.n_slice):
                height = self.elev_g_clean[slice_idx, iy, ix]
                # 跳过无效值（-100 表示无数据）
                if height > -90:  # -100 表示无数据，其他为有效高度
                    # 计算实际高度 = slice_h0 + slice_idx * slice_dh
                    actual_height = self.slice_h0 + slice_idx * self.slice_dh
                    return float(actual_height)
            
            # 如果所有 slice 都无效，返回 None
            return None
            
        except Exception as e:
            return None