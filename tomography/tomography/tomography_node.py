#!/usr/bin/python3
# import argparse
import glob
import os
import sys
import pathlib
import time
import pickle
from typing import Optional
import numpy as np
import open3d as o3d
import importlib

  
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2


from .tomogram import Tomogram

# equal sys.path.append("../")
# parent_dir = pathlib.Path(__file__).resolve().parent
# if str(parent_dir) not in sys.path:
#    sys.path.append(str(parent_dir))

from .config import POINT_FIELDS_XYZI, GRID_POINTS_XYZI
from .config import Config
from .config import scene

# rsg_root = os.path.dirname(os.path.abspath(__file__)) + '/../..'


class Tomography(Node):
    def __init__(self, cfg: Config):
        print("########################### 进入tomography_node.py的41行的Tomography()类的__init__()初始化函数 ###########################", flush=True)
        super().__init__('pointcloud_tomography')#初始化节点，设置节点名称为pointcloud_tomography
        
        self.declare_parameter("rsg_root", None)#Node 对象 → 代表节点 → 告诉 ROS2 系统 → 我需要用这个参数#对象就是节点
        rsg_root_param = self.get_parameter("rsg_root")#从节点 读到 临时变量
        self.declare_parameter("scene_name", None)#Node 对象 → 代表节点 → 告诉 ROS2 系统 → 我需要用这个参数
        scene_name_param = self.get_parameter("scene_name")#从节点 读到 临时变量

        if rsg_root_param.value is None:
            raise ValueError("Missing required parameter: rsg_root")
        
        if scene_name_param.value is None:
            raise ValueError("Missing required parameter: scene_name")
        
        self.rsg_root = rsg_root_param.get_parameter_value().string_value#将临时变量的值 给对象
        self.scene_name = scene_name_param.get_parameter_value().string_value.lower()#将临时变量的值 统一成小写后 给对象
        
        # scene_module_name = f"config.scene_{self.scene_name}"
        # Use relative import for dynamic loading or absolute package path
        # Assuming 'tomography' is the package name
        scene_module_name = f"tomography.config.scene_{self.scene_name}"#.py文件    拼接出 .py场景配置文件的路径和前缀
        scene_class_name = f"Scene{self.scene_name.capitalize()}"#.py文件中类名    capitalize()将字符串的首字母大写，这个单词翻译就是首字母大写
                                                                 #并且拼接场景名 Bulilding->SceneBulilding，这是类名
        try:
            scene_module = importlib.import_module(scene_module_name)#输入tomography.config.scene_building
                                                                     #对应tomography/config/scene_building.py文件，返回scene_module对象
        except ModuleNotFoundError:
             # Fallback to relative import if running as script or different structure
            scene_module = importlib.import_module(f".config.scene_{self.scene_name}", package="tomography")

        scene_cfg: scene.Scene = getattr(scene_module, scene_class_name)()#获取scene_module对象中的scene_class_name类，并实例化

        self.cfg = cfg #保存为实例变量，等号右边的cfg作用域仅限于__init__,所以需要赋值给实例变量

        self.qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.export_dir = self.rsg_root + cfg.map.export_dir
        self.pcd_file = scene_cfg.pcd.file_name
        # default 场景：从 pcd 目录选取第一个 default_*.pcd
        if self.scene_name == 'default' or self.pcd_file is None:
            pcd_dir = os.path.join(self.rsg_root, 'pcd')
            default_pcds = sorted(glob.glob(os.path.join(pcd_dir, 'default_*.pcd')))
            if not default_pcds:
                raise FileNotFoundError(
                    f"No default_*.pcd found in {pcd_dir}. "
                    "Please add default_xxx.pcd to tomogram_rsc/pcd/"
                )
            self.pcd_file = os.path.basename(default_pcds[0])
            self.get_logger().info(f"Using first default PCD: {self.pcd_file}")
        self.resolution = scene_cfg.map.resolution#保存为实例变量
        self.ground_h = scene_cfg.map.ground_h#保存为实例变量
        self.slice_dh = scene_cfg.map.slice_dh

        self.center = np.zeros(2, dtype=np.float32)#保存为实例变量
        self.tomogram = Tomogram(scene_cfg)#保存为实例变量

        self.get_logger().info(f"PCD file name: {self.pcd_file}")
        if self.pcd_file is None:
            print("########################### tomography_node.py的92行,没找到pcd文件 ###########################", flush=True)
            raise ValueError("PCD file name is not specified.")
        else:
            print("###########################tomography_node.py的95行,开始加载pcd文件###########################", flush=True)
            points = self.loadPCD()
            print("###########################tomography_node.py的97行,加载pcd文件完成###########################", flush=True)

        # Process
        print("###########################tomography_node.py的100行,开始切片和方格化建图###########################", flush=True)
        self.process(points)
        print("###########################tomography_node.py的103行,完成切片和方格化建图###########################", flush=True)

    def initROS(self):
        self.map_frame = self.cfg.ros.map_frame
        pointcloud_topic = self.cfg.ros.pointcloud_topic
        layer_G_topic = self.cfg.ros.layer_G_topic
        layer_C_topic = self.cfg.ros.layer_C_topic
        tomogram_topic = self.cfg.ros.tomogram_topic


        self.pointcloud_pub = self.create_publisher(PointCloud2, pointcloud_topic, self.qos)

        self.layer_G_pub_list = []
        self.layer_C_pub_list = []

        for i in range(self.n_slice):
            layer_G_pub = self.create_publisher(PointCloud2, layer_G_topic + str(i), self.qos)
            self.layer_G_pub_list.append(layer_G_pub)
            layer_C_pub = self.create_publisher(PointCloud2, layer_C_topic + str(i), self.qos)
            self.layer_C_pub_list.append(layer_C_pub)

        self.tomogram_pub = self.create_publisher(PointCloud2, tomogram_topic, self.qos)

    def loadPCD(self):
        pcd = o3d.io.read_point_cloud(f"{self.rsg_root}/pcd/{self.pcd_file}")
        points = np.asarray(pcd.points).astype(np.float32)

        self.get_logger().info(f"PCD points: {points.shape[0]}")

        if points.shape[1] > 3:
            points = points[:, :3]

        # default_xxx.pcd：默认基于“地面平面”向上裁剪 3m（避免高处噪声/天花板影响建图）
        # 仅对 default 场景生效，避免影响其它预设场景
        if getattr(self, "scene_name", "").lower() == "default":
            before_n = int(points.shape[0])
            print("开始裁减{} 3m以上的点云".format(self.pcd_file), flush=True)

            plane_model = None  # (a, b, c, d) for ax+by+cz+d=0
            try:
                # 用下采样点云做地面平面估计，避免过慢
                pcd_ds = o3d.geometry.PointCloud()
                pcd_ds.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                pcd_ds = pcd_ds.voxel_down_sample(voxel_size=max(0.05, float(self.resolution)))

                # 迭代分割多个平面，选择“最水平且内点最多”的平面作为地面
                tmp = pcd_ds
                best_inliers = 0
                for _ in range(3):
                    if len(tmp.points) < 200:
                        break
                    model, inliers = tmp.segment_plane(
                        distance_threshold=0.05,
                        ransac_n=3,
                        num_iterations=300
                    )
                    if len(inliers) == 0:
                        break
                    a, b, c, d = model
                    norm = float(np.sqrt(a * a + b * b + c * c))
                    if norm <= 1e-9:
                        break
                    # “水平”判定：法向量与 Z 轴夹角小（|c| 接近 1）
                    horizontal_score = abs(float(c) / norm)
                    if horizontal_score > 0.90 and len(inliers) > best_inliers:
                        best_inliers = len(inliers)
                        plane_model = (float(a), float(b), float(c), float(d))
                    # 移除本次平面内点，继续找下一平面（类似聚类分出多个平面）
                    tmp = tmp.select_by_index(inliers, invert=True)

            except Exception as e:
                self.get_logger().warn(f"[default] Ground plane detection failed, fallback to ground_h: {e}")

            if plane_model is not None:
                a, b, c, d = plane_model
                n = np.array([a, b, c], dtype=np.float32)
                n_norm = float(np.linalg.norm(n))
                # 统一法向量朝上（c>0），保证“高度”是朝上的正值
                if c < 0:
                    a, b, c, d = -a, -b, -c, -d
                    n = -n
                n_norm = float(np.linalg.norm(n))
                # 点到平面的有符号距离（朝上为正），即“相对地面高度”
                height = (points[:, 0] * a + points[:, 1] * b + points[:, 2] * c + d) / max(n_norm, 1e-6)
                points = points[height <= 3.0]
                self.get_logger().info(
                    f"[default] Ground plane: {a:.4f}x+{b:.4f}y+{c:.4f}z+{d:.4f}=0, "
                    f"kept height<=3.0m"
                )
            else:
                # 回退：按 ground_h 裁剪（旧逻辑）
                z_max = float(self.ground_h) + 3.0
                points = points[points[:, 2] <= z_max]
                self.get_logger().info(f"[default] Fallback crop with z <= {z_max:.2f}m (ground_h based)")

            after_n = int(points.shape[0])
            print("已去除3m以上的点云", flush=True)
            self.get_logger().info(f"[default] Crop points: {before_n} -> {after_n}")
            if after_n == 0:
                raise ValueError(
                    "[default] All points were cropped by the 3m filter. "
                    "Please check point cloud frame / ground plane detection."
                )
        
        self.points_max = np.max(points, axis=0)
        self.points_min = np.min(points, axis=0)           
        self.points_min[-1] = self.ground_h
        self.map_dim_x = int(np.ceil((self.points_max[0] - self.points_min[0]) / self.resolution)) + 4
        self.map_dim_y = int(np.ceil((self.points_max[1] - self.points_min[1]) / self.resolution)) + 4
        n_slice_init = int(np.ceil((self.points_max[2] - self.points_min[2]) / self.slice_dh))
        self.center = (self.points_max[:2] + self.points_min[:2]) / 2
        self.slice_h0 = self.points_min[-1] + self.slice_dh
        self.tomogram.initMappingEnv(self.center, self.map_dim_x, self.map_dim_y, n_slice_init, self.slice_h0)

        self.get_logger().info(f"Map center: [{self.center[0]:.2f}, {self.center[1]:.2f}]", )
        self.get_logger().info(f"Dim_x: {self.map_dim_x}")
        self.get_logger().info(f"Dim_y: {self.map_dim_y}")
        self.get_logger().info(f"Num slices init: {n_slice_init}")

        self.VISPROTO_I, self.VISPROTO_P = \
            GRID_POINTS_XYZI(self.resolution, self.map_dim_x, self.map_dim_y)

        return points
        
    def process(self, points):        
        t_map = 0.0
        t_trav = 0.0
        t_simp = 0.0
        t_all = 0.0
        n_repeat = 10

        """ 
        GPU time benchmark, where CUDA events are synchronized for correct time measurement.
        The function is repeatedly run for n_repeat times to calculate the average processing time of each modules.
        The time of the first warm-up run is excluded to reduce timing fluctuation and exclude the overhead in initial invocations.
        See https://docs.cupy.dev/en/stable/user_guide/performance.html for more details
        """
        for i in range(n_repeat + 1):
            t_start = time.time()
            layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c, t_gpu = self.tomogram.point2map(points)

            if i > 0:
                t_map += t_gpu['t_map']
                t_trav += t_gpu['t_trav']
                t_simp += t_gpu['t_simp']
                t_all += (time.time() - t_start) * 1e3

        self.get_logger().info(f"Num slices simp: {layers_g.shape[0]}")
        self.get_logger().info(f"Num repeats (for benchmarking only): {n_repeat}")
        self.get_logger().info(f" -- avg t_map  (ms): {t_map / n_repeat}")
        self.get_logger().info(f" -- avg t_trav (ms): {t_trav / n_repeat}")
        self.get_logger().info(f" -- avg t_simp (ms): {t_simp / n_repeat}")
        self.get_logger().info(f" -- avg t_all  (ms): {t_all / n_repeat}")

        self.n_slice = layers_g.shape[0]
        
        map_file = os.path.splitext(self.pcd_file)[0]

        self.exportTomogram(np.stack((layers_t, trav_grad_x, trav_grad_y, layers_g, layers_c)), map_file)

        # 保存数据以便定期重新发布
        self.points = points
        self.layers_g = layers_g
        self.layers_t = layers_t
        self.layers_c = layers_c

        self.initROS()
        self.publishPoints(points)
        self.publishLayers(self.layer_G_pub_list, layers_g, layers_t)
        self.publishLayers(self.layer_C_pub_list, layers_c, None)
        self.publishTomogram(layers_g, layers_t)
        
        # 添加定时器，定期重新发布数据，确保RViz重新订阅时能显示
        # 每2秒发布一次，频率足够低不会造成性能问题，但足够高能及时响应RViz的重新订阅
        self.publish_timer = self.create_timer(2.0, self.periodic_publish)

    def exportTomogram(self, tomogram, map_file):        
        data_dict = {
            'data': tomogram.astype(np.float16),
            'resolution': self.resolution,
            'center': self.center,
            'slice_h0': self.slice_h0,
            'slice_dh': self.slice_dh,
        }
        file_name = map_file + '.pickle'
        with open(self.export_dir + file_name, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.get_logger().info(f"Tomogram exported: {file_name}")

    def publishPoints(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()

        header.frame_id = self.map_frame

        point_msg = pc2.create_cloud_xyz32(header, points)
        self.pointcloud_pub.publish(point_msg)

    def publishLayers(self, pub_list, layers, color=None):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()

        header.frame_id = self.map_frame

        layer_points = self.VISPROTO_P.copy()
        layer_points[:, :2] += self.center

        for i in range(layers.shape[0]):
            layer_points[:, 2] = layers[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            if color is not None:
                layer_points[:, 3] = color[i, self.VISPROTO_I[:, 0], self.VISPROTO_I[:, 1]]
            else:
                layer_points[:, 3] = 1.0
        
            valid_points = layer_points[~np.isnan(layer_points).any(axis=-1)]
            points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, valid_points)
            pub_list[i].publish(points_msg) 

    def publishTomogram(self, layers_g, layers_t):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame

        # Extract grid points for all layers at once
        idx_x = self.VISPROTO_I[:, 0]
        idx_y = self.VISPROTO_I[:, 1]
        
        flat_g = layers_g[:, idx_x, idx_y].copy()
        flat_t = layers_t[:, idx_x, idx_y].copy()
        
        n_slice = flat_g.shape[0]

        # Apply tomogram visibility logic
        for i in range(n_slice - 1):
            diff = flat_g[i + 1] - flat_g[i]
            mask_h = diff < self.slice_dh
            flat_g[i, mask_h] = np.nan
            flat_t[i + 1, mask_h] = np.minimum(flat_t[i, mask_h], flat_t[i + 1, mask_h])

        # Flatten arrays to list of points
        g_all = flat_g.flatten()
        t_all = flat_t.flatten()
        
        # Create corresponding XY coordinates
        base_xy = self.VISPROTO_P[:, :2] + self.center
        xy_all = np.tile(base_xy, (n_slice, 1))

        # Filter valid points
        valid_mask = ~np.isnan(g_all)
        
        if np.any(valid_mask):
            global_points = np.column_stack((
                xy_all[valid_mask],
                g_all[valid_mask],
                t_all[valid_mask]
            )).astype(np.float32)
        else:
            global_points = np.empty((0, 4), dtype=np.float32)

        points_msg = pc2.create_cloud(header, POINT_FIELDS_XYZI, global_points)
        self.tomogram_pub.publish(points_msg)

    def periodic_publish(self):
        """定期重新发布所有数据，确保RViz重新订阅时能显示"""
        if hasattr(self, 'points') and hasattr(self, 'layers_g'):
            self.publishPoints(self.points)
            self.publishLayers(self.layer_G_pub_list, self.layers_g, self.layers_t)
            self.publishLayers(self.layer_C_pub_list, self.layers_c, None)
            self.publishTomogram(self.layers_g, self.layers_t)


def main(args=None):
    rclpy.init(args=args)#解析args，并将参数存储到本节点（此时参数还没到对象内部）
    
    cfg = Config()    
    node = Tomography(cfg)

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
