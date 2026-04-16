import glob
import os
import re
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped, PoseStamped

from pct_planner.utils import traj2ros
from pct_planner.planner_wrapper import TomogramPlanner
from pct_planner.config import Config

class PCTPlanner(Node):
    def __init__(self):
        super().__init__('pct_planner')
        
        # Declare parameters
        self.declare_parameter("rsg_root", rclpy.Parameter.Type.STRING)
        self.declare_parameter("scene_name", "Plaza")
        
        # Get parameters
        rsg_root_param = self.get_parameter("rsg_root")
        if rsg_root_param.value is None:
            self.get_logger().error("Missing required parameter: rsg_root")
            raise ValueError("Missing required parameter: rsg_root")
        self.rsg_root = rsg_root_param.get_parameter_value().string_value
        
        scene_name = self.get_parameter("scene_name").get_parameter_value().string_value
        self.get_logger().info(f"Using scene: {scene_name}")
        
        # Scene configuration
        print("########################### 配置场景 PctPlanner()中__init__方法中配置场景 ###########################", flush=True)
        self.configure_scene(scene_name)
        
        self.cfg = Config()
        
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        print("########################### 设置路径的话题 ###########################", flush=True)
        self.path_pub = self.create_publisher(Path, "/pct_path", qos)
        print("########################### 创建TomogramPlanner对象，并传入配置文件和RSG根目录 ###########################", flush=True)
        self.planner = TomogramPlanner(self.cfg, self.rsg_root)

        # RViz 可选范围边界框（不影响原来的地图/点云显示）
        self.bounds_marker_pub = self.create_publisher(Marker, "/pct_map_bounds", qos)
        self.bounds_marker_timer = self.create_timer(1.0, self.publish_bounds_marker)

        # 交互式取点（来自 RViz Publish Point）：起点 + 无限途径点
        self.has_start = False
        self.points_seq = []  # 已确认的点序列，每个元素为 np.array([x,y,z])

        # 旧逻辑兼容字段（避免旧函数被调用时报错）
        self.point1_begin = None  # 起点坐标 [x, y, z]
        self.point2_end = None    # 终点坐标 [x, y, z]

        # 累计大路径（用于 RViz 不丢弃旧路径）
        self.big_path_traj = None  # shape: (N,3) numpy
        self.segment_count = 0
        self.segments = []  # 每段轨迹的列表，用于重写带标题的文件

        # 输出文件（写到启动命令所在目录）
        cwd = os.getcwd()
        self.big_path_file = os.path.join(cwd, "整体路径.txt")
        self.current_multi_path_file = None  # 例如 “pct_path_2paths.txt”、“pct_path_3paths.txt”

        # 2D Nav Goal 分段导出：基于当前累计路径做差分导出
        self._nav_goal_path_point_counts = []  # 每个 pathN.txt 已写入的点数，用于计算下一段差分

        # 启动时删除旧的路径文件，防止在旧文件上追加
        for fname in os.listdir(cwd):
            if (
                fname == "整体路径.txt"
                or fname.endswith("条路径.txt")
                or (fname.startswith("pct_path_") and fname.endswith("paths.txt"))
                or fname in ("onebigpath.txt", "onebigpath_with_title.txt")
                or re.fullmatch(r"path\d+\.txt", fname)
            ):
                try:
                    os.remove(os.path.join(cwd, fname))
                except OSError:
                    pass

        # 订阅 RViz 的 Publish Point 按钮
        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            "/clicked_point",
            self.clicked_point_callback,
            10,
        )

        # 订阅 RViz 的 2D Nav Goal（结束本轮并清空显示）
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_pose_callback,
            10,
        )

        # 启动提示：等待用户在 RViz 中点击取点
        print("请输入起点：", flush=True)
        self.get_logger().info("\n请输入起点：")

    def configure_scene(self, scene_name):
        scene_name = scene_name.strip().capitalize() if scene_name else ''
        if scene_name == 'Default':
            # 从 tomogram 目录选取第一个 default_*.pickle（相对路径）
            tomo_dir = os.path.join(self.rsg_root, 'tomogram')
            default_pickles = sorted(glob.glob(os.path.join(tomo_dir, 'default_*.pickle')))
            if not default_pickles:
                raise FileNotFoundError(
                    f"No default_*.pickle found in {tomo_dir}. "
                    "Run tomography with scene_name:=default first."
                )
            self.tomo_file = os.path.basename(default_pickles[0]).replace('.pickle', '')
            self.start_pos_xy = np.array([0.0, 0.0], dtype=np.float32)
            self.end_pos_xy = np.array([5.0, 5.0], dtype=np.float32)
        elif scene_name == 'Plaza':
            self.tomo_file = 'plaza3_10'
            self.start_pos_xy = np.array([0.0, 0.0], dtype=np.float32)
            self.end_pos_xy = np.array([23.0, 10.0], dtype=np.float32)
        elif scene_name == 'Building':
            self.tomo_file = 'building2_9'
            print("########################### 选择Building场景名称，并设置起点和终点坐标 ###########################", flush=True)
            self.start_pos_xy = np.array([5.0, 5.0], dtype=np.float32)
            self.end_pos_xy = np.array([-6.0, -1.0], dtype=np.float32)
        elif scene_name == 'Spiral':
            self.tomo_file = 'spiral0.3_2'
            self.start_pos_xy = np.array([-16.0, -6.0], dtype=np.float32)
            self.end_pos_xy = np.array([-26.0, -5.0], dtype=np.float32)
        else:
            self.get_logger().error(f"Invalid scene name: {scene_name}")
            raise ValueError(f"Invalid scene name: {scene_name}")

    def clicked_point_callback(self, msg: PointStamped):
        """接收 RViz 的 Publish Point 按钮点击"""
        # 提取 x, y, z 坐标
        x = msg.point.x
        y = msg.point.y
        z = msg.point.z
        
        # 如果 z 为 0 或接近 0，从 tomogram 中查询高度
        if abs(z) < 0.01:
            # 确保 tomogram 已加载
            if not self.planner.is_tomogram_loaded():
                self.get_logger().info(f"Loading tomogram for height query: {self.tomo_file}")
                self.planner.loadTomogram(self.tomo_file)
            
            if self.planner.is_tomogram_loaded() and hasattr(self.planner, 'query_height_at_xy'):
                queried_z = self.planner.query_height_at_xy(x, y)
                if queried_z is not None:
                    z = queried_z
                    self.get_logger().info(
                        f"Auto-inferred height: z={z:.2f} (queried from tomogram at [{x:.2f}, {y:.2f}])"
                    )
                else:
                    self.get_logger().warn(
                        f"Could not query height at [{x:.2f}, {y:.2f}], using z=0.0"
                    )

        p = np.array([x, y, z], dtype=np.float32)

        # 第一次点击：设置起点
        if not self.has_start:
            self.points_seq = [p]
            self.has_start = True
            self.get_logger().info(f"起点坐标为：({x:.2f}, {y:.2f}, {z:.2f})")
            print("请输入第一个途径点：", flush=True)
            self.get_logger().info("请输入第一个途径点：")
            return

        # 后续每次点击：作为“下一个途径点”，规划一段并累计显示/写文件
        prev = self.points_seq[-1]
        curr = p
        self.points_seq.append(curr)
        self.get_logger().info(f"途径点为：({x:.2f}, {y:.2f}, {z:.2f})")

        ok = self._plan_and_accumulate(prev, curr)
        if not ok:
            # 规划失败：回退该点，继续等待用户重新点同一个途径点
            self.points_seq.pop()
            return

        # 成功后继续提示下一个途径点（可无限输入）
        print("请输入下一个途径点：", flush=True)
        self.get_logger().info("请输入下一个途径点：")
        return

    def goal_pose_callback(self, msg: PoseStamped):
        """RViz 2D Nav Goal：导出 pathN.txt，但不清空当前路径与点序列。"""
        # 点击 2D Nav Goal 时仅触发 pathN.txt 导出；
        # 保留 RViz 上现有路径，且下一次 Publish Point 继续从上一个点衔接规划。
        self._export_path_files_on_nav_goal()

        self.get_logger().info("已导出 pathN.txt，当前路径与起点状态保持不变，可继续追加途径点。")
        if self.has_start:
            print("请输入下一个途径点：", flush=True)
            self.get_logger().info("请输入下一个途径点：")
        else:
            print("请输入起点：", flush=True)
            self.get_logger().info("\n请输入起点：")

    def _segment_title(self, k: int) -> str:
        return f"path{k}:"

    def _export_z(self, z_value: float) -> float:
        """导出到 txt 时的 z：保持原始数值"""
        return float(z_value)

    def _flip_xyz_if_needed(self, x_value: float, y_value: float, z_value: float):
        """根据配置决定是否对导出 xyz 同时取反。"""
        flip = bool(getattr(self.planner, "flip_xyz_output", False))
        if flip:
            return -float(x_value), -float(y_value), -float(z_value)
        return float(x_value), float(y_value), float(z_value)

    def _traj_for_local_export(self, traj_3d: np.ndarray) -> np.ndarray:
        """导出到本地 txt 前，将轨迹转换到点云原始坐标系（若可用）。"""
        if hasattr(self, "planner") and hasattr(self.planner, "transform_traj_to_original_frame"):
            try:
                return self.planner.transform_traj_to_original_frame(traj_3d)
            except Exception as e:
                self.get_logger().warn(f"Transform to original frame failed, fallback to aligned frame: {e}")
                return traj_3d
        return traj_3d

    def _merge_traj_skip_duplicate_joint(self, base: np.ndarray, extra: np.ndarray) -> np.ndarray:
        """将 extra 拼到 base 后；若首尾点重合则去掉 extra 的首点（与段间拼接规则一致）。"""
        base = np.asarray(base, dtype=np.float64)
        extra = np.asarray(extra, dtype=np.float64)
        if extra.size == 0:
            return base
        if base.size == 0:
            return extra.copy()
        if np.allclose(base[-1, :3], extra[0, :3], rtol=0.0, atol=1e-4):
            return np.vstack([base, extra[1:]])
        return np.vstack([base, extra])

    def _write_xyz_path_txt(self, file_path: str, traj: np.ndarray) -> None:
        """与 整体路径.txt 相同的逐行格式写入。"""
        traj = np.asarray(traj, dtype=np.float64)
        with open(file_path, "w", encoding="utf-8") as f:
            for pt in traj:
                x_out, y_out, z_out = self._flip_xyz_if_needed(float(pt[0]), float(pt[1]), float(pt[2]))
                z_out = self._export_z(z_out)
                f.write(f"{x_out:.6f} {y_out:.6f} {z_out:.6f}\n")

    def _export_path_files_on_nav_goal(self) -> None:
        """若有累计路径则写入 pathN.txt（path1=整体，pathk=相对之前导出的新增段）。"""
        if self.big_path_traj is None or len(self.big_path_traj) == 0:
            return

        curr = np.asarray(self._traj_for_local_export(self.big_path_traj), dtype=np.float64)
        committed = int(sum(self._nav_goal_path_point_counts))
        if committed >= len(curr):
            self.get_logger().info("2D Nav Goal：当前无新增路径点，跳过 pathN.txt 导出。")
            return

        segment = curr[committed:].copy()
        n_file = len(self._nav_goal_path_point_counts) + 1
        out_path = os.path.join(os.getcwd(), f"path{n_file}.txt")
        self._write_xyz_path_txt(out_path, segment)
        self._nav_goal_path_point_counts.append(int(len(segment)))

        self.get_logger().info(
            f"2D Nav Goal：已写入 {os.path.basename(out_path)}（{len(segment)} 点），"
            f"累计轨迹 {len(curr)} 点。"
        )

    def _append_to_files(self, traj_3d: np.ndarray, seg_idx: int):
        # “整体路径.txt”：始终保存当前轮的整体路径坐标（所有段拼接）
        if self.big_path_traj is not None:
            export_big = self._traj_for_local_export(self.big_path_traj)
            with open(self.big_path_file, "w", encoding="utf-8") as f1:
                for pt in export_big:
                    x_out, y_out, z_out = self._flip_xyz_if_needed(float(pt[0]), float(pt[1]), float(pt[2]))
                    z_out = self._export_z(z_out)
                    f1.write(f"{x_out:.6f} {y_out:.6f} {z_out:.6f}\n")

        # “pct_path_Npaths.txt”：根据当前段数命名，写入每段前加 path1:/path2: 标题行
        cwd = os.getcwd()
        new_multi = os.path.join(cwd, f"pct_path_{seg_idx}paths.txt")
        # 删除上一轮 “pct_path_(N-1)paths.txt”
        if seg_idx > 1:
            old_multi = os.path.join(cwd, f"pct_path_{seg_idx-1}paths.txt")
            if os.path.exists(old_multi):
                try:
                    os.remove(old_multi)
                except OSError:
                    pass
        # 重写当前 pct_path_Npaths.txt，包含从第1段到当前段
        with open(new_multi, "w", encoding="utf-8") as f2:
            for k, seg in enumerate(self.segments, start=1):
                f2.write(self._segment_title(k) + "\n")
                export_seg = self._traj_for_local_export(seg)
                for pt in export_seg:
                    x_out, y_out, z_out = self._flip_xyz_if_needed(float(pt[0]), float(pt[1]), float(pt[2]))
                    z_out = self._export_z(z_out)
                    f2.write(f"{x_out:.6f} {y_out:.6f} {z_out:.6f}\n")
        self.current_multi_path_file = new_multi

    def _point_in_bounds(self, p_xyz: np.ndarray) -> bool:
        if self.planner.center is None or self.planner.map_dim is None or self.planner.resolution is None:
            return True
        dim_x, dim_y = self.planner.map_dim
        center_x, center_y = self.planner.center
        resolution = self.planner.resolution
        half_extent_x = dim_x * resolution / 2.0
        half_extent_y = dim_y * resolution / 2.0
        min_x = center_x - half_extent_x
        max_x = center_x + half_extent_x
        min_y = center_y - half_extent_y
        max_y = center_y + half_extent_y
        return (min_x <= float(p_xyz[0]) <= max_x) and (min_y <= float(p_xyz[1]) <= max_y)

    def _plan_and_accumulate(self, p_from: np.ndarray, p_to: np.ndarray) -> bool:
        """规划一段路径并累计发布/写文件。成功返回 True。"""
        # 确保 tomogram 已加载
        if not self.planner.is_tomogram_loaded():
            self.get_logger().info(f"Loading tomogram: {self.tomo_file}")
            self.planner.loadTomogram(self.tomo_file)

        # 边界检查（避免 C++ 越界崩）
        if not self._point_in_bounds(p_from) or not self._point_in_bounds(p_to):
            self.get_logger().warn("规划失败：点不在有效范围内，请在 tomogram 范围内重新取点。")
            return False

        start_xy = p_from[:2]
        end_xy = p_to[:2]
        start_z = float(p_from[2])
        end_z = float(p_to[2])

        self.get_logger().info("下面进行路径规划：")
        traj_3d = self.planner.plan(start_xy, end_xy, start_z, end_z)
        if traj_3d is None or len(traj_3d) == 0:
            self.get_logger().warn("规划失败：无法生成可行轨迹，请重新取点。")
            return False

        traj_3d = np.asarray(traj_3d, dtype=np.float32)
        self.segment_count += 1
        self.segments.append(traj_3d)

        # 累计大路径：后续段去掉第一个点，避免重复
        if self.big_path_traj is None:
            self.big_path_traj = traj_3d
        else:
            self.big_path_traj = np.vstack([self.big_path_traj, traj_3d[1:]])

        # 发布累计路径（RViz 中不会丢弃之前段）
        self.path_pub.publish(traj2ros(self.big_path_traj))

        # 写文件（写本段坐标；大文件会自然累加）
        self._append_to_files(traj_3d, self.segment_count)

        self.get_logger().info(
            f"已生成第 {self.segment_count} 段全局路径，并已追加写入 "
            f"{os.path.basename(self.big_path_file)} / "
            f"{os.path.basename(self.current_multi_path_file) if self.current_multi_path_file else ''}"
        )
        return True

    def publish_bounds_marker(self):
        """在 RViz 中发布可选点范围矩形框（基于 tomogram 的中心、尺寸、分辨率）"""
        if self.planner.center is None or self.planner.map_dim is None or self.planner.resolution is None:
            return

        center_x, center_y = float(self.planner.center[0]), float(self.planner.center[1])
        dim_x, dim_y = int(self.planner.map_dim[0]), int(self.planner.map_dim[1])
        res = float(self.planner.resolution)

        half_extent_x = dim_x * res / 2.0
        half_extent_y = dim_y * res / 2.0

        min_x, max_x = center_x - half_extent_x, center_x + half_extent_x
        min_y, max_y = center_y - half_extent_y, center_y + half_extent_y

        z = 0.2  # 让线框略微抬起，避免与地面/点云完全重合看不清

        p1 = Point(x=min_x, y=min_y, z=z)
        p2 = Point(x=max_x, y=min_y, z=z)
        p3 = Point(x=max_x, y=max_y, z=z)
        p4 = Point(x=min_x, y=max_y, z=z)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pct_planner"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.08  # 线宽（米）

        # 绿色边框
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9

        # 闭合矩形：p1->p2->p3->p4->p1
        marker.points = [p1, p2, p3, p4, p1]

        self.bounds_marker_pub.publish(marker)

    def plan_with_points(self):
        #"""使用缓存的起点和终点进行路径规划"""
        print("########################### 使用缓存的起点和终点进行路径规划 ###########################", flush=True)
        if self.point1_begin is None or self.point2_end is None:
            return

        # 确保 tomogram 已加载，以便进行边界检查
        if not self.planner.is_tomogram_loaded():
            print("########################### 确保 tomogram 已加载，以便进行边界检查 ###########################", flush=True)
            self.get_logger().info(f"Loading tomogram for bounds check: {self.tomo_file}")
            self.planner.loadTomogram(self.tomo_file)

        # 简单检查交互点是否落在 tomogram 地图范围内，避免 C++ 端越界崩溃
        try:
            print("########################### 简单检查交互点是否落在 tomogram 地图范围内，避免 C++ 端越界崩溃 ###########################", flush=True) 
            # 调试信息：检查 tomogram 属性是否已加载
            print(f"[DEBUG] map_dim: {self.planner.map_dim}, center: {self.planner.center}, resolution: {self.planner.resolution}", flush=True)
            if self.planner.map_dim is None or self.planner.center is None or self.planner.resolution is None:
                # 如果还没加载 tomogram，跳过检查
                print("########################### 还没加载 tomogram，跳过边界检查 ###########################", flush=True)
                print("[DEBUG] Tomogram属性未完全加载，跳过边界检查", flush=True)
                self.get_logger().warn("Tomogram not loaded yet, skip bounds check.")
            else:
                print("[DEBUG] 进入边界检查逻辑", flush=True)
                print("########################### 进入边界检查逻辑 ###########################", flush=True)
                dim_x, dim_y = self.planner.map_dim  # [dim_x, dim_y]
                print(f"########################### map_dim: {self.planner.map_dim} ###########################", flush=True)
                center_x, center_y = self.planner.center
                print(f"########################### center: {self.planner.center} ###########################", flush=True)
                resolution = self.planner.resolution
                print(f"########################### resolution: {self.planner.resolution} ###########################", flush=True)
                
                # 地图的实际边界（以 center 为中心，向两侧扩展）
                # 注意：map_dim 是网格数量，实际地图范围是 center ± (map_dim * resolution / 2)
                half_extent_x = dim_x * resolution / 2.0
                half_extent_y = dim_y * resolution / 2.0
                
                min_x = center_x - half_extent_x
                max_x = center_x + half_extent_x
                min_y = center_y - half_extent_y
                max_y = center_y + half_extent_y
                
                # 检查点是否在地图范围内
                print("########################### 检查点是否在地图范围内 ###########################", flush=True)
                start_in_bounds = (min_x <= self.point1_begin[0] <= max_x and
                                  min_y <= self.point1_begin[1] <= max_y)
                goal_in_bounds = (min_x <= self.point2_end[0] <= max_x and
                                 min_y <= self.point2_end[1] <= max_y)
                print("起点在地图边界内吗？start_in_bounds: ", start_in_bounds, flush=True)
                print("终点在地图边界内吗？goal_in_bounds: ", goal_in_bounds, flush=True)

                if not start_in_bounds or not goal_in_bounds:
                    print("########################### 点不在有效范围内 ###########################", flush=True)
                    self.get_logger().warn(
                        f"Interactive start/goal is outside map bounds.\n"
                        f"  Map center: [{center_x:.2f}, {center_y:.2f}]\n"
                        f"  Map bounds: X=[{min_x:.2f}, {max_x:.2f}], Y=[{min_y:.2f}, {max_y:.2f}]\n"
                        f"  Start: [{self.point1_begin[0]:.2f}, {self.point1_begin[1]:.2f}] "
                        f"{'(IN BOUNDS)' if start_in_bounds else '(OUT OF BOUNDS)'}\n"
                        f"  Goal: [{self.point2_end[0]:.2f}, {self.point2_end[1]:.2f}] "
                        f"{'(IN BOUNDS)' if goal_in_bounds else '(OUT OF BOUNDS)'}\n"
                        f"  Please pick points inside the tomogram area."
                    )
                    self.get_logger().warn("规划失败：点不在有效范围内")
                    print("########################### 规划失败：点不在有效范围内 ###########################", flush=True)
                    return
        except Exception as e:
            # 如果检查过程本身出错，不阻塞规划，直接继续使用原逻辑
            self.get_logger().warn(f"Bounds check failed ({e}), continue planning anyway.")

        # 更新当前起终点，然后调用统一的规划函数
        print("########################### 更新当前起终点，然后调用统一的规划函数 ###########################", flush=True)
        self.start_pos_xy = self.point1_begin[:2]  # 只取 x, y
        self.end_pos_xy = self.point2_end[:2]      # 只取 x, y
        
        # 记录 z 坐标用于 slice 计算
        print("########################### 记录 z 坐标用于 slice 计算 ###########################", flush=True)
        self.start_z = self.point1_begin[2]
        self.end_z = self.point2_end[2]
        
        print("########################### 调用统一的规划函数!!!!! self.pct_plan()###########################", flush=True)
        
        # 输出起始点和终点坐标
        start_x, start_y = float(self.start_pos_xy[0]), float(self.start_pos_xy[1])
        start_z = float(self.start_z) if self.start_z is not None else 0.0
        end_x, end_y = float(self.end_pos_xy[0]), float(self.end_pos_xy[1])
        end_z = float(self.end_z) if self.end_z is not None else 0.0
        print(f"########################### 输入self.pct_plan()里的起始点坐标: ({start_x:.2f}, {start_y:.2f}, {start_z:.2f}) ###########################", flush=True)
        print(f"########################### 输入self.pct_plan()里的终点坐标: ({end_x:.2f}, {end_y:.2f}, {end_z:.2f}) ###########################", flush=True)
        
        success, reason = self.pct_plan()
        print(f"########################### self.pct_plan()规划函数返回结果：success: {success}, reason: {reason} ###########################", flush=True)
        # 根据规划结果给出中文提示
        if success:
            print("########################### 规划成功 ###########################", flush=True)
            # 使用当前起点/终点与高度信息生成更详细的提示
            sx, sy = float(self.start_pos_xy[0]), float(self.start_pos_xy[1])
            gx, gy = float(self.end_pos_xy[0]), float(self.end_pos_xy[1])
            sz = float(self.start_z) if self.start_z is not None else 0.0
            gz = float(self.end_z) if self.end_z is not None else 0.0
            print(f"########################### 规划成功，起点坐标: ({sx:.2f}, {sy:.2f}, {sz:.2f}) ###########################", flush=True)
            print(f"########################### 规划成功，终点坐标: ({gx:.2f}, {gy:.2f}, {gz:.2f}) ###########################", flush=True)
            self.get_logger().info(
                f"从 ({sx:.2f}, {sy:.2f}, {sz:.2f}) 点到 ({gx:.2f}, {gy:.2f}, {gz:.2f}) 点规划成功"
            )
        else:
            print("########################### 规划失败 ###########################", flush=True)
            if reason:
                print(f"########################### 规划失败，原因是：{reason} ###########################", flush=True)
                self.get_logger().info(f"规划失败，原因是：{reason}")
            else:
                print("########################### 规划失败，原因是：未知错误，请检查起点和终点是否合理 ###########################", flush=True)
                self.get_logger().info("规划失败，原因是：未知错误，请检查起点和终点是否合理")

    def pct_plan(self):
        # 如果需要查询高度但 tomogram 还未加载，先加载
        if self.planner.elev_g is None and self.planner.elev_g_clean is None:
            self.get_logger().info(f"Loading tomogram: {self.tomo_file}")
            self.planner.loadTomogram(self.tomo_file)
        else:
            self.get_logger().debug("Tomogram already loaded, skipping reload")

        # 如果有 z 坐标，传递给 plan 函数用于计算 slice index
        start_z = getattr(self, 'start_z', None)
        end_z = getattr(self, 'end_z', None)
        
        if start_z is not None or end_z is not None:
            start_z_info = f"z={start_z:.2f}" if start_z is not None else "z=None"
            end_z_info = f"z={end_z:.2f}" if end_z is not None else "z=None"
            self.get_logger().info(
                f"Planning from {self.start_pos_xy} ({start_z_info}) "
                f"to {self.end_pos_xy} ({end_z_info})"
            )
        else:
            self.get_logger().info(f"Planning from {self.start_pos_xy} to {self.end_pos_xy}")
        
        print("########################### 调用308行接口planner.plan()函数进行路径规划 ###########################", flush=True)
        
        start_z_str = f"{start_z:.2f}" if start_z is not None else "None"
        end_z_str = f"{end_z:.2f}" if end_z is not None else "None"
        print(f"########################### 输入的起点坐标和终点坐标xyz：{self.start_pos_xy} (z={start_z_str}) 到 {self.end_pos_xy} (z={end_z_str}) ###########################", flush=True)
        traj_3d = self.planner.plan(self.start_pos_xy, self.end_pos_xy, start_z, end_z)
        
        if traj_3d is not None:
            print(f"########################### planner.plan()函数返回结果：traj_3d: {traj_3d} ###########################", flush=True)
            print(f"########################### 输出的路径的起点坐标和终点坐标xyz：{traj_3d[0]} (z={traj_3d[0][2]:.2f}) 到 {traj_3d[-1]} (z={traj_3d[-1][2]:.2f}) ###########################", flush=True)
        else:
            print("########################### planner.plan()函数返回结果：traj_3d: None ###########################", flush=True)
        success = False
        reason = ""

        if traj_3d is not None:
            traj_3d = np.asarray(traj_3d, dtype=np.float32)
            # 与交互式流程保持一致：成功规划后自动累计并落盘
            self.segment_count += 1
            self.segments.append(traj_3d)
            if self.big_path_traj is None:
                self.big_path_traj = traj_3d
            else:
                self.big_path_traj = np.vstack([self.big_path_traj, traj_3d[1:]])
            self._append_to_files(traj_3d, self.segment_count)

            self.path_pub.publish(traj2ros(traj_3d))
            self.get_logger().info("Trajectory published")
            # 在提示前加一个换行，视觉上更清晰
            print("请输入导航起点：", flush=True)
            self.get_logger().info("\n请输入导航起点：")
            success = True
        else:
            self.get_logger().warn("Failed to generate trajectory")
            reason = "无法生成可行轨迹（起点或终点可能在障碍物内部，或两者之间不存在连通的可通行区域）"

        return success, reason


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = PCTPlanner()
        rclpy.spin(node)
    except ValueError as e:
        print(f"Error initializing node: {e}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
