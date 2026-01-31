import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped

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

        # 交互式起点 / 终点（来自 RViz Publish Point）
        self.point1_begin = None  # 起点坐标 [x, y, z]
        self.point2_end = None    # 终点坐标 [x, y, z]
        self.click_count = 0      # 点击计数器

        # 订阅 RViz 的 Publish Point 按钮
        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            "/clicked_point",
            self.clicked_point_callback,
            10,
        )
        
        # 启动时先按默认起点/终点规划一次
        # 规划完成后会在 pct_plan() 中显示"请输入导航起点："
        self.pct_plan()

    def configure_scene(self, scene_name):
        if scene_name == 'Spiral':
            self.tomo_file = 'spiral0.3_2'
            self.start_pos_xy = np.array([-16.0, -6.0], dtype=np.float32)
            self.end_pos_xy = np.array([-26.0, -5.0], dtype=np.float32)
        elif scene_name == 'Building':
            self.tomo_file = 'scans1_399MB'
            print("########################### 选择Building场景名称，并设置起点和终点坐标 ###########################", flush=True)
            self.start_pos_xy = np.array([5.0, 5.0], dtype=np.float32)
            self.end_pos_xy = np.array([-6.0, -1.0], dtype=np.float32)
        elif scene_name == 'Plaza':
            self.tomo_file = 'plaza3_10'
            self.start_pos_xy = np.array([0.0, 0.0], dtype=np.float32)
            self.end_pos_xy = np.array([23.0, 10.0], dtype=np.float32)
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
        
        # 更新点击计数
        self.click_count += 1
        
        # 根据点击次数决定是起点还是终点
        if self.click_count % 2 == 1:  # 奇数次：更新起点
            print("########################### 得到起点坐标 ###########################", flush=True)
            self.point1_begin = np.array([x, y, z], dtype=np.float32)
            self.get_logger().info(f"导航起点坐标为：({x:.2f}, {y:.2f}, {z:.2f})")
            self.get_logger().info("请输入目的地位置：")
        else:  # 偶数次：更新终点并规划
            print("########################### 得到目标点坐标 ###########################", flush=True)
            self.point2_end = np.array([x, y, z], dtype=np.float32)
            self.get_logger().info(f"目的地坐标为：({x:.2f}, {y:.2f}, {z:.2f})")
            self.get_logger().info("下面进行路径规划：")
            
            # 进行路径规划
            self.plan_with_points()

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
