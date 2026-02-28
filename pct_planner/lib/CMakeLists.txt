cmake_minimum_required(VERSION 3.13.0)
project(planning_core CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# 不要在这里设置 CMAKE_BUILD_TYPE，让父级 CMakeLists.txt 控制
# 这样可以避免 ament 安装系统生成错误的文件名

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
# set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -march=native") # have to enable because of gtsam

# 仅使用 ROS Humble 的 GTSAM，避免与 /usr/local 安装的 GTSAM 4.3 混用导致头文件重定义
set(GTSAM_DIR "/opt/ros/humble/lib/cmake/GTSAM")
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

# 编译本库时不要搜索 /usr/local 下的 GTSAM，避免与 ROS 的 GTSAM 混用
list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/usr/local/include")

# 确保只使用 ROS 的 GTSAM 头文件：先加入 GTSAM 路径，避免 /usr/local 的 GTSAM 被优先包含
include_directories(BEFORE ${GTSAM_INCLUDE_DIR})
include_directories(
  src
  ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
)

find_package(pybind11_vendor REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_subdirectory(src/common)
add_subdirectory(src/a_star)
add_subdirectory(src/trajectory_optimization)
add_subdirectory(src/map_manager)
add_subdirectory(src/ele_planner)

# Install pybind11 modules to the python package directory
install(TARGETS 
  a_star 
  traj_opt 
  ele_planner 
  py_map_manager
  a_star_search
  gpmp_optimizer
  map_manager
  ele_planner_lib
  common_smoothing
  DESTINATION ${PYTHON_INSTALL_DIR}/pct_planner/lib
)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py "")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py DESTINATION ${PYTHON_INSTALL_DIR}/pct_planner/lib)

# Install python stubs
install(FILES
  src/a_star/a_star.pyi
  src/trajectory_optimization/traj_opt.pyi
  src/ele_planner/ele_planner.pyi
  src/map_manager/py_map_manager.pyi
  DESTINATION ${PYTHON_INSTALL_DIR}/pct_planner/lib
  OPTIONAL
)
