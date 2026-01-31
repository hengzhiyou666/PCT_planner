# OSQP 路径配置：优先使用环境变量或 CMake 参数，否则尝试系统默认路径
if(DEFINED ENV{OSQP_DIR})
    set(osqp_DIR "$ENV{OSQP_DIR}")
elseif(NOT osqp_DIR)
    # 尝试 ROS2 Humble 的 OSQP 安装路径
    if(EXISTS "/opt/ros/humble/lib/cmake/osqp")
        set(osqp_DIR "/opt/ros/humble/lib/cmake/osqp")
    else()
        # 尝试其他常见路径
        set(osqp_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rdparty/osqp-1.0.0/lib/cmake/osqp")
    endif()
endif()

find_package(osqp REQUIRED)

add_library(common_smoothing SHARED
    affine_constraint.cc
    discrete_points_math.cc
    osqp_spline1d_solver.cc
    osqp_spline2d_solver.cc
    polynomialxd.cc
    spline1d_constraint.cc
    spline1d_kernel_helper.cc
    spline1d_kernel.cc
    spline1d_seg.cc
    spline1d.cc
    spline2d_constraint.cc
    spline2d_kernel.cc
    spline2d_seg.cc
    spline2d.cc
    solver/osqp/osqp_interface.cc
    utils/math.cc
)
# OSQP 头文件和库文件配置
if(TARGET osqp::osqpstatic)
    # 如果找到了 osqp 目标，使用它提供的 include 路径和链接
    target_link_libraries(common_smoothing osqp::osqpstatic)
elseif(TARGET osqp::osqp)
    # 尝试使用 osqp::osqp 目标
    target_link_libraries(common_smoothing osqp::osqp)
else()
    # 否则手动设置路径（ROS2 Humble 的 OSQP）
    if(EXISTS "/opt/ros/humble/include/osqp")
        target_include_directories(common_smoothing PUBLIC "/opt/ros/humble/include")
    else()
        # 从 osqp_DIR 推断
        get_filename_component(OSQP_ROOT_DIR "${osqp_DIR}" DIRECTORY)
        get_filename_component(OSQP_ROOT_DIR "${OSQP_ROOT_DIR}" DIRECTORY)
        target_include_directories(common_smoothing PUBLIC "${OSQP_ROOT_DIR}/include")
    endif()
    # 链接库文件
    if(EXISTS "/opt/ros/humble/lib/libosqp.a")
        target_link_libraries(common_smoothing "/opt/ros/humble/lib/libosqp.a")
    elseif(EXISTS "/opt/ros/humble/lib/libosqp.so")
        target_link_libraries(common_smoothing "/opt/ros/humble/lib/libosqp.so")
    else()
        message(FATAL_ERROR "Cannot find OSQP library file")
    endif()
endif()