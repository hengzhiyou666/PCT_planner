class ConfigPlanner():
    use_quintic = True
    # 值越小，转向越平滑；过小可能导致窄区域不可达
    max_heading_rate = 1.0
    # 轨迹优化迭代次数，适当增大可减少抖动
    max_optimizer_iterations = 300
    # 轨迹后处理平滑（拉普拉斯）参数
    enable_traj_post_smooth = True
    traj_post_smooth_iterations = 12
    traj_post_smooth_alpha = 0.22


class ConfigWrapper():
    tomo_dir = '/tomogram/'


class Config():
    planner = ConfigPlanner()
    wrapper = ConfigWrapper()