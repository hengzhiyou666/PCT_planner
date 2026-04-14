class ConfigPlanner():
    use_quintic = True
    max_heading_rate = 2.0


class ConfigWrapper():
    tomo_dir = '/tomogram/'


class Config():
    planner = ConfigPlanner()
    wrapper = ConfigWrapper()