import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tomography'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # 安装 Python 包内的配置文件（如 config/lidar_filter.yaml）
    package_data={
        'tomography': [
            'config/*.yaml',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('lib/' + package_name, ['scripts/tomography_node']),
    ],
    install_requires=[
        'setuptools',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='alex',
    maintainer_email='1523924956@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'tomography_node = tomography.tomography_node:main',
    ],
},
)
