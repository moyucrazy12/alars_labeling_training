import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'alars_labeling_training'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            os.path.join('share', package_name),
            ['package.xml'],
        ),
        # install trained models
        (
            os.path.join('share', package_name, 'trained_models'),
            glob('trained_models/*'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Cristhian',
    maintainer_email='ckmc@kth.se',
    description='ROS 2 package (no executables) for ALARS perception: dataset labeling, YOLO OBB training, and model management',
    license='MIT',
    tests_require=['pytest'],
)