from setuptools import find_packages, setup

package_name = 'replay'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chen',
    maintainer_email='chensunlai2004@gmail.com',
    description='goal point replay node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'replay_node = replay.replay_node:main',
        ],
    },
)
