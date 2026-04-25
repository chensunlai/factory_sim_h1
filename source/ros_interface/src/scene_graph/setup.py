from setuptools import find_packages, setup

package_name = 'scene_graph'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    package_data={
        package_name: ['models/*.pt', 'models/*.bin', 'models/*.safetensors'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'mcp',
        'starlette',
        'numpy',
        'torch',
        'open3d',
        'faiss-cpu',
        'ultralytics',
        'open_clip_torch',
        'pillow',
        'opencv-python',
    ],
    zip_safe=True,
    maintainer='chen',
    maintainer_email='chensunlai2004@gmail.com',
    description='Incremental scene graph mapping and MCP server for ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scene_graph_node = scene_graph.scene_graph_mcp:main',
            'scene_graph_mcp = scene_graph.scene_graph_mcp:main',
        ],
    },
)
