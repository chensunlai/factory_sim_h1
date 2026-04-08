from setuptools import find_packages, setup

package_name = 'mcp_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    scripts=['scripts/network_server'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'mcp'],
    zip_safe=True,
    maintainer='chen',
    maintainer_email='chensunlai2004@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
)
