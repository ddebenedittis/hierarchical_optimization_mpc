from setuptools import find_packages, setup

package_name = 'hompc_approximator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/data', []),
        ('share/' + package_name + '/nn', []),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Davide De Benedittis',
    maintainer_email='davide.debenedittis@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_generator = hompc_approximator.data_generator:main',
            'nn_regressor = hompc_approximator.nn_regressor:main',
            'nn_test = hompc_approximator.nn_test:main',
        ],
    },
)
