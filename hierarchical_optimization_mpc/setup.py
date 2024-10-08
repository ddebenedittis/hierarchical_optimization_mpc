from setuptools import setup

package_name = 'hierarchical_optimization_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Davide De Benedittis',
    maintainer_email='davide.debenedittis@gmail.com',
    description='Hierarchical Optimization Model Predictive Control',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'example_single_robot = hierarchical_optimization_mpc.example_single_robot:main',
            'example_multi_robot = hierarchical_optimization_mpc.example_multi_robot:main',
            'test_solve_times = hierarchical_optimization_mpc.test_solve_times:main',
            'toy_problem_1 = hierarchical_optimization_mpc.toy_problem_1:main',
            'toy_problem_2 = hierarchical_optimization_mpc.toy_problem_2:main',
        ],
    },
)
