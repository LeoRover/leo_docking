from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=["leo_docking", "leo_docking.states"], package_dir={"": "src"}
)

setup(**setup_args)
