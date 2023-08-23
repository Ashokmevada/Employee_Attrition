from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT = '-e .'

def get_requirements(file_path : str) -> List[str]:
    '''
    This function returns a list of requirements
    
    '''

    requirements = []
    with open(file_path, 'r') as f:
        requirements = f.readlines()
        requirements = [ req.replace('/n' , '') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)


setup(
    author='Ashok Mevada',
    author_email='ashokmevada18@gmail.com',
    name='Employee_Attrition',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)