import os
from setuptools import setup, find_packages

# def _process_requirements():
#     packages = open('requirements.txt').read().strip().split('\n')
#     requires = []
#     for pkg in packages:
#         if pkg.startswith('git+ssh'):
#             return_code = os.system('pip install {}'.format(pkg))
#             assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
#         else:
#             requires.append(pkg)
#     return requires

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="cbpy",
    version="0.0.1",
    author="Peng Tao",
    author_email="taopeng543@gmail.com",
    url='https://github.com/PengTao-HUST/CBP',
    description="Implement Chaotic Backpropagation Algorithm",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'nolds',
        'statsmodels',
        'pandas',
        'torchvision',
        'tqdm',
        'seaborn',
        'torch',
        'pytest',
        'matplotlib',
        'numpy'
    ],
    license="MIT Licence",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ]
)
