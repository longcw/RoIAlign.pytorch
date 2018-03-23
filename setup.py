from setuptools import setup, find_packages

setup(
    name='roi_align',
    version='0.0.1',
    description='PyTorch version of RoIAlign',
    author='Long Chen',
    author_email='longch1024@gmail.com',
    url='https://github.com/longcw/RoIAlign.pytorch',
    install_requires=[
        'cffi',
    ],
    packages=find_packages(exclude=('tests',)),

    package_data={
        'roi_align': [
            '_ext/crop_and_resize/*.so',
        ]
    }
)
