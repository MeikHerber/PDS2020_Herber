from setuptools import setup

setup(
    name='PDS2020_Herber',
    version='0.0.1',
    description="Programming Data Science",
    author="Meik Herber",
    author_email="meikherber@gmx.de",
    packages=["nextbike"],
    install_requires=['pandas', 'sklearn', 'keras', 'click'],
    entry_points={
        'console_scripts': ['pds=nextbike.models.regression.cli:main']
    }
)
