from setuptools import setup, find_packages

setup(
    name='be_datahive',
    version='0.1.3',
    packages=find_packages(),
    install_requires=[
        'requests',
        'pandas',
        'numpy',
    ],
    author='Lucas Schneider',
    author_email='lucas.schneider@cs.ox.ac.uk',
    description='A Python wrapper for the BEDATAHIVE API',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Lucas749/be_datahive',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)
