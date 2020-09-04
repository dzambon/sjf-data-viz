from setuptools import setup
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='sjf_data_viz',  # Required
    version='0.1.0',  # Required
    description='SJF Fascinating Informatics -- Data visualization project',  # Optional
    long_description=read('README.md'),
    url='https://github.com/dzambon/sjf-data-viz',  # Optional
    author='D. Zambon',
    author_email='daniele.zambon@usi.ch',  # Optional
    packages=['sjf_data_viz'],
    install_requires=['matplotlib', 'sklearn', 'umap-learn', 'spotipy']
)

