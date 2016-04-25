# omepslid
Script for loading Murphy lab image collections into OMERO.servers.

# Mac

## Installation
1. Install [homebrew](http://brew.sh/).  
`$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`
2. Tap the OMERO homebrew tap.  
`$ brew tap ome/alt`
3. Install python through homebrew. This python will be located in `/usr/local/bin/`.  
`$ brew install python`
4. Install OMERO's version of [Ice](https://zeroc.com/products/ice) through homebrew with python bindings.  
`$ brew install zeroc-ice35 --with-python`
5. Install the required python packages: [numpy](http://www.numpy.org/), [Pillow](http://python-pillow.org/), [javabridge](https://pypi.python.org/pypi/javabridge), and [python-bioformats](https://pypi.python.org/pypi/python-bioformats).  
`$ /usr/local/bin/pip install numpy Pillow javabridge python-bioformats`
6. Download and unzip [OMERO.py](https://downloads.openmicroscopy.org/omero/5.2.2/#py) (link might not be latest version).

## Usage
Before running any of the scripts, you need to set your `PYTHONPATH` environment variable to know about OMERO.py:  
`$ export PYTHONPATH=$PYTHONPATH:/path/to/OMERO.py/lib/python`

Then, you can run any of the scripts using the homebrew python:  
`$ /usr/local/bin/python /path/to/script <arg> ...`
