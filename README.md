MOViE
=====

Built on top of [plotly.js], MOViE (Multiobjective Optimization Visualization Environment) is a visualization platform used to visualize data in multiobjective optimization. It includes a range of data preprocessing, exploration, and visualization techniques. It can be used through an online interface or as a module for the Python programming language.

[plotly.js]: https://plot.ly/

## Table of contents

* [Setup](#setup)
* [Testing](#testing)
* [Contributing](#contributing)

Setup
----------

    # install Python (3.4+ comes with Pip, a tool for installing Python packages)
    https://www.python.org/downloads/
    
    # install virtualenv
    pip3 install virtualenv
    
    # create a new virtual Python environment on your machine
    virtualenv —-python=python3 —-system-site-packages movie_venv
    source movie_venv/bin/activate
    
    # install plotly, numpy
    pip install plotly
    pip install numpy
    
    # get the project
    git clone https://github.com/jsi-ci/MOViE.git
    cd MOViE/
    git fetch origin
    git checkout -b devel origin/devel
    
    # alternatively, if forked
    git clone https://github.com/<username>/MOViE.git
    cd MOViE/
    git remote add upstream https://github.com/jsi-ci/MOViE.git
    git remote show upstream
    git remote show origin
    git checkout -b devel origin/devel
    
