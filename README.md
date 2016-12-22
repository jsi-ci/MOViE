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
    
    # install numpy
    pip install numpy
    
    # get the project
    git clone https://github.com/jsi-ci/MOViE.git
    cd MOViE/
    git fetch origin
    git checkout -b devel-submodules origin/devel-submodules
    
    # alternatively, if forked
    git clone https://github.com/<username>/MOViE.git
    cd MOViE/
    git remote add upstream https://github.com/jsi-ci/MOViE.git
    git remote show upstream
    git remote show origin
    git checkout -b devel-submodules origin/devel-submodules
    
    # initialize and update included submodules
    git submodule update --init
    
    # install plotly
    cd plotly_py/
    make setup_subs
    pip install -r requirements.txt
    pip install -r optional-requirements.txt
    python setup.py develop

Read more about [submodules], [plotly installation].

[submodules]: https://gist.github.com/gitaarik/8735255
[plotly installation]: https://github.com/plotly/plotly.py/blob/master/contributing.md
