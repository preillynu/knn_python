## cython wrapped CUDA/C++

This code makes an explicit cython class that wraps the C++ class, exposing it in python. It involves a little bit more repitition than the swig code in principle, but
in practice it's MUCH easier.

to install

`$ python setup.py install`

to test:

`$ nosetests`

you need a relatively recent version of cython (>=0.16).



