Seminar Knowledge Mining
========================

Set up instructions
-------------------

### Unix

1. Install these dependencies by using your system's package manager if you
don't have them already.

    | Depdendency |   Debian package   |    Arch package   |
    | ----------- | ------------------ | ----------------- |
    | Python 3    | python3            | python            |
    | Pip         | python3-pip        | python-pip        |
    | Virtualenv  | python3-virtualenv | python-virtualenv |
    | Cython      | cython3            | cython            |
    | Fortran     | gfortran           | gcc-fortran       |
    | Blas        | libblas-dev        | blas              |
    | Lapack      | liblapack-dev      | lapack            |

2. Create a virtual environment inside the repository root by runnning
`virtualenv .` or if you have multiple Python versions `virtualenv -p python3
.`.
3. Activate your virtual environment using `source bin/activate`. Make sure
that the repository name is in front of your shell promt now.
4. Install dependencies inside your virtual environment using `pip install -r
requirements.txt`.

### Windows

1. Create a virtual environment inside the repository root by runnning
`virtualenv .` or if you have multiple Python versions `virtualenv -p
C:\Python34\python.exe .`.
2. Activate your virtual environment using `Scripts\activate`. Make sure that
the repository name is in front of your shell promt now.
3. Download these dependencies. If in doubt, use the link before the last in
each list. Run `pip install <path-to-file>` on each of those.

    - [NumPy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy)
    - [SciPy](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy)
    - [Scikit-learn](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-learn)
    - [Scikit-image](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-image)

4. Install remaining dependencies inside your virtual environment using `pip
install -r requirements.txt`.
