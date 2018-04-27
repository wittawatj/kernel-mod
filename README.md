# kmod

[![Build Status](https://travis-ci.com/wittawatj/kmod.svg?token=yWUaYGwontVUwf9G8fLY&branch=master)](https://travis-ci.com/wittawatj/kmod)

A kernel test for model comparison

* Developed on Python 3.6.3. Some effort will be made to support Python 2.7. 
    Collaborators of this project should use Python 3.6+.

* Depends on the `kgof` package. This can be obtained from [its git
  repository](https://github.com/wittawatj/kernel-gof). Will need the Python 3
  version of the package. Collaborators should install `kgof` from its
  development [repo](https://github.com/wittawatj/kgof) which is private.
  Contact Wittawat. See the instruction below.

* Depends on the `freqopttest` (containing the UME two-sample test) package
  from  [its git repository](https://github.com/wittawatj/interpretable-test).
  You will need the Python 3 version of the package. Again, collaborators should install `freqopttest` from its development [repo](https://github.com/wittawatj/fotest).

 ## Install `kgof` and `freqopttest`

 If you are a collaborator of this project, you should do the followings.

 1. Contact Wittawat to add you to the two Github private repositories.
 2. Clone the two repositories. You will get two folders, say, at
    `/path/to/kgof` and `/path/to/freqopttest`.
 3. Install the two packages with
        pip install -e /path/to/kgof
        pip install -e /path/to/freqopttest

    Make sure to install them in a Python 3 environment.

4. In Python, make sure you can `import freqopttest` and `import kgof` without
   any error.

