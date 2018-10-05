# kmod

[![Build Status](https://travis-ci.com/wittawatj/kmod.svg?token=yWUaYGwontVUwf9G8fLY&branch=master)](https://travis-ci.com/wittawatj/kmod)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/wittawatj/kmod/blob/master/LICENSE)

A kernel test for model comparison

* Developed on Python 3.6.3 (installed with Anaconda). Some effort will be made
  to support Python 2.7.  Collaborators of this project should use Python 3.6+.

* Depends on the `kgof` package. This can be obtained from [its git
  repository](https://github.com/wittawatj/kernel-gof). Will need the Python 3
  version of the package. Collaborators should install `kgof` from its
  development [repo](https://github.com/wittawatj/kgof) which is private.
  Contact Wittawat. See the instruction below.

* Depends on the `freqopttest` (containing the UME two-sample test) package
  from  [its git repository](https://github.com/wittawatj/interpretable-test).
  You will need the Python 3 version of the package. Again, collaborators should install `freqopttest` from its development [repo](https://github.com/wittawatj/fotest).

 ## Install `kgof` and `freqopttest`

 If you are a coder/collaborator of this project, you should do the followings.

 1. Contact Wittawat to add you to the two Github private repositories.
 2. Clone the two repositories. You will get two folders, say, at
    `/path/to/kgof` and `/path/to/freqopttest`.
 3. Install the two packages with

        pip install -e /path/to/kgof
        pip install -e /path/to/freqopttest

    Make sure to install them in a Python 3 environment.

4. In Python, make sure you can `import freqopttest` and `import kgof` without
   any error.

In total, there will be three repositories. During development make sure to
`git pull` all of them often.

## Coding guideline

* Use `autograd.numpy` instead of `numpy`. Part of the code relies on
  `autograd` to do automatic differentiation. Also use `np.dot(X, Y)` instead
  of `X.dot(Y)`. `autograd` cannot differentiate the latter. Also, do not use
  `x += ...`.  Use `x = x + ..` instead.

## Other repos

* [https://github.com/janesjanes/GAN_training_code](https://github.com/janesjanes/GAN_training_code)

## Sharing resource files 

Generally it is not a good idea to push large files (e.g., trained GAN models)
to this repository. Since git maintains all the history, the size of the
repository can get large quickly. For sharing non-text-file resources (e.g.,
GAN models, a large collection of sample images), we will use Google Drive.
I recommend a command-line client called `drive` which can be found
[here](https://github.com/odeke-em/drive). If you use a commonly used Linux
distribution, see [this
page](https://github.com/odeke-em/drive/blob/master/platform_packages.md) for
installation instructions.
If you cannot use `drive` or prefer not to, you can use other clients that you
like (for instance, [the official
client](https://www.google.com/drive/download/)). You can also just go with
manual downloading of all the shared files from
the web, and saving them to your local directory. A drawback of this approach
is that, when we update some files, you will need to manually update them. With
the `drive` client, you simply run `drive pull` to get the latest update
(`drive` does not automatically synchronize in realtime. Other clients might.). 

To have access to our shared Google Drive folder:

1. Ask Wittawat to share with you the folder on Google Drive. You will need a
   Google account. Make sure you have a write access so you can push your
   files. Once shared, on [your Google Drive page](https://drive.google.com),
   you should see a folder called `kmod_share` on the "Shared with me" tab.
   This folder contains all the resource files (not source code) related to
   this project. Move it to your drive so that you can sync later by right
   clicking, and selecting "Add to my drive".


2. On your local machine, create a parent folder anywhere to contain all
   contents on your Google Drive (e.g., `~/Gdrive/`). We will refer to this
   folder as `Gdrive/`. Assume that you use the `drive` client. `cd` to this
   folder and run `drive init` to mark this folder as the root folder for your
   Google Drive.
   
   To get the contents in `kmod_share`: 
   
   1. Create a subfolder `Gdrive/kmod_share/`.
   2. `cd` to this subfolder and run `drive pull`. This will pull all contents 
   from the remote `kmod_share` folder to your local folder.

3. In `kmod.config.py`, modify the value of `shared_resource_path`
   key to point to your local folder `Gdrive/kmod_share/`. 

* Make sure to do `drive pull` often to get the latest update.

* After you make changes or add files, run `drive push` under `kmod_share`
   to push the contents to the remote folder for other collaborators to see.

        


