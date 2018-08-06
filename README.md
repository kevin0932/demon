# Thesis Submission: README
This is the modified DeMoN version for my master thesis "Global Structure-from-Motion from Learned Relative Poses", at CVG, ETH Zurich.

It is based on three major frameworks (forked repos can be found via links attached and please use the branches named "thesis_submission"):
  1. DeMoN (https://github.com/kevin0932/demon.git)
  2. COLMAP (https://github.com/kevin0932/colmap.git)
  3. TheiaSfM (https://github.com/kevin0932/TheiaSfM.git)

Please checkout the branch "thesis_submission" and find the code/scripts for thesis submission. (The rest branches are just experimental/prototype scripts).

Major scripts for the thesis pipeline and experiments can be found in the folder "thesis_scripts" of demon (separate README files for each folder):
  1. Folder "framework" includes all the code for the thesis pipeline and please find the complete pipeline script at the file "globalSfM_from_DeMoN_complete_pipeline.sh" (with descriptions on other scripts);
  2. Folder "experiment_scripts_for_ETH3D" includes some scripts for experiments with ETH3D datasets;
  3. Folder "experiments" includes the preliminary experiments that were done in the beginning of the thesis, mainly including the trials on DeMoN learnt depth/flow.

Please ignore the duplicated scripts in other folders if there are any.

Feel free to drop a line to the email: [kevin.qixuan.zhangATgmail.com]() if more explanations of the scripts are needed.

---
# Data Formats and Compatibility

## DeMoN
As described in the paper, DeMoN adopts 6-parameter representation for relative motion predictions, i.e., 3-element angle-axis representation for relative rotation, and x-,y-,z-values for relative translations. Such relative pose representation corresponds to the extrinsic parameters in the camera matrices, which should be the same with COLMAP. For angle-axis representations, there are codes in the scripts that do the conversion to other formats.

The inputs to DeMoN are images of any types but their size should be adapted to 256*192 with the intrinsic adjustment process. The outputs of DeMoN are defaultly stored in HDF5 files. Be aware that the depth prediction has already scaled with DeMoN estimated scale factors and DeMoN defaultly does not store/output the scale factor predictions. Here, the code has been adapted so that those DeMoN estimated scale factors are outputed!

## COLMAP
The camera poses are stored as extrinsic parameters in COLMAP, which is different from TheiaSfM. The Main data formats include the .db database files that may store extracted features, (inlier) matches, camera poses, and the text/binary files storing the intermediate/optimized results (camera poses, text files for pose comparison or geo-registration).

## TheiaSfM
The relative poses of the 2nd camera are stored a bit differently than those in DeMoN and COLMAP. Here the camera position of the 2nd camera in the coordinate of the 1st reference camera is used, instead of the extrinsic parameters. Also, there is an important data storage file, i.e., the matchfile that may serve as the start of the TheiaSfM pipeline and stores the matching information pre-computed. Such matchfiles are serialized with [Cereal library](https://uscilab.github.io/cereal/), so the corresponding library should be used when you produce or read from the matchfiles.

---
# FAQs (for the problems I met during my thesis, mainly on the installation or setups)
## Camera Pose Representation Difference
Be careful with the camera poses produced by DeMoN, COLMAP and TheiaSfM, since they may store them in different formats (either poses or extrinsic parameters). Conversions may be required. Here in the experiments done in this thesis, DeMoN and COLMAP use extrinsic parameters, while TheiaSfM adopts the 2nd camera position in the coordinate of the reference camera as the relative poses.

## DeMoN Intrinsic Adjustment
Details are described in the thesis as well as the script coming with DeMoN. There is also [an closed GitHub issue](https://github.com/lmb-freiburg/demon/issues/15) discussing the process.

## Library Configuration on Ubuntu 16
For different libraries' configuration on the same Ubuntu 16 machine, different versions of GCC/G++ compilters and CMAKE tools may be required for their incompatibility and update history!

For example, the tensorflow library (I used v1.4 or v1.3) may be dealt with if you use the default GCC5 compilter, by setting corresponding configuration flags to disable incompatible CXX11_ABI version usage (--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" ). See details in the official tutorial.

Besides, for the dependency library lmbspecialops of DeMoN, it should be compiled with GCC 4 and later cmake version (e.g. >=3.2) with easiest effort. Otherwise, the CXX11_ABI should be dealt with if you use GCC 5.

### GCC installation

A general GCC compiler setup can be found in the following script! Usually GCC 4.8 or 4.9 can be stable and compatible with most libraries!
https://gist.github.com/jtilly/2827af06e331e8e6b53c


### CMAKE installation

Building and installing CMake
The easiest way to install CMake is from source. Head over to the CMake downloads page (https://cmake.org/download/) and get the latest “Unix/Linux Source” *.tar.gz file. Make sure to set the --prefix flag correctly, otherwise you won’t have permissions to install files at the default location.

    $ tar -xf cmake*.tar.gz
    $ cd cmake*
    $ ./configure --prefix=$HOME
    $ make
    $ make install

You should now have the most up-to-date installation of cmake. Check the version by typing:

    $ cmake --version


### Building and installing HDF5

Download the latest ‘configure’ version of HDF5 (https://support.hdfgroup.org/HDF5/release/obtainsrc.html#conf). After extacting the tarball run configure with the following options:

    $ ./configure --disable-parallel --without-szlib --without-pthread --prefix=$HOME

Be careful with the option "--disable-parallel", since sometimes you do need a parallel version of HDF5, which may be required when you want to setup some libraries with support of OpenMP/OpenMPI!     

Compile and install:

    $ make
    $ make install

All of the important HDF5 tools will be at $HOME/bin and libraries/include files at: $HOME/lib and $HOME/include.

### VTK building with Python 3 wrapper

Building VTK with Python3 Wrappers http://ghoshbishakh.github.io/blog/blogpost/2016/07/13/building-vtk-with-python3-wrappers.html.

However, to save your effort and clean setup for different projects, you may use a virtualenv tool in python or anaconda to manage all your dependencies for each project. In my master thesis, I use anaconda with tensorflow and create an venv with python 3 in my workspace. then I can easily install VTK with python3 interface with an existing conda receipe (e.g. https://github.com/menpo/conda-vtk). Just find info here (https://github.com/lmb-freiburg/demon/issues/6) and follow the commands in https://stackoverflow.com/questions/43184009/install-vtk-with-anaconda-3-6.

    # install vtk 7.0 with python 3.5 wrapper in the virtual env "envB". Just change it to the venv you are using!
    $ conda install -n envB -c menpo vtk=7 python=3.5


Some sample commands to install VTK from source with Python3 interface:

    $ cd ~/inst/src
    $ git clone https://github.com/Kitware/VTK.git
    $ mkdir build-VTK
    $ cd build-VTK
    $ ccmake -D CMAKE_BUILD_TYPE=Release \
             -D CMAKE_INSTALL_PREFIX=~/inst \
             -D VTK_WRAP_PYTHON=ON \
             -D VTK_PYTHON_VERSION=3 \
             -D VTK_INSTALL_PYTHON_MODULE_DIR=~/VIRT_ENV/lib/python3.5/site-packages \
             -D PYTHON_EXECUTABLE=~/VIRT_ENV/bin/python3.5 \
             ../VTK

---
# The rest is the original README content from DeMoN

---

# DeMoN: Depth and Motion Network

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

DeMoN is "A computer algorithm for reconstructing a scene from two projections"<sup>1</sup>.
The network estimates the depth and relative camera motion for pairs of images; it addresses the important two view case in structure from motion.

![Teaser](teaser.png)

If you use this code for research please cite:

    @InProceedings{UZUMIDB17,
      author       = "B. Ummenhofer and H. Zhou and J. Uhrig and N. Mayer and E. Ilg and A. Dosovitskiy and T. Brox",
      title        = "DeMoN: Depth and Motion Network for Learning Monocular Stereo",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      month        = " ",
      year         = "2017",
      url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/UZUMIDB17"
    }

See the [project website](https://lmb.informatik.uni-freiburg.de/people/ummenhof/depthmotionnet) for the paper and other material.

<sup>1 This is the title of H. C. Longuet-Higgins paper from 1981, which perfectly describes what our method does. DeMoN shows that complex geometric relations can be learnt by a ConvNet.</sup>

## Requirements

Building and using requires the following libraries and programs

    tensorflow 1.4.0
    cmake 3.7.1
    python 3.5
    cuda 8.0.61 (required for gpu support)
    VTK 7.1 with python3 interface (required for visualizing point clouds)

The versions match the configuration we have tested on an ubuntu 16.04 system.
DeMoN can work with other versions of the aforementioned dependencies, e.g. tensorflow 1.3, but this is not well tested.

The binary package from [vtk.org](http://www.vtk.org) does not come with a python3 interface.
To enable python3 support VTK needs to be built from source.
Alternatively, there are also VTK packages with python3 support available in Anaconda via the conda package manager.

The network also depends on our [lmbspecialops](https://github.com/lmb-freiburg/lmbspecialops) library which is included as a submodule.



## Build instructions

The following describes how to install tensorflow and demon into a new virtualenv and run the inference example.
We will use ```pew``` (```pip3 install pew```) to manage a new virtualenv named ```demon_venv``` in the following:

```bash
# create virtualenv
pew new demon_venv
```

The following commands all run inside the virtualenv:

```bash
# install python module dependencies
pip3 install tensorflow-gpu # or 'tensorflow' without gpu support
pip3 install pillow # for reading images
pip3 install matplotlib # required for visualizing depth maps
pip3 install Cython # required for visualizing point clouds
```

```bash
# clone repo with submodules
git clone --recursive https://github.com/lmb-freiburg/demon.git

# build lmbspecialops
DEMON_DIR=$PWD/demon
mkdir $DEMON_DIR/lmbspecialops/build
cd $DEMON_DIR/lmbspecialops/build
cmake .. # add '-DBUILD_WITH_CUDA=OFF' to build without gpu support
# (optional) run 'ccmake .' here to adjust settings for gpu code generation
make
pew add $DEMON_DIR/lmbspecialops/python # add to python path

# download weights
cd $DEMON_DIR/weights
./download_weights.sh

# run example
cd $DEMON_DIR/examples
python3 example.py # opens a window with the depth map (and the point cloud if vtk is available)
```

## Data reader op & evaluation

The data reader op and the evaluation code have additional dependencies.
The code for the data reader is in the ```multivih5datareaderop``` directory.
See the corresponding [readme](multivih5datareaderop/README.md) for more details.

For the evaluation see the example [```examples/evaluation.py```](examples/evaluation.py).
The evaluation code requires the following additional python3 packages, which can be installed with ```pip```:

```
h5py
minieigen
pandas
scipy
scikit-image
xarray
```
Note that the evaluation code also depends on the data reader op.



## License

DeMoN is under the [GNU General Public License v3.0](LICENSE.txt)
