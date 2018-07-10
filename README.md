# IMEDNet: Image-to-Motion Encoder-Decoder Network(s)

Neural networks for transforming MNIST digit images into dynamic movement primitives (DMPs).

## Install

### Cloning

The repository uses [Git LFS](https://github.com/git-lfs/git-lfs) to manage large data files and models.

To clone the repository without data and model files, run:
```bash
$ GIT_LFS_SKIP_SMUDGE=1 git clone git@repo.ijs.si:deep_learning/imednet.git
```

Afterwards, to pull the data and model files, run:
```bash
$ cd imednet
$ git lfs pull
```

### Prerequisites:

* [pytorch](http://pytorch.org/)
* [torchvision](https://github.com/pytorch/vision)
* [numpy](http://www.numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [python-mnist](https://github.com/sorki/python-mnist)
