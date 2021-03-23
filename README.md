# Grasping in 6DoF: An Orthographic Approach to Generalized Grasp Affordance Predictions

![Orthographic pipeline](https://github.com/m-rios/msc-thesis/blob/master/data/results/Plots/pipeline2.png)

Grasp detection research focuses at the moment on finnding neural networks that given a
RGB-D image or point cloud, yield a parametric grasp description that can be used to
firmly grip target objects. There is a need for these models to be small and efficient, such
that they can be used in embedded hardware. Furthermore these models tend to only
work for top-down views, which highly restrict the ways objects can be grasped. In this
work, we focus on improving an existing shallow network, GG-CNN, and propose a new
orthographic pipeline to enable the use of these models independently of the orientation
of the camera.

## Getting started

After clonning this repository run `git submodule init && git submodule update`
to download the ggcnn dependency.

Next install the dependencies and setup a virtualenv using `make {cpu|gpu}`
depending on whether you have a dedicated graphics cardcompatible with tensorflow or not.

This codebase was developed and tested on python2.7

## Paper
Available at [link](https://www.ai.rug.nl/oel/papers/extending_GG-CNN_OEL.pdf)

![paper](https://github.com/m-rios/msc-thesis/blob/master/paper.png)

Please adequately refer to the papers any time this code is being used. If you do publish a paper where MORE helped your research, we encourage you to cite the following paper in your publications:
```
@article{munozextending,
  title={Extending GG-CNN through Automated Model Space Exploration using Knowledge Transfer},
  author={Mu{\~n}oz, Mario R{\'\i}os and Schomaker, Lambert and Kasaei, S Hamidreza}
}
```

## Authors
Mario Ríos Muñoz and [Hamidreza Kasaei](https://hkasaei.github.io/)

Work done while at [RUG](https://www.rug.nl/).
