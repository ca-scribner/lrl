lrl: Learn Reinforcement Learning
=================================

See [here](https://lrl.readthedocs.io/en/latest/) for complete documentation.

Overview
--------

lrl is a Python package for applying (and hopefully, learning!) basic Reinforcement Learning algorithms.  It is intended to be an early stepping stone for someone trying to understanding the basic concepts of planning and learning, providing out-of-the-box implementations of some simple environments and algorithms in a well documented, readable, and digestible way to give someone platform from which to build understanding.

The overall goal of the author in writing this package was to provide people interested in Reinforcement Learning a starting point and handrail to help them as they began learning.  The fastest, most efficient implementation is less important here than code which can be read and learned from by someone new to the topic and with intermediate Python skills.

The idea behind this package originated as material to support students completing the Georgia Tech OMSCS class CS 7641: Machine Learning.  CS 7641 takes a top-down approach to learning Machine Learning, whereby students are encouraged to find existing implementations of the algorithms discussed and apply them to gain a feel for their overall properties and pitfalls.  While Python implementations for Supervised Learning are well represented ([scikit-learn](https://scikit-learn.org/)) and a student-driven project for Randomized Optimization and Search ([mlrose](https://mlrose.readthedocs.io/)) is available, robust and well-commented implementations for simple Reinforcement Learning were less ubiquitous.  This package attempts to fill that void.

Installation Instructions
-------------------------

lrl is accessible using pip

```
pip install lrl
```

or, you can pull the source to your working directory so you can play along at home

```
git clone https://github.com/ca-scribner/lrl.git lrl
pip install -e lrl
```