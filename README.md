# AEFusion: A multi-scale fusion network combining Axial attention and Entropy feature Aggregation for infrared and visible images

Bicao Li, Jiaxi Lu, Zhoufeng Liu, Zhuhong Shao, Chunlei Li, Yifan Du, Jie Huang 

## [paper](https://www.sciencedirect.com/science/article/abs/pii/S1568494622009061)

## Platform
Python =3.6  
Pytorch =1.5.0  
scipy =1.2.0

## Training Dataset

[MS-COCO 2014](http://images.cocodataset.org/zips/train2014.zip) (T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, and C. L. Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014. 3-5.) is utilized to train our network.


## Train and Test
    train_network.py
    test.py

### Tips:

The evaluation metrics in the paper can be found [here](https://github.com/thfylsty/Objective-evaluation-for-image-fusion).

# Citation

```
@article{LI2023109857,
author = {Bicao Li, Jiaxi Lu, Zhoufeng Liu, Zhuhong Shao, Chunlei Li, Yifan Du and Jie Huang},
title = {AEFusion: A multi-scale fusion network combining Axial attention and Entropy feature Aggregation for infrared and visible images},
journal = {Applied Soft Computing},
volume = {132},
pages = {109857},
year = {2023},
issn = {1568-4946},
doi = {https://doi.org/10.1016/j.asoc.2022.109857},
url = {https://www.sciencedirect.com/science/article/pii/S1568494622009061}
```
If you have any question, please email to us (lbc@zut.edu.cn or lujiaxi@zut.edu.cn).

