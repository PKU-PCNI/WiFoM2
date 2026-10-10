# WiFo-M2
H. Zhang, S. Gao, X. Cheng, “WiFo-M2: Empower Wireless Communications With Plug-and-Play Environment Sensing via Foundation Model” IEEE Transactions on Cognitive Communications and Networking, vol. 12, pp. 11654-11669, 2026. 
<br>

## 📢 Latest Updates

🔥🔥🔥 Last Updated on 2026.10.10 🔥🔥🔥 
- [2026.8.4] We released the pre-training weights of the proposed models and part of the testing datasets as open source.
- [2026.10.10] We released the network structure of some downstream PHY modules and the training code of  WiFo-M2 to enhance their performance as a reference. Note that the network structure of downstream PHY modules is reproduced by us and may differ slightly from the original paper, but this does not affect the capability evaluation of WiFo-M2.


## Dependencies and Installation
- Python 3.9 (Recommend to use [Anaconda](https://www.anaconda.com/))
- Pytorch 2.0.0
- NVIDIA GPU + CUDA

## Model Weights
We have released the pre-trained weights of WiFo-M2-Img and WiFo-M2-LiDAR for inference in [Model](https://huggingface.co/AmZPKU/WiFoM2/tree/main).

## Dataset Preparation
The dataset used for inference is available in [[Testing Dataset]](https://huggingface.co/datasets/pku-pcni-lab/Multi-modal_dataset_for_WiFo-M2/tree/main).

## Get Started
By running downstream_PHY_modules_performance.py, the training of a downstream PHY module can be initiated. You can manually choose to add WiFo-M2-Img feature or WiFo-M2-LiDAR feature, or both, to the PHY module to enhance its performance. You can configure it in CFG.

WiFoM2_plus_downstream_PHY_network.py defines the network structure of downstream PHY modules, as well as the loading of weights for WiFo-M2-Img and WiFo-M2-LiDAR models, and the inference of multi-modal features corresponding to CSI at a certain time slot. The operation of downstream_PHY_modules_performance.py depends on WiFoM2_plus_downstream_PHY_network.py.


## Citation
@ARTICLE{zhangWiFoM2,
  author={Zhang, Haotian and Gao, Shijian and Cheng, Xiang},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={WiFo-M2: Empower Wireless Communications With Plug-and-Play Environment Sensing via Foundation Model}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCCN.2026.3717927}}

```
