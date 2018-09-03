# SSR
structured sparsity regularization

we propose a novel filter pruning scheme, termed structured sparsity regularization (SSR), to simultaneously speedup the computation and reduce the memory overhead of CNN, which can be well supported by various off-the-shelf deep learning libraries.

## Running

#### 1. download dataset (mnist)
```
python dataset/download_and_convert_mnist.py 
```

#### 2. training and testing
```
./run.sh
```

## Experimental results

| Method | #Filter/Node | FLOPs | #Param. | CPU(ms) | Speedup | Top-1 Err.↑ |
|----------|----------|----------|----------|----------|----------|----------|
| LeNet | 20-50-500 | 2.3M | 0.43M | 26.4 | 1× | 0% |
| SSL[23] | 3-15-175 | 162K | 45K | 7.3 | 3.62× | 0.05% |
| SSL[23] | 2-11-134 | 91K | 26K | 6.0 | 4.40× | 0.20% |
| TE[42] | 2-12-127 | 95K | 27K | 5.7 | 4.62× | 0.02% |
| TE[42] | 2-7-99 | 65K | 13K | 5.5 | 4.80× | 0.20% |
| CGES[57] | - | 332K | 156K | - | - | 0.01% |
| CGES+[57] | - | - | 43K | - | - | 0.04% |
| GSS[43] | 3-11-109 | 119K | 21K | 6.7 | 3.94× | 0.08% |
| GSS[43] | 3-8-82 | 95K | 12K | 5.6 | 4.71× | 0.20% |
| SSR-L2,1 | 3-11-108 | 118K | 21K | 6.6 | 4.00× | 0.05% |
| SSR-L2,1 | 2-8-77 | 67K | 11K | 4.8 | 5.50× | 0.18% |

## Note
>[23] W. Wen, C. Wu, Y. Wang, et al. Learning structured sparsity in deep neural networks. In NIPS, 2016.

>[42] P. Molchanov, S. Tyree, T. Karras, et al. Pruning convolutional neural networks for resource efficient inference. In ICLR, 2017.

>[43] A. Torfi and R. A. Shirvani. Attention-based guided structured sparsity of deep neural networks. arXiv preprint arXiv:1802.09902, 2018. 

>[57] J. Yoon and S. J. Hwang. Combined group and exclusive sparsity for deep neural networks. In ICML, 2017.
