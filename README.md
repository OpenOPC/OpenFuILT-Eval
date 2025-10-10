# OpenFuILT-Eval

This repository contains the official code for full-chip scale OPC evaluation.

Install the OpenFuILT-Eval as a package using

```shell
python setup.py install
```

## Dependency

To use this project, ensure the following dependencies are installed:

1. **Environment**: Python **3.11**
2. **PyTorch**: With CUDA support for GPU acceleration
3. **TorchLitho**: Install from the official repository: [TorchLitho-Lite](https://github.com/OpenOPC/TorchLitho-Lite)
4. **OpenFuILT**: Install from the official repository: [OpenFuILT](https://github.com/OpenOPC/OpenFuILT)

## IO Specifications

- **Pixel Size**:
   The pixel of the mask represents an area of $s \times s \ nm^2$.
- **Mask**:
   A pickle file with a specified pixel size.
- **Target**:
   A GDSII file with a resolution of 1 nm.

**Note**: Ensure that the target and the mask are aligned at the source point $(0, 0)$. This implies:

​	The size of the mask is $[H, W]$, and the size of the target is $[H \times s, W \times s]$.

## Support Metrics

+ Edge Placement Error (EPE)
+ L2 Losss
+ Process Variation Band (PVB)
+ Shot (Optional)

## Example Code

The aligned inputs can be generated using OpenFuILT. 

Below is an example code snippet demonstrating how to evaluate and obtain the metrics results:

```python
from openFuILTEval import Evaluator

evaluator = Evaluator(
    pixel=14,
    macro_size=[2, 2],
    mask_path="mask.pkl",
    target_path="target.gds",
)

evaluator.evaluate()
evaluator.report()
```

You are also encouraged to use your generated full-chip mask and target GDSII files with the evaluation framework for comprehensive validation.

## Citation

If you find this project helpful in your research, please consider citing our paper:

```ini
@inproceedings{yin2024fuilt,
  title={FuILT: Full chip ILT system with boundary healing},
  author={Yin, Shuo and Zhao, Wenqian and Xie, Li and Chen, Hong and Ma, Yuzhe and Ho, Tsung-Yi and Yu, Bei},
  booktitle={Proceedings of the 2024 International Symposium on Physical Design},
  pages={13--20},
  year={2024}
}
```

