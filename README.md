# DySpec-U: customized image unmixing algorithm for DySpec
Dynamic Spectra (DySpec) is a spectrum-transforming fluorescent barcode coupled with time-lapse imaging to exponentially increase the number of co-imaged proteins. Theoretically, if *F* fluorophores are imaged in each cycle, *N* cycles of imaging can visualize *F<sup>N</sup>* proteins. To decode proteins from DySpec images, we developed a customized algorithm called `DySpec Unmixing (DySpec-U)`. The detailed decoding principle is introduced in `Supplementary Note 7` of 《Dynamic Spectra: Fluorescence-Transforming Barcodes for High-Dimensional Proteomic Imaging》. The step-by-step operations are outlined below.

### Hardware
- Workstation: DELL T5860
- CPU: Intel Xeon W7-2475X
- GPU: NVIDIA RTX 5090 graphics processing card

### Software
- [PyCharm](https://www.jetbrains.com/zh-cn/pycharm/)
- [elastix](https://elastix.dev/)
- [QuPath](https://qupath.github.io/)
- [Fiji](https://imagej.net/software/fiji/downloads)
- [ZEN](https://www.zeiss.com/microscopy/zh/products/software/zeiss-zen.html)


### Steps to install python environment and libraries:

#### 1. Install PyCharm
- Download PyCharm (2024.3.2) from the [official PyCharm website](https://www.jetbrains.com/zh-cn/pycharm/)

#### 2. Create DySpec-U environment
- In PyCharm software, create a new environment named `DySpec-U` for DySpec

#### 3. Download required libraries
- The required python libraries and versions are listed in `Supplementary Table 10` of 《Dynamic Spectra: Fluorescence-Transforming Barcodes for High-Dimensional Proteomic Imaging》.
- To install `XXX`, execute the following command in your terminal or command prompt:

```bash
pip install XXX
```

#### 4. Verify Installation
After installation, you can verify that the libraries are installed correctly by running this Python script:

```python
import numpy as np
import cv2
import pandas as pd

print(f"numpy version: {np.__version__}")
print(f"cv2 version: {cv2.__version__}")
print(f"pandas version: {pd.__version__}")
```

Successful installation of the libraries can be verified by confirming that the version numbers are displayed without any error messages.

### Run the code
- Execute Steps 1 to 10 sequentially, following the instructions in `Supplementary Note 7` of 《Dynamic Spectra: Fluorescence-Transforming Barcodes for High-Dimensional Proteomic Imaging》.
