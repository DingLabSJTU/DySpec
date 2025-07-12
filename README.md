# Customized algorithm for DySpec
Dynamic Spectra (DySpec) is a spectrum-transforming fluorescent barcode coupled with time-lapse imaging to exponentially increase the number of co-imaged proteins. Theoretically, if *F* fluorophores are imaged in each cycle, *N* cycles of image can visualize *F<sup>N</sup>* proteins. To decode proteins from DySpec images, we developed two customized algorithms here: `Image subtraction` and `Linear unmixing`. The step-by-step operations and critical considerations are outlined in detail below.

### Hardware
- Computer workstation equipped with an AMD Ryzen 5975WX CPU
- NVIDIA RTX 3090 graphics processing card

### Software
- [Python](https://www.python.org/).
- [Jupyter Notebook](https://jupyter.org/).
- [Fiji](https://imagej.net/software/fiji/downloads)
- [ZEN](https://www.zeiss.com/microscopy/zh/products/software/zeiss-zen.html)
- [elastix](https://elastix.dev/).

### Steps to install python environment and libraries:

#### 1. Install Jupyter Notebook 7.x (or Python 3.x)
- `Jupyter Notebook 7.x` is available for download from the [official Jupyter website](https://jupyter.org/).
- `Python 3.x` is available for download from the [official Python website](https://www.python.org/).

#### 2. Update `pip` (optional)
It's a good practice to keep `pip` up to date. Open a terminal (or command prompt) and run the following command:

```bash
python -m pip install --upgrade pip
```

#### 3. Install Required Libraries
The required python libraries include: 
- `PIL`
- `opencv-python`
- `numpy`
- `scipy`
- `scikit-image`
- `pandas`
- `openpyxl`
- `itertools`
- `os`
- `SimpleITK`
- `SimpleITK-SimpleElastix`<br>

For example, to install `numpy`, execute the following command in your terminal or command prompt:

```bash
pip install numpy
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

This script will print the installed versions of the libraries. If you see the version numbers printed without any errors, the libraries were installed successfully.

### Run the code
- Execute the integrated Notebook from Step 1 to Step 7 in order.
- Test images are included in each subfolder.
