# pythresh
A collection of thresholding methods implemented for numpy and pytorch arrays


## Description
`img_threshold` is a Python package providing a variety of image thresholding techniques with a focus on support for PyTorch gradient functionality. This package fills the gap for advanced thresholding methods that are differentiable and can be used in deep learning pipelines.

## Installation

Install `img_threshold` using pip:

```bash
pip install git+https://https://github.com/hallurr/pythresh.git
```

## Examples

For illustrative purposes a normal optical coherence tomography image has been chosen to demonstrate from the publicly available [kaggle databank](https://www.kaggle.com/datasets/paultimothymooney/kermany2018). 
<div align="center">
  <img src="examples/NORMAL-OCT.jpeg" alt="Normal OCT">
</div>






### Phansalkar’s thresholding method



Phansalkar’s thresholding is suited for local thresholding.

The normalized image is assessed for each pixel whether its value is greater than the following threshold:

$$
\mu \cdot \left(1.0 + p \cdot \exp(-q \cdot \mu) + k \cdot \left(\frac{\sigma}{r} - 1\right)\right)
$$

  
where $\mu$ and $\sigma$ are the mean and the standard deviation of neighbourhood, respectively. 
$k$, $r$, $p$, and $q$ are changeable parameters defaulted to:\
$k=0.25$\
$r=0.5$\
$p=2.0$\
$q=10.0$\
#### Example

```
import sys
import os

# Add the parent directory to the sys.path list
sys.path.append(os.path.abspath('../')) 

# Phansalkar method example
from img_threshold.threshold_methods import phansalkar
from img_threshold.utils import *

# load the Normal OCT image
img = load_image('NORMAL-OCT.jpeg')[:, :, 0]

# Perform Phansalkar’s thresholding with different radii neighborhoods
img_thresholded_r5 = phansalkar(img, radius=5)
img_thresholded_r10 = phansalkar(img, radius=10)
img_thresholded_r25 = phansalkar(img, radius=25)

# Show the input and outputs
images = [load_image('NORMAL-OCT.jpeg')[:, :, 0], 
          phansalkar(img, radius=10), 
          phansalkar(img, radius=25), 
          phansalkar(img, radius=50)]

titles = ['Original Image', 
          'Phansalkar Threshold\n radius = 5', 
          'Phansalkar Threshold\n radius = 10', 
          'Phansalkar Threshold\n radius = 50']

show_images(images, titles=titles, cmap='gray', figsize=(15, 10))
```
<div align="center">

<img src="examples/Phansalkar.png" alt="Phansalkar Thresholding">
</div>

