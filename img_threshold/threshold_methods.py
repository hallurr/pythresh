import numpy as np
import torch
import scipy.ndimage as ndimage


def create_circular_kernel(radius):
    diameter = 2 * radius + 1
    Y, X = np.ogrid[:diameter, :diameter]
    dist_from_center = np.sqrt((X - radius)**2 + (Y - radius)**2)
    kernel = dist_from_center <= radius
    return kernel.astype(float)

def apply_local_stats_via_convolution(image, radius):
    kernel = create_circular_kernel(radius)
    kernel_sum = np.sum(kernel)

    radnum = radius * 2 + 1
    # Pad image with NaNs to handle borders
    padded_image = np.pad(image, radnum, mode='constant', constant_values=0)

    # Convolve to get the sum for mean calculation
    local_sums = ndimage.convolve(padded_image, kernel, mode='constant', cval=0)
    local_means = local_sums / kernel_sum

    # Convolve squared image to compute standard deviation
    local_sums_sq = ndimage.convolve(padded_image**2, kernel, mode='constant', cval=0)
    local_means_sq = local_sums_sq / kernel_sum
    local_vars = local_means_sq - local_means**2
    local_stds = np.sqrt(local_vars)

    # Correct the boundary effects
    # valid_mask = ~np.isnan(local_means)
    # local_means = np.where(valid_mask, local_means, np.nan)
    # local_stds = np.where(valid_mask, local_stds, np.nan)

    return local_means[radnum:-radnum, radnum:-radnum], local_stds[radnum:-radnum, radnum:-radnum]

def validate_image(image, must_be_2d=False, must_be_3d=False, must_be_np=False, must_be_tensor=False):
    """Check if the input is a valid image array."""
    if must_be_np and not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if must_be_tensor and not isinstance(image, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")
    if must_be_2d and image.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if must_be_3d and image.ndim != 3:
        raise ValueError("Input must be a 3D array")
    if image.ndim not in {2, 3}:
        raise ValueError("Input must be a 2D or 3D array")

def phansalkar(image, radius=10, k=0.25, r=0.5, p=2.0, q=10.0):

    """
    Phansalkar’s thresholding.

    μ and σ are the mean and the standard deviation of the neighbourhood, respectively.
    Threshold is μ∗(1.0+p∗exp(−q∗μ)+k∗(σ/r−1))

    Described in N. Phansalskar, S. More, and A. Sabale, et al., 
    Adaptive local thresholding for detection of nuclei in diversity stained cytology images, 
    International Conference on Communications and Signal Processing (ICCSP), 2011.
    [https://ieeexplore.ieee.org/document/5739305/]
    """
    
    # Check if the vector is a grayscale numpy array or tensor
    if isinstance(image, np.ndarray):
        validate_image(image, must_be_2d=True, must_be_np=True)
        # normalize the image between 0 and 1
        image_norm = image / np.max(image)
        
        # Get the local means and local stds
        local_means, local_stds = apply_local_stats_via_convolution(image_norm, radius)
        phansalkar_img = np.zeros_like(image_norm)
        phansalkar_img = image_norm > (local_means * (1.0 + p * np.exp(-q * local_means) + k * ((local_stds / r) - 1)))
        return phansalkar_img
        
    elif isinstance(image, torch.Tensor):
        validate_image(image, must_be_2d=True, must_be_tensor=True)
    else:
        print('test4')
        raise ValueError("The input vector must be a grayscale numpy array or a tensor.")
    
    
    
def otsu(image):
    """
    Using a "Faster Approach" the threshold with the maximum between class variance also has the minimum within class variance.
    
    Described in N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," 
    in IEEE Transactions on Systems, Man, and Cybernetics, 
    vol. 9, no. 1, pp. 62-66, Jan. 1979, 
    doi: 10.1109/TSMC.1979.4310076.
    [https://ieeexplore.ieee.org/document/4310076]."""

    
    # Check if the vector is a grayscale numpy array or tensor
    if isinstance(image, np.ndarray):
        validate_image(image, must_be_2d=True, must_be_np=True)
        
        # make sure it is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        image_flat = image.flatten() # Flatten the image
        max_BCV = 0
        threshold_value = None
        for i, threshold in enumerate(range(1, 256)):
            foreground = image_flat[image_flat > threshold]
            background = image_flat[image_flat <= threshold]
            if len(foreground) == 0 or len(background) == 0:
                continue
            W_b = len(foreground) / len(image.flatten())
            W_f = len(background) / len(image.flatten())
            between_class_variances = W_b * W_f * (np.mean(foreground) - np.mean(background))**2
            if between_class_variances > max_BCV:
                max_BCV = between_class_variances
                threshold_value = threshold
                
        otsu_img = np.zeros_like(image)
        otsu_img = image > threshold_value
        return otsu_img
        
    elif isinstance(image, torch.Tensor):
        validate_image(image, must_be_2d=True, must_be_tensor=True)
    else:
        print('test4')
        raise ValueError("The input vector must be a grayscale numpy array or a tensor.")