"""_summary_
This module provides various denoising techniques for 3D thermal data arrays.
Functions
---------
total_variation_minimization(T, weight=0.1, max_iter=100)
graph_signal_denoising(T, tau=10)
deep_image_prior_denoising(T, num_iter=5000, lr=0.01)
Examples
--------
>>> T_denoised_gsp = graph_signal_denoising(T, tau=10)
>>> T_denoised_dip = deep_image_prior_denoising(T, num_iter=5000, lr=0.01)
Notes
-----
- The `total_variation_minimization` function requires the `scikit-image` package.
- The `graph_signal_denoising` function requires the `pygsp` package.
- The `deep_image_prior_denoising` function requires the `torch` package and a CUDA-enabled GPU for optimal performance.
"""

def total_variation_minimization(T, weight=0.1, max_iter=100):
    """
    Perform Total Variation Minimization on a 3D array of thermal data.

    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    weight : float, optional
        Regularization parameter (default is 0.1).
    max_iter : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    T_tv : numpy.ndarray
        TV-minimized thermal data with the same dimensions as input T.

    Example
    -------
    >>> T_tv = total_variation_minimization(T, weight=0.1)
    """
    import numpy as np
    import warnings
    import skimage
    
    try:
        from skimage.restoration import denoise_tv_chambolle
    except ImportError:
        raise ImportError("Please install the 'scikit-image' package: pip install scikit-image")

    height, width, frames = T.shape
    T_tv = np.zeros_like(T)

    for k in range(frames):
        T_tv[:, :, k] = denoise_tv_chambolle(T[:, :, k], weight=weight, channel_axis=False, max_num_iter=max_iter)

    return T_tv



def graph_signal_denoising(T, tau=10):
    """
    Perform Graph Signal Processing-based denoising on thermal data.
    
    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    tau : float, optional
        Time parameter for the heat kernel filter (default is 10).
    
    Returns
    -------
    T_denoised : numpy.ndarray
        Denoised thermal data with the same dimensions as input T.
    
    Example
    -------
    >>> T_denoised = graph_signal_denoising(T, tau=10)
    """
    try:
        from pygsp import graphs, filters
    except ImportError:
        raise ImportError("Please install the 'pygsp' package: pip install pygsp")
    import numpy as np
    
    height, width, frames = T.shape
    T_denoised = np.zeros_like(T)
    
    # Create a 2D grid graph representing the image structure
    G = graphs.Grid2d(N1=height, N2=width)
    G.compute_laplacian()
    
    # Define a heat kernel filter on the graph
    g = filters.Heat(G, tau=tau)
    
    # Apply the filter to each frame
    for k in range(frames):
        signal = T[:, :, k].flatten()
        signal_denoised = g.filter(signal)
        T_denoised[:, :, k] = signal_denoised.reshape(height, width)
    
    return T_denoised

def deep_image_prior_denoising(T, num_iter=5000, lr=0.01):
    """
    Perform denoising using the Deep Image Prior method.

    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    num_iter : int, optional
        Number of iterations for optimization (default is 5000).
    lr : float, optional
        Learning rate for the optimizer (default is 0.01).

    Returns
    -------
    T_denoised : numpy.ndarray
        Denoised thermal data with the same dimensions as input T.

    Example
    -------
    >>> T_denoised = deep_image_prior_denoising(T, num_iter=5000, lr=0.01)
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np

        # Check if CUDA is available and use GPU if possible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        height, width, frames = T.shape
        T_denoised = np.zeros_like(T)

        # Define a simple CNN architecture
        class DeepImagePriorNet(nn.Module):
            def __init__(self):
                super(DeepImagePriorNet, self).__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 1, kernel_size=3, padding=1)
                )

            def forward(self, x):
                return self.net(x)

        # Process each frame individually
        for k in range(frames):
            print(f"Processing frame {k+1}/{frames}")
            # Get the noisy image
            img_noisy_np = T[:, :, k]

            # Normalize the image
            img_noisy_np = (img_noisy_np - img_noisy_np.min()) / (img_noisy_np.max() - img_noisy_np.min())
            img_noisy = torch.from_numpy(img_noisy_np).float().unsqueeze(0).unsqueeze(0).to(device)

            # Initialize the network and optimizer
            net = DeepImagePriorNet().to(device)
            optimizer = optim.Adam(net.parameters(), lr=lr)

            # Random input noise
            input_noise = torch.randn_like(img_noisy).to(device)

            # Define the loss function (MSE)
            mse_loss = nn.MSELoss()

            # Optimization loop
            for i in range(num_iter):
                optimizer.zero_grad()
                output = net(input_noise)
                loss = mse_loss(output, img_noisy)
                loss.backward()
                optimizer.step()

                if i % 500 == 0:
                    print(f"Iteration {i}/{num_iter}, Loss: {loss.item()}")

            # Get the denoised image
            denoised_img = output.detach().cpu().squeeze().numpy()
            # Rescale to original intensity range
            denoised_img = denoised_img * (T[:, :, k].max() - T[:, :, k].min()) + T[:, :, k].min()
            T_denoised[:, :, k] = denoised_img

            # Free up memory
            del net, optimizer, input_noise, img_noisy, output
            torch.cuda.empty_cache()

        return T_denoised

    except Exception as e:
        print(f"Error in deep_image_prior_denoising: {e}")
        return None

def super_resolution_imaging(T, scale_factor=2):
    """
    Perform Super-Resolution Imaging on a 3D array of thermal data using deep learning.

    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    scale_factor : int, optional
        Upscaling factor (default is 2).

    Returns
    -------
    T_sr : numpy.ndarray
        Super-resolved thermal data.

    Example
    -------
    >>> T_sr = super_resolution_imaging(T, scale_factor=2)
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("Please install 'torch': pip install torch")
    import numpy as np
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    class SRCNN(nn.Module):
        def __init__(self):
            super(SRCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
            self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)
            return x

    height, width, frames = T.shape
    T_sr = np.zeros((height * scale_factor, width * scale_factor, frames))

    # Load pre-trained model weights or initialize randomly
    model = SRCNN().to(device)
    # Note: In practice, you should train the model on appropriate data
    # Here, we assume the model is randomly initialized

    for k in range(frames):
        img = T[:, :, k]
        img = (img - img.min()) / (img.max() - img.min())
        img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

        # Upsample the image
        img_upsampled = F.interpolate(img, scale_factor=scale_factor, mode='bicubic', align_corners=False)

        # Apply the super-resolution model
        with torch.no_grad():
            output = model(img_upsampled)

        # Convert back to numpy array
        sr_img = output.squeeze().cpu().numpy()
        sr_img = sr_img * (T[:, :, k].max() - T[:, :, k].min()) + T[:, :, k].min()
        T_sr[:, :, k] = sr_img

        # Free up memory
        torch.cuda.empty_cache()

    return T_sr


