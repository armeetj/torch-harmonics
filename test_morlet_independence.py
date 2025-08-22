"""
Test script to verify that MorletFilterBasis3d produces linearly independent basis functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_harmonics.filter_basis import MorletFilterBasis3d

def recommend_grid_size(kernel_shape, r_cutoff=1.0, width=1.0, safety_factor=2.0):
    """
    Recommend minimum grid size to avoid aliasing for given kernel shape.
    
    The highest frequency component is approximately max(kernel_shape)//2.
    For proper sampling, we need at least 2 points per wavelength (Nyquist).
    """
    max_freq = max(kernel_shape) // 2
    if max_freq == 0:
        max_freq = 1
    
    # Wavelength = 2 * width / frequency
    min_wavelength = 2 * width / max_freq
    
    # Grid spacing in the support region
    grid_extent = 2 * r_cutoff  # from -r_cutoff to +r_cutoff
    
    # Minimum points needed: 2 points per wavelength * safety factor
    min_points_per_dim = int(safety_factor * grid_extent / min_wavelength)
    
    return max(min_points_per_dim, 8)  # At least 8 points per dimension

def plot_filters(basis_functions, D, H, W, kernel_shape):
    """
    Plot all filters as cross-section slices.
    Each row is a filter, each column is a cross-section slice (XY, XZ, YZ planes).
    """
    if len(basis_functions) == 0:
        print("No basis functions to plot")
        return
    
    n_filters = len(basis_functions)
    
    # Limit the number of filters to plot to avoid memory issues
    max_filters_to_plot = 50  # Increased from 20
    if n_filters > max_filters_to_plot:
        print(f"Too many filters ({n_filters}) to plot all. Plotting first {max_filters_to_plot} filters.")
        basis_functions = basis_functions[:max_filters_to_plot]
        n_filters = max_filters_to_plot
    
    n_cols = 3  # XY, XZ, YZ cross-sections
    
    fig, axes = plt.subplots(n_filters, n_cols, figsize=(12, min(3 * n_filters, 150)))  # Increased max height, reduced per-filter height
    if n_filters == 1:
        axes = axes.reshape(1, -1)
    
    # Get center indices for cross-sections
    center_z = D // 2
    center_y = H // 2
    center_x = W // 2
    
    for i, basis_func in enumerate(basis_functions):
        # Reshape from flattened back to 3D
        func_3d = basis_func.reshape(D, H, W)
        
        # XY plane (slice at center Z)
        xy_slice = func_3d[center_z, :, :]
        im1 = axes[i, 0].imshow(xy_slice.numpy(), cmap='RdBu_r', origin='lower')
        axes[i, 0].set_title(f'Filter {i}: XY plane (z={center_z})')
        axes[i, 0].set_xlabel('X')
        axes[i, 0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[i, 0], shrink=0.8)
        
        # XZ plane (slice at center Y)
        xz_slice = func_3d[:, center_y, :]
        im2 = axes[i, 1].imshow(xz_slice.numpy(), cmap='RdBu_r', origin='lower')
        axes[i, 1].set_title(f'Filter {i}: XZ plane (y={center_y})')
        axes[i, 1].set_xlabel('X')
        axes[i, 1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)
        
        # YZ plane (slice at center X)
        yz_slice = func_3d[:, :, center_x]
        im3 = axes[i, 2].imshow(yz_slice.numpy(), cmap='RdBu_r', origin='lower')
        axes[i, 2].set_title(f'Filter {i}: YZ plane (x={center_x})')
        axes[i, 2].set_xlabel('Y')
        axes[i, 2].set_ylabel('Z')
        plt.colorbar(im3, ax=axes[i, 2], shrink=0.8)
        
        # Add filter statistics as text
        func_min, func_max = basis_func.min().item(), basis_func.max().item()
        func_norm = torch.norm(basis_func).item()
        axes[i, 0].text(0.02, 0.98, f'Range: [{func_min:.3f}, {func_max:.3f}]\nNorm: {func_norm:.3f}', 
                       transform=axes[i, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot with lower DPI to handle more filters
    filename = f'morlet_filters_{kernel_shape[0]}x{kernel_shape[1]}x{kernel_shape[2]}.png'
    plt.savefig(filename, dpi=75, bbox_inches='tight')  # Reduced from 150 to 75
    print(f"\nSaved filter visualization to: {filename}")
    
    # Also save a summary plot showing all filters in a grid
    plot_filter_grid(basis_functions, D, H, W, kernel_shape)

def plot_filter_grid(basis_functions, D, H, W, kernel_shape):
    """
    Create a compact grid showing XY cross-sections of all filters.
    """
    if len(basis_functions) == 0:
        return
    
    n_filters = len(basis_functions)
    
    # Arrange filters in a roughly square grid
    grid_size = int(np.ceil(np.sqrt(n_filters)))
    
    # Adjust figure size based on grid size, with lower DPI we can go bigger
    fig_size = min(max(grid_size * 2, 8), 20)  # Scale with grid size but cap at reasonable size
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(fig_size, fig_size))
    if grid_size == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)
    
    center_z = D // 2
    
    for i in range(grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        
        if i < n_filters:
            # Plot the XY cross-section
            func_3d = basis_functions[i].reshape(D, H, W)
            xy_slice = func_3d[center_z, :, :]
            
            im = axes[row, col].imshow(xy_slice.numpy(), cmap='RdBu_r', origin='lower')
            axes[row, col].set_title(f'Filter {i}', fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        else:
            # Hide unused subplots
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save the grid plot with lower DPI
    filename = f'morlet_filters_grid_{kernel_shape[0]}x{kernel_shape[1]}x{kernel_shape[2]}.png'
    plt.savefig(filename, dpi=75, bbox_inches='tight')  # Reduced from 150 to 75
    print(f"Saved filter grid to: {filename}")

def test_linear_independence():
    """Test that MorletFilterBasis3d produces linearly independent basis functions."""
    
    # Create a small test case
    kernel_shape = (5,5,5)  # 8 basis functions total
    basis = MorletFilterBasis3d(kernel_shape=kernel_shape)
    
    print(f"Testing MorletFilterBasis3d with kernel_shape={kernel_shape}")
    print(f"Total basis functions: {basis.kernel_size}")
    
    # Create a 3D grid
    D, H, W = 64,64,64
    
    # Check if grid size is adequate
    recommended_size = recommend_grid_size(kernel_shape, r_cutoff=1.0, width=1.0)
    print(f"Current grid size: {D}×{H}×{W}")
    print(f"Recommended minimum grid size: {recommended_size}×{recommended_size}×{recommended_size}")
    
    if min(D, H, W) < recommended_size:
        print(f"⚠️  WARNING: Grid may be too coarse for kernel_shape={kernel_shape}")
        print(f"   This could cause aliasing and linear dependence issues.")
        print(f"   Consider increasing grid size to at least {recommended_size}×{recommended_size}×{recommended_size}")
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H) 
    z = torch.linspace(-1, 1, D)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.stack([Z, Y, X], dim=0)  # Shape: [3, D, H, W]
    
    print(f"Grid shape: {grid.shape}")
    
    # Compute basis functions
    r_cutoff = 1.0
    width = 1.0
    
    iidx, vals = basis.compute_support_vals(grid, r_cutoff=r_cutoff, width=width)
    
    print(f"Number of non-zero entries: {len(vals)}")
    print(f"Index tensor shape: {iidx.shape}")
    
    # Analyze the support distribution
    r = torch.sqrt(grid[0]**2 + grid[1]**2 + grid[2]**2)
    support_points = (r <= r_cutoff).sum().item()
    print(f"Grid points within support (r <= {r_cutoff}): {support_points} out of {D*H*W} total")
    print(f"Support fraction: {support_points/(D*H*W):.3f}")
    
    # Check how many basis functions have non-zero support
    unique_basis_indices = torch.unique(iidx[:, 0])
    print(f"Basis functions with non-zero support: {len(unique_basis_indices)} out of {basis.kernel_size}")
    
    # Reconstruct full basis functions for analysis
    basis_functions = []
    
    for k in range(basis.kernel_size):
        # Find indices for this basis function
        mask = iidx[:, 0] == k
        if mask.sum() == 0:
            print(f"Warning: Basis function {k} has no support!")
            continue
            
        k_indices = iidx[mask, 1:]  # [nnz_k, 3] - spatial indices
        k_vals = vals[mask]        # [nnz_k] - values
        
        # Create sparse representation
        full_func = torch.zeros(D, H, W)
        full_func[k_indices[:, 0], k_indices[:, 1], k_indices[:, 2]] = k_vals
        
        basis_functions.append(full_func.flatten())
        
        print(f"Basis function {k}: {mask.sum()} non-zero values, "
              f"range=[{k_vals.min():.4f}, {k_vals.max():.4f}]")
        
        # For debugging: show the (p,m,n) indices for this basis function
        if k < 10:  # Only show first 10 to avoid clutter
            p_idx = k // (kernel_shape[1] * kernel_shape[2])
            m_idx = (k // kernel_shape[2]) % kernel_shape[1] 
            n_idx = k % kernel_shape[2]
            print(f"  -> 3D index: ({p_idx}, {m_idx}, {n_idx})")
    
    if len(basis_functions) < 2:
        print("Not enough basis functions to test independence")
        return
    
    # Stack basis functions into a matrix
    basis_matrix = torch.stack(basis_functions, dim=1)  # [spatial_points, n_basis]
    print(f"Basis matrix shape: {basis_matrix.shape}")
    
    # Compute rank to check linear independence
    try:
        rank = torch.linalg.matrix_rank(basis_matrix, tol=1e-6)
        n_functions = basis_matrix.shape[1]
        
        print(f"Matrix rank: {rank}")
        print(f"Number of basis functions: {n_functions}")
        
        if rank == n_functions:
            print("✅ Basis functions are linearly independent!")
        else:
            print(f"❌ Basis functions are NOT linearly independent!")
            print(f"   Expected rank: {n_functions}, Got rank: {rank}")
            
            # Compute condition number for additional insight
            try:
                U, S, Vh = torch.linalg.svd(basis_matrix)
                cond_num = S.max() / S.min()
                print(f"   Condition number: {cond_num:.2e}")
                print(f"   Singular values: {S}")
            except Exception as e:
                print(f"   Could not compute SVD: {e}")
        
    except Exception as e:
        print(f"Error computing matrix rank: {e}")
    
    # Additional test: check for identical functions
    print("\nChecking for identical basis functions...")
    for i in range(len(basis_functions)):
        for j in range(i+1, len(basis_functions)):
            diff = torch.norm(basis_functions[i] - basis_functions[j])
            if diff < 1e-10:
                print(f"❌ Basis functions {i} and {j} are identical (diff={diff:.2e})")
            elif diff < 1e-6:
                print(f"⚠️  Basis functions {i} and {j} are very similar (diff={diff:.2e})")
    
    print("Test completed!")
    
    # Print summary and recommendations
    print("\n" + "="*60)
    print("SUMMARY:")
    if rank == n_functions:
        print("✅ All basis functions are linearly independent!")
    else:
        print("❌ Linear dependence detected!")
        print("\nPOSSIBLE CAUSES:")
        print("1. Grid resolution too low for high-frequency basis functions (aliasing)")
        print("2. Insufficient spatial support (small r_cutoff)")
        print("3. Numerical precision issues")
        
        print(f"\nRECOMMENDATIONS:")
        rec_size = recommend_grid_size(kernel_shape, r_cutoff=r_cutoff, width=width)
        if min(D, H, W) < rec_size:
            print(f"• Increase grid size to at least {rec_size}×{rec_size}×{rec_size}")
        if r_cutoff < 1.5:
            print(f"• Consider increasing r_cutoff from {r_cutoff} to 1.5 or higher")
        if max(kernel_shape) > 4 and min(D, H, W) < 20:
            print(f"• For kernel_shape={kernel_shape}, use grid size ≥ 20×20×20")
    
    print("="*60)
    
    # Plot all filters as cross-section slices
    plot_filters(basis_functions, D, H, W, kernel_shape)

if __name__ == "__main__":
    test_linear_independence()