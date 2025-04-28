import torch
import taichi as ti
import time
import numpy as np
import warp as wp # Import Warp
import warp.sparse # Import Warp sparse module
import warp.optim # Import Warp optim module
import warp.optim.linear
import os # Needed maybe for CUDA device selection for Warp init

# --- Global Configuration Options ---
# Choose backend (cuda for performance where possible, cpu fallback)
TARGET_ARCH = 'cuda' # Options: 'cuda', 'cpu'
TARGET_DTYPE_TORCH = torch.float32
TARGET_DTYPE_TAICHI = ti.f32
TARGET_DTYPE_WARP = wp.types.float32 # Assuming float32 for consistency

# --- Setup ---
def setup(arch_str):
    """Initializes Taichi and Warp, determines PyTorch device."""
    print(f"Attempting to use arch: {arch_str}")

    # PyTorch Device Setup
    torch_device_type = arch_str
    if torch_device_type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU for PyTorch, Taichi, and Warp.")
        torch_device_type = 'cpu'
        arch_str = 'cpu' # Ensure other frameworks also use CPU
    torch_device = torch.device(torch_device_type)
    print(f"Using PyTorch device: {torch_device}")

    # Taichi Initialization
    taichi_arch = ti.cuda if arch_str == 'cuda' else ti.cpu
    ti.init(arch=taichi_arch, default_fp=TARGET_DTYPE_TAICHI)

    # Warp Initialization
    # Warp picks up CUDA_VISIBLE_DEVICES similar to torch.
    # If torch_device is 'cuda:1', set os.environ before wp.init if necessary.
    # For simple examples, wp.init() usually syncs with default CUDA device.
    # explicitly pass device_id if needed: wp.init(device_id=torch_device.index)
    if arch_str == 'cuda':
       wp.init() # Initializes CUDA device by default
       print(f"Using Warp device: {wp.get_device().name}")
    else:
       # Warp CPU backend might not be as feature complete or performant
       # Check Warp docs for explicit_cpu_init if needed, but often cuda init required for linalg
       try:
            wp.init() # Try CUDA init, might be required for linalg even if we aim for CPU comparison later
            print(f"Warp initialized (likely CUDA) even if target is CPU. Using device: {wp.get_device().name}")
       except Exception as e:
             print(f"Warp CUDA init failed (this is expected if CUDA not available): {e}")
             print("Warp linear algebra typically requires CUDA. Warp comparison may not work on pure CPU.")
             # If targeting CPU and CUDA is not available, Warp sparse solvers might not work.
             # We'll handle this by skipping the Warp test if it fails.
             pass # Continue with other tests

    return torch_device # Return the determined torch device

# --- Problem Creation ---
def create_problem(n: int, dtype: torch.dtype, device: torch.device):
    """Creates the sparse matrix A and vector b using PyTorch, returns nnz data."""
    print(f"\nCreating {n}x{n} PyTorch matrix and vector on {device}...")

    # Create a tridiagonal matrix (1D Laplacian with Dirichlet boundary conditions)
    # A is symmetric positive definite, suitable for LLT
    A_torch = torch.diag(torch.ones(n, dtype=dtype) * 2.0, 0)
    A_torch += torch.diag(torch.ones(n - 1, dtype=dtype) * -1.0, 1)
    A_torch += torch.diag(torch.ones(n - 1, dtype=dtype) * -1.0, -1)
    A_torch = A_torch.to(device) # Move A to specified device

    # Create a vector b (e.g., a simple source term)
    b_torch = torch.ones(n, dtype=dtype).to(device) # Move b to specified device
    
    # A_torch = torch.load("/debug/jacobian_100.pt").to(device)
    # b_torch = torch.load("/debug/residual_100.pt").flatten().to(device)

    print(f"PyTorch A shape: {A_torch.shape}")
    print(f"PyTorch b shape: {b_torch.shape}")

    # Get non-zero elements and their indices for sparse methods
    # Add .contiguous() to ensure contiguous memory for Taichi/Warp kernel usage
    # .to(device) ensures they are on the correct device BEFORE contiguous is called if possible
    A_nnz_indices = A_torch.nonzero(as_tuple=False).to(device).contiguous().type(torch.int32)
    A_nnz_values = A_torch[A_nnz_indices[:, 0], A_nnz_indices[:, 1]].to(device).contiguous()
    num_non_zeros = A_nnz_values.shape[0]

    print(f"Number of non-zero elements in A: {num_non_zeros}")

    return A_torch, b_torch, A_nnz_indices, A_nnz_values

# --- Kernels for Taichi Interop (defined globally or associated with Taichi section) ---

# Kernel to copy vector from PyTorch tensor to Taichi ndarray
@ti.kernel
def copy_vec_torch_to_ti_kernel(dst: ti.types.ndarray(), src: ti.types.ndarray(), size: ti.i32):
    for i in range(size):
        dst[i] = src[i]

# Kernel to populate the Taichi SparseMatrixBuilder from PyTorch non-zero data
@ti.kernel
def populate_taichi_builder_kernel(builder: ti.types.sparse_matrix_builder(),
                                  A_data: ti.types.ndarray(), # PyTorch tensor of nnz values
                                  A_indices: ti.types.ndarray(), # PyTorch tensor of nnz indices (int)
                                  num_non_zeros: ti.i32):
    # Use struct for access, more explicit than Python-scope access for nnz_indices
    # Consider alternative if struct access is limiting: pass row/col indices as separate ndarrays
    for i in range(num_non_zeros):
        row = A_indices[i, 0]
        col = A_indices[i, 1]
        val = A_data[i]
        builder[row, col] += val # Add to the builder

# Kernel to copy data from Taichi ndarray to PyTorch tensor
@ti.kernel
def copy_ti_ndarray_to_torch_kernel(dst: ti.types.ndarray(), src: ti.types.ndarray(), size: ti.i32):
    for i in range(size):
        dst[i] = src[i]


# --- Solver Functions ---
def solve_pytorch_dense(A_pt: torch.Tensor, b_pt: torch.Tensor, device: torch.device):
    """Solves Ax=b using PyTorch's dense solver."""
    print("\nSolving Ax=b using PyTorch (dense)...")
    # Synchronize for accurate timing on CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # torch.linalg.solve requires inputs to be on the same device
    x_pt = torch.linalg.solve(A_pt, b_pt)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    solve_time = end_time - start_time
    print(f"PyTorch solve time: {solve_time:.6f} seconds")
    return x_pt, solve_time

# Remove the 'arch: ti.arch' parameter
def solve_taichi_sparse(A_pt: torch.Tensor, b_pt: torch.Tensor,
                        A_nnz_indices: torch.Tensor, A_nnz_values: torch.Tensor,
                        n: int, device: torch.device): # Removed 'arch' parameter
    """Solves Ax=b using Taichi's sparse solver."""
    print("\nSolving Ax=b using Taichi (sparse LLT)...")

    # Create Taichi SparseMatrixBuilder
    max_triplets = n * 3 # Sufficient for tridiagonal
    K_ti_builder = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=max_triplets, dtype=TARGET_DTYPE_TAICHI)

    # Create Taichi ndarray for vector b (destination for copy)
    b_ti_ndarray = ti.ndarray(dtype=TARGET_DTYPE_TAICHI, shape=n)

    # Copy b_torch data into b_ti_ndarray using a kernel
    copy_vec_torch_to_ti_kernel(b_ti_ndarray, b_pt, n) # Pass b_torch directly

    # Populate the sparse builder using a kernel
    num_non_zeros = A_nnz_values.shape[0]
    populate_taichi_builder_kernel(K_ti_builder, A_nnz_values, A_nnz_indices, num_non_zeros) # Pass PyTorch tensors directly

    # Build the Taichi sparse matrix
    A_ti = K_ti_builder.build()

    # Solve Ax=b
    ti.sync() # Synchronize for accurate timing
    start_time = time.perf_counter()

    solver = ti.linalg.SparseSolver(solver_type="LLT")
    # Solver setup (analysis and factorization) is part of the solve time here
    # In practice, analyze_pattern and factorize can be done once if matrix structure/values don't change.
    try:
        solver.analyze_pattern(A_ti)
        solver.factorize(A_ti)
        x_ti_ndarray = solver.solve(b_ti_ndarray)
        solve_succeeded = True
    except Exception as e:
        print(f"Taichi solver failed: {e}")
        solve_succeeded = False
        x_ti_ndarray = ti.ndarray(dtype=TARGET_DTYPE_TAICHI, shape=n) # Create dummy result


    ti.sync() # Synchronize after the Taichi op
    end_time = time.perf_counter()
    solve_time = end_time - start_time

    print(f"Taichi solve time (including factorization): {solve_time:.6f} seconds")
    # solver.info() is not supported on all backends/solver types, removed call

    # Convert Taichi ndarray output back to PyTorch Tensor (on device)
    # Create empty PyTorch tensor on the target device
    x_ti_torch = torch.empty_like(b_pt)
    # Copy data using kernel
    copy_ti_ndarray_to_torch_kernel(x_ti_torch, x_ti_ndarray, n)

    # Free Taichi result ndarray memory
    del x_ti_ndarray # Important for cleaning up device memory

    if not solve_succeeded:
        x_ti_torch = torch.full_like(b_pt, torch.nan) # Indicate failure in result tensor

    return x_ti_torch, solve_time

def solve_warp_sparse(A_pt: torch.Tensor, b_pt: torch.Tensor, # A_pt needed for comparison, maybe remove later if not
                      A_nnz_indices: torch.Tensor, A_nnz_values: torch.Tensor,
                      n: int, device: torch.device):
    """Solves Ax=b using Warp's sparse iterative solver (CG)."""
    print("\nSolving Ax=b using Warp (sparse CG)...")

    # wp_device = wp.get_device() # No longer need to pass this explicitly to from_torch

    try:
        # --- Add Sorting Step ---
        # (Keep the sorting logic from the previous suggestion)
        sort_by_col_indices = torch.argsort(A_nnz_indices[:, 1], stable=True)
        indices_temp = A_nnz_indices[sort_by_col_indices]
        values_temp = A_nnz_values[sort_by_col_indices]
        final_sort_indices = torch.argsort(indices_temp[:, 0], stable=True)
        A_nnz_indices_sorted = indices_temp[final_sort_indices]
        A_nnz_values_sorted = values_temp[final_sort_indices]
        # --- End Sorting Step ---

        # Convert sorted inputs to Warp arrays.
        # Remove the 'device=wp_device' argument.
        # Warp infers the device from the input PyTorch tensor's device.
        wp_row_indices = wp.from_torch(A_nnz_indices_sorted[:, 0].contiguous()) # REMOVED device=...
        wp_col_indices = wp.from_torch(A_nnz_indices_sorted[:, 1].contiguous()) # REMOVED device=...
        wp_data = wp.from_torch(A_nnz_values_sorted.contiguous())             # REMOVED device=...
        b_warp = wp.from_torch(b_pt)                                         # REMOVED device=...

        block_type_warp = TARGET_DTYPE_WARP

        # Build Warp sparse matrix in BSR format
        A_warp = wp.sparse.bsr_from_triplets(
            rows_of_blocks=n,
            cols_of_blocks=n,
            rows=wp_row_indices,
            columns=wp_col_indices,
            values=wp_data,
            prune_numerical_zeros=False
        )

        # --- Create Preconditioner ---
        # Use the 'diag' preconditioner based on the image
        # jacobi_preconditioner = wp.optim.linear.preconditioner(A_warp, ptype='diag')

        x_warp_array = wp.zeros(n, dtype=TARGET_DTYPE_WARP)  # Shape n, correct dtype, automatically on current Warp device

        # --- Solve Ax=b using Conjugate Gradient (CG) ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        res = wp.optim.linear.cr(
            A_warp,
            b_warp,
            x_warp_array,
            # M=jacobi_preconditioner,
            tol=1e-4,
            atol=1e-4,
            maxiter=n * 2,
            use_cuda_graph=True,
        )

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        solve_time = end_time - start_time

        print(f"Warp CG solve time: {solve_time:.6f} seconds")

        if res:
            print(f"Warp CG solver converged successfully with residual {res[1]:.6e}")
            solve_succeeded = True
        else:
            solve_succeeded = False

        # Convert Warp array output back to PyTorch Tensor
        # x_warp_array is already on the correct device
        x_warp_torch = wp.to_torch(x_warp_array)

        # Fill with NaN if solve failed based on the success_flag
        if not solve_succeeded:
             x_warp_torch.fill_(torch.nan)

        # Clean up Warp arrays (optional, GC handles this)
        # del wp_row_indices, wp_col_indices, wp_data, b_warp, A_warp, x_warp_array

    except Exception as e:
        print(f">>> Warp solver process FAILED due to exception: {e}")
        solve_succeeded = False # Exception indicates failure
        solve_time = 0.0 # Indicate failure/skipped time
        x_warp_torch = torch.full_like(b_pt, torch.nan) # Indicate failure in result tensor

    return x_warp_torch, solve_time




# --- Comparison ---
def compare_results(x_pt: torch.Tensor, x_ti: torch.Tensor, x_warp: torch.Tensor,
                    A_pt: torch.Tensor, b_pt: torch.Tensor, timings: dict, device: torch.device):
    """Compares the results and prints norms, residuals, and timings."""
    print("\n--- Comparison ---")

    # --- Compute and print Difference Norms (comparing Taichi and Warp against PyTorch dense solution) ---
    print("\n--- Difference Norms vs PyTorch Dense Solution ---")
    if not torch.isnan(x_ti).any(): # Check if Taichi result is valid
        difference_ti = x_pt - x_ti
        difference_norm_ti_abs = torch.linalg.norm(difference_ti)
        print(f"L2 norm of difference ||x_torch - x_ti|| (Abs): {difference_norm_ti_abs.item():.6e}")
    else:
        print("Taichi solution invalid (contains NaN). Difference norm skipped.")
        difference_norm_ti_abs = torch.inf # Indicate large difference

    if not torch.isnan(x_warp).any(): # Check if Warp result is valid
        difference_warp = x_pt - x_warp
        difference_norm_warp_abs = torch.linalg.norm(difference_warp)
        print(f"L2 norm of difference ||x_torch - x_warp|| (Abs): {difference_norm_warp_abs.item():.6e}")
    else:
         print("Warp solution invalid (contains NaN). Difference norm skipped.")
         difference_norm_warp_abs = torch.inf # Indicate large difference

    # Optional: Relative difference norms
    x_pt_norm = torch.linalg.norm(x_pt)
    if x_pt_norm.abs().item() > 1e-9:
         print(f"L2 norm of PyTorch dense solution ||x_torch||: {x_pt_norm.item():.6e}")
         if difference_norm_ti_abs != torch.inf:
             difference_norm_ti_rel = difference_norm_ti_abs / x_pt_norm
             print(f"Relative L2 difference: ||x_torch - x_ti|| / ||x_torch||: {difference_norm_ti_rel.item():.6e}")
         if difference_norm_warp_abs != torch.inf:
             difference_norm_warp_rel = difference_norm_warp_abs / x_pt_norm
             print(f"Relative L2 difference: ||x_torch - x_warp|| / ||x_torch||: {difference_norm_warp_rel.item():.6e}")
    else:
         print("Norm of PyTorch dense solution is close to zero, cannot calculate relative differences.")


    # --- Compute and print Residual Norms (calculated using PyTorch matmul) ---
    print("\n--- Residual Comparison (calculated using PyTorch matmul) ---")

    # Calculate residual for PyTorch's solution: r = b - A @ x_torch
    residual_torch = b_pt - (A_pt @ x_pt)
    residual_norm_torch_abs = torch.linalg.norm(residual_torch)
    print(f"PyTorch Solution Residual ||b - A @ x_torch|| (Abs): {residual_norm_torch_abs.item():.6e}")

    # Calculate residual for Taichi's solution: r = b - A @ x_ti
    if not torch.isnan(x_ti).any():
        residual_taichi = b_pt - (A_pt @ x_ti)
        residual_norm_taichi_abs = torch.linalg.norm(residual_taichi)
        print(f"Taichi Solution Residual ||b - A @ x_ti|| (Abs): {residual_norm_taichi_abs.item():.6e}")
    else:
        print("Taichi solution invalid. Residual calculation skipped.")
        residual_norm_taichi_abs = torch.inf

    # Calculate residual for Warp's solution: r = b - A @ x_warp
    if not torch.isnan(x_warp).any():
         residual_warp = b_pt - (A_pt @ x_warp)
         residual_norm_warp_abs = torch.linalg.norm(residual_warp)
         print(f"Warp Solution Residual ||b - A @ x_warp|| (Abs): {residual_norm_warp_abs.item():.6e}")
    else:
         print("Warp solution invalid. Residual calculation skipped.")
         residual_norm_warp_abs = torch.inf


    # Calculate Relative Residuals
    b_norm = torch.linalg.norm(b_pt)
    if b_norm.abs().item() > 1e-9:
        print(f"L2 norm of vector b ||b||: {b_norm.item():.6e}")
        residual_norm_torch_rel = residual_norm_torch_abs / b_norm
        print(f"PyTorch Solution Residual ||b - A @ x_torch|| (Rel): {residual_norm_torch_rel.item():.6e}")
        if residual_norm_taichi_abs != torch.inf:
             residual_norm_taichi_rel = residual_norm_taichi_abs / b_norm
             print(f"Taichi Solution Residual ||b - A @ x_ti|| (Rel): {residual_norm_taichi_rel.item():.6e}")
        if residual_norm_warp_abs != torch.inf:
             residual_norm_warp_rel = residual_norm_warp_abs / b_norm
             print(f"Warp Solution Residual ||b - A @ x_warp|| (Rel): {residual_norm_warp_rel.item():.6e}")
    else:
         print("Norm of b is close to zero, cannot calculate relative residuals.")


    # Print timings comparison
    print(f"\n--- Timing Comparison ---")
    for name, t in timings.items():
        print(f"{name}: {t:.6f} seconds")

    # Optional: Pretty print speedup vs PyTorch dense
    if timings.get('PyTorch (dense)', 0) > 1e-9: # Avoid division by zero
         pt_time = timings['PyTorch (dense)']
         if timings.get('Taichi (sparse LLT)', 0) > 1e-9 and timings['Taichi (sparse LLT)'] < pt_time:
              print(f"Taichi (sparse LLT) was {pt_time / timings['Taichi (sparse LLT)']:.2f}x faster than PyTorch.")
         if timings.get('Warp (sparse)', 0) > 1e-9 and timings['Warp (sparse)'] < pt_time:
              print(f"Warp (sparse) was {pt_time / timings['Warp (sparse)']:.2f}x faster than PyTorch.")
         # Add comparison between Taichi and Warp if both ran
         if timings.get('Taichi (sparse LLT)', 0) > 1e-9 and timings.get('Warp (sparse)', 0) > 1e-9:
             taichi_time = timings['Taichi (sparse LLT)']
             warp_time = timings['Warp (sparse)']
             if taichi_time < warp_time:
                 print(f"Taichi (sparse LLT) was {warp_time / taichi_time:.2f}x faster than Warp.")
             elif warp_time < taichi_time:
                 print(f"Warp (sparse) was {taichi_time / warp_time:.2f}x faster than Taichi.")


# --- Main Execution ---
if __name__ == '__main__':
    N = 100 # Matrix dimension
    timings = {} # Dictionary to store timings
    wp.init()

    # 1. Setup environments and determine device
    torch_device = setup(TARGET_ARCH)

    # 2. Create problem data (A and b)
    A_torch, b_torch, A_nnz_indices, A_nnz_values = create_problem(N, TARGET_DTYPE_TORCH, torch_device)

    # 3. Solve using PyTorch's Dense Solver
    # Ensure A_torch and b_torch are on the correct device
    x_torch, solve_time_pt = solve_pytorch_dense(A_torch, b_torch, torch_device)
    timings['PyTorch (dense)'] = solve_time_pt

    # 4. Solve using Taichi's Sparse Solver
    # Pass the original PyTorch tensors holding nnz data and b_torch
    x_ti_torch, solve_time_ti = solve_taichi_sparse(A_torch, b_torch,
                                                    A_nnz_indices, A_nnz_values,
                                                    N, torch_device)  # Removed the arch argument
    timings['Taichi (sparse LLT)'] = solve_time_ti

    # 5. Solve using Warp's Sparse Solver
    # Pass the original PyTorch tensors holding nnz data and b_torch
    x_warp_torch, solve_time_warp = solve_warp_sparse(A_torch, b_torch,
                                                      A_nnz_indices, A_nnz_values,
                                                      N, torch_device)
    if solve_time_warp > 0.0 or not torch.isnan(x_warp_torch).any(): # Only add if Warp test ran
        timings['Warp (sparse)'] = solve_time_warp

    # 6. Compare Results and Timings
    # Pass all results (converted to PyTorch tensors on device)
    compare_results(x_torch, x_ti_torch, x_warp_torch, A_torch, b_torch, timings, torch_device)

    print("\nDone.")
