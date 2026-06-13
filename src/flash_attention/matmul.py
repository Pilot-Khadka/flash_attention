import torch
import triton
import triton.language as tl

torch.manual_seed(0)
DEVICE = triton.runtime.driver.active.get_active_torch_device()
"""
Consider matrix 3x4 (M*N)
    a11, a12, a13, a14
    a21, a22, a23, a24
    a31, a32, a33, a34

    row-major format -> [0, 1, 2] [3, 4, 5]
    i = 0 -> 0
    i = 1 -> 1

    col-access -> [0, 4, 8]
    i = 0 -> 0
    i = 1 -> i*4(N)
    
    ptrs -> []

    A (M*K) * B (K*N) = C (M*N)
    C[m,n] = sum A[m,k][B[k,n]]
    For row-major stride =1
    For col-major stride = 
"""


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m_mask = offs_m < M
    n_mask = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = B_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        acc = tl.dot(a, b, acc=acc)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    matmul_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return C


def test_matmul(size=(256, 256), atol=1e-2, rtol=1e-2):
    A = torch.randn(size, device=DEVICE, dtype=torch.float16)
    B = torch.randn(size, device=DEVICE, dtype=torch.float16)

    C_triton = triton_matmul(A, B)

    C_torch = torch.matmul(A, B).to(torch.float32)

    torch.testing.assert_close(C_triton, C_torch, atol=atol, rtol=rtol)
    print("Numerical test passed")


def benchmark(size=(1024, 1024), iters=100):
    A = torch.randn(size, device=DEVICE, dtype=torch.float16)
    B = torch.randn(size, device=DEVICE, dtype=torch.float16)

    for _ in range(10):
        triton_matmul(A, B)
        torch.matmul(A, B)

    torch.cuda.synchronize()

    import time

    triton_start = time.time()
    for _ in range(iters):
        triton_matmul(A, B)
    torch.cuda.synchronize()
    triton_time = (time.time() - triton_start) / iters

    torch_start = time.time()
    for _ in range(iters):
        torch.matmul(A, B)
    torch.cuda.synchronize()
    torch_time = (time.time() - torch_start) / iters

    print(f"Triton: {triton_time * 1e3:.3f} ms")
    print(f"PyTorch: {torch_time * 1e3:.3f} ms")


if __name__ == "__main__":
    test_matmul((128, 128))
    benchmark((1024, 1024))
