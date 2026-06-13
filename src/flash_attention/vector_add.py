import torch
import triton
import triton.language as tl

torch.manual_seed(0)
DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):

    # because grid (10, )
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # to prevent reading/adding elements beyond n_elements
    # if the n_element is not a power of 2, will have empty space in block_size
    # random values could be read, oob access
    mask = offsets < n_elements

    # load data from DRAM/VRAM/HBM to SRAM
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # shape (BLOCK_SIZE)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    output = x + y

    # write data back to DRAM
    tl.store(pointer=output_ptr + offsets, value=output, mask=mask)


def add(
    x: torch.Tensor,
    y: torch.Tensor,
    block_size: int = 1024,
):
    # preallocate the output
    # gpu kernels write into preallocated memory than allocating themselves
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    n_elements = output.numel()

    # example:
    # n_elements =10,000
    # block_size = 1024
    # one kernel instance handles 1024 elements
    # we need 10,000/1024 = 9.76 ~10 kernel instances
    # cdiv  = ceil(a/b)
    # grid returns (10, )
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # block size: one triton program responsible for 1024 elements
    # eg: gpu has 80 SMs
    # grid: (10, )
    # SM0 -> Program 0
    # SM1 -> Program 1 and so on.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    return output


def test_add_kernel(
    size: tuple,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    device=DEVICE,
):
    if isinstance(size, int):
        size = (size,)

    x = torch.randn(size=size, device=device)
    y = torch.randn(size=size, device=device)

    z_tri = add(x, y)
    z_pytorch = x + y

    torch.testing.assert_close(z_tri, z_pytorch, atol=atol, rtol=rtol)
    print("Numerical test passed")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    quantiles = [0.5, 0.2, 0.8]
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    def operation():
        return x + y

    if provider == "torch":
        median_ms, min_ms, max_ms = triton.testing.do_bench(
            fn=operation,
            quantiles=quantiles,
        )
    if provider == "triton":
        median_ms, min_ms, max_ms = triton.testing.do_bench(
            fn=operation,
            quantiles=quantiles,
        )

    # calculate memory bandwidth
    # 3 because
    # read x
    # read y
    # write output
    bytes_moved = 3 * x.numel() * x.element_size()

    def to_gbps(time_ms):
        seconds = time_ms / 1000
        gigabytes = bytes_moved / 1e9
        return gigabytes / seconds

    return (
        to_gbps(median_ms),
        to_gbps(max_ms),
        to_gbps(min_ms),
    )


if __name__ == "__main__":
    test_add_kernel(size=(1024, 1024))
    test_add_kernel(size=(1023, 1023))
    test_add_kernel(size=(1025, 1025))
    benchmark.run(print_data=True, show_plots=True)
