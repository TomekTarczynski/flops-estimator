"""
GPU GEMM Benchmark Script
-------------------------
The script performs GEMM benchmarks to estimate actual peak flops.
It uses pytorch to run GEMM on GPU.
The results are stored in YAML file
"""

import argparse
import gc
import logging
import subprocess
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
import yaml

TERA = 1e12
device = torch.device('cuda:0')

def get_power_limit() -> float:
    """
    Retrieve the max GPU Power Limit [W] using nvidia-smi

    Returns:
        float: Power limit in watts.
    """
    
    result = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits", "--id=0"], capture_output=True, text=True, check=True)
    logging.debug(f"Result in get_power_limit():\n{result.stdout}")
    return float(result.stdout.strip())

def get_sm_clock() -> float:
    """
    Retrieve max clock frequency of a GPU core.

    Returns:
        float: Max clock frequency in MHz
    """
    
    result = subprocess.run(["nvidia-smi", "--query-gpu=clocks.max.sm", "--format=csv,noheader,nounits", "--id=0"], capture_output=True, text=True, check=True)
    logging.debug(f"Result in get_sm_clock():\n{result.stdout}")
    return float(result.stdout.strip())   

def benchmark_gemm(M: int, N: int, K: int, device: torch.device, dtype: torch.dtype, n_warmup: int, n_repeat: int) -> dict:
    """
    Return average empirical flops for dense matrix multiplication.
    The function computes C = A @ B, where:
        - A is of shape [M, N]
        - B is of shape [N, K]
        - C is of shape [M, K]
    The final result is median among 'n_repeat' repetitions.

    Args:
        M (int) - dimension of a matrix
        N (int) - dimension of a matrix
        K (int) - dimension of a matrix
        device (string) - The device on which matrix multiplication should be done
        dtype - The dtype (torch.float32 or torch.float16) used to store the data
        n_warmup (int) - The number of warmup matrix multiplications, before the flops are measured
        n_repeat (int) - Number of times the matrix multiplication is done.

    Returns:
        dict: Dictionary containing all the information about the input and the results
    """

    A = torch.rand(M, N, device = device, requires_grad=False, dtype = dtype)
    B = torch.rand(N, K, device = device, requires_grad=False, dtype = dtype)
    C = torch.rand(M, K, device = device, requires_grad=False, dtype = dtype)
    
    total_time_s = []
    tflops = []
    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)

    n_ops = int(2 * M * N * K)
    
    for i in range(n_warmup):
        torch.mm(A, B, out=C)
        
    for i in range(n_repeat):    
        start.record()
        with torch.inference_mode():
            torch.mm(A, B, out=C)
        end.record()
        torch.cuda.synchronize()
        total_time_s.append(start.elapsed_time(end) / 1000) # measured in seconds
        tflops.append(n_ops / total_time_s[i] / TERA)

    del A
    del B
    del C
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    result = {
        "M": int(M),
        "N": int(N),
        "K": int(K),
        "n_ops": int(2 * M * N * K),
        "time_elapsed": total_time_s,
        "flops": tflops,
        "median_flops": float(np.median(np.array(tflops))),
        "dtype": str(dtype)
    }

    return result

def get_device_information(sm_cores: dict) -> dict:
    """
    Retrieves all essential information about the GPU.

    Args:
        sm_cores (dict): The dictionary that maps compute capabilities into number of cores per Streaming Multiprocessor

    Returns:
        dict: Details about the GPU
    """
    
    props = torch.cuda.get_device_properties(0)
    
    device_information = {
        "device_name": torch.cuda.get_device_name(0),
        "number_of_sm": props.multi_processor_count,
        "power_limit": {
            "value": get_power_limit(),
            "unit": "W"
        },
        "compute_capability": (props.major, props.minor),
        "cores_per_sm": sm_cores['cuda_cores_per_sm'][f"{props.major}.{props.minor}"],
        "sm_clock": {
            "value": get_sm_clock(),
            "unit": "MHz"
        },
        "memory": {
            "value": round(torch.cuda.mem_get_info()[1] / (1024**3), 2),
            "unit": "GiB"
        }
    }
    device_information["total_number_of_cores"] = device_information["number_of_sm"] * device_information["cores_per_sm"]
    device_information["theoretical_flops"] = {
        "value": round((device_information["total_number_of_cores"] * device_information["sm_clock"]["value"] * 2 * 1e6) / TERA, 2), # Multpied by 2 because add and multiply are made in a single operation
        "unit": "tflops",
        "precision": "float32"
    }
    
    logging.info(f"Device information:\n{device_information}")    
    return device_information

def get_test_cases(device_information: dict, memory_overhead: float = 0.02) -> pd.DataFrame:
    """
    Create test cases for benchmarking matrix multiplication.
    For simplicity only square matrices are multiplies.
    All test cases of form 2 * power of two or 3 * power of two are created.
    Test cases are filtered to those, which fit int the memory.

    Args:
        device_information (dict): The information about GPU
        memory_overhead (float): Memory overhead, measures in percentages, needed for matrix multiplication.

    Returns:
        pd.DataFrame: DataFrame containing benchmark case. It consists of following columns:
            o M - The dimension of a matrix
            o N - The dimension of a matrix
            o K - The dimension of a matrix
            o dtype - The type of float that should be used during matrix multiplication 
    """
    dim_size = [2 * 2**i for i in range(20)] + [3 * 2**i for i in range(20)]
    all_dims = list(product(dim_size, (torch.float32, torch.float16)))
    df = pd.DataFrame({
        "M": [x[0] for x in all_dims],
        "N": [x[0] for x in all_dims],
        "K": [x[0] for x in all_dims],
        "dtype": [x[1] for x in all_dims]
    })
    df["total_memory"] = (df["M"] * df["N"] + df["N"] * df["K"] + df["M"] * df["K"]) * df["dtype"].apply(lambda x: x.itemsize)
    df = df.loc[df["total_memory"] < (device_information["memory"]["value"] * 1024**3 / (1 + memory_overhead))]
    
    dtype_order = [torch.float16, torch.float32]
    df["dtype"] = pd.Categorical(df["dtype"], categories = dtype_order, ordered=True)
    df = df.sort_values(by=["dtype", "total_memory"]).reset_index(drop=True)
    return df


def parse_arguments() -> argparse.Namespace:
    """
    Parses CLI arguments.

    Returns:

        args: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--memory_overhead', type=float, default=0.02, help="Assumed memory overhead needed for multiplications. Measured in percentages (eg. 0.02)")
    parser.add_argument('--output_file', type=str, required=True, help="The output file in which results will be stored")
    parser.add_argument('--n_warmup', type=int, default=20, help="Number of warmup GEMM done before flops are measured")
    parser.add_argument('--n_repeat', type=int, default=25, help="Number of times GEMM is repeated. The results are averaged")
    
    args = parser.parse_args()

    assert args.memory_overhead >= 0, "Memory overhead cannot be negative!"
    

    return args

def main() -> None:
    """
    The main function
    """

    args = parse_arguments()
        
    if not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available!")
    else:
        logging.info("Cuda is available")
    
    
    with open("sm_cores.yaml", "r") as file:
        sm_cores = yaml.safe_load(file)
        logging.info("sm_cores.yaml loaded successfully")    
    
    device_information = get_device_information(sm_cores = sm_cores)
    test_cases_df = get_test_cases(device_information = device_information, memory_overhead = args.memory_overhead)
    
    gemm_results = []
    for row in test_cases_df.itertuples(index=False):
        gemm_results.append(benchmark_gemm(
            M = row.M,
            N = row.N,
            K = row.K,
            device = device,
            dtype = row.dtype,
            n_warmup = args.n_warmup,
            n_repeat = args.n_repeat))
        logging.info(f"N: {gemm_results[-1]['N']} dtype: {gemm_results[-1]['dtype']} flops: {gemm_results[-1]['median_flops']:.2f}")

    max_flops16 = max([r['median_flops'] for r in gemm_results if r['dtype'] == 'torch.float16'])
    max_flops32 = max([r['median_flops'] for r in gemm_results if r['dtype'] == 'torch.float32'])
    min_efficient_size16 = min([r['N'] for r in gemm_results if (r['dtype'] == 'torch.float16') and (r['median_flops'] > 0.95 * max_flops16)])
    min_efficient_size32 = min([r['N'] for r in gemm_results if (r['dtype'] == 'torch.float32') and (r['median_flops'] > 0.95 * max_flops32)])
    
    output = {
        'device_information': device_information,
        'flops_summary': {
            'max_flops16': {"value": max_flops16, "unit": "teraflops"},
            'max_flops32': {"value": max_flops32, "unit": "teraflops"},
            'flops16_speedup': max_flops16 / max_flops32,
            'efficiency_flops16': max_flops16 / device_information['theoretical_flops']['value'],
            'efficiency_flops32': max_flops32 / device_information['theoretical_flops']['value'],
            'min_efficient_size16': min_efficient_size16,
            'min_efficient_size32': min_efficient_size32
        },
        'flops_details': gemm_results
    }
            
    with open(args.output_file, "w") as file:
        yaml.safe_dump(output, file, default_flow_style=False, indent=2, sort_keys=False)    

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True)
    
    main()
