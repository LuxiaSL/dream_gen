"""
VRAM Profiler - Detailed GPU memory diagnostics

Provides detailed VRAM usage information when OOM or other GPU memory
issues are detected. Helps diagnose memory leaks, fragmentation, and
contention between PyTorch and ComfyUI.
"""

import logging
import subprocess
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_nvidia_smi_info() -> Dict[str, Any]:
    """
    Get detailed GPU info from nvidia-smi.
    
    Returns:
        Dictionary with GPU memory and process info
    """
    try:
        # Get memory info
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 5:
                gpu_info = {
                    'total_mb': int(parts[0]),
                    'used_mb': int(parts[1]),
                    'free_mb': int(parts[2]),
                    'utilization_pct': int(parts[3]),
                    'temperature_c': int(parts[4])
                }
            else:
                gpu_info = {'raw': result.stdout.strip()}
        else:
            gpu_info = {'error': result.stderr}
        
        # Get process info
        proc_result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory,name',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        
        processes = []
        if proc_result.returncode == 0 and proc_result.stdout.strip():
            for line in proc_result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) >= 3:
                    processes.append({
                        'pid': int(parts[0]),
                        'memory_mb': int(parts[1]),
                        'name': parts[2]
                    })
        
        gpu_info['processes'] = processes
        return gpu_info
        
    except Exception as e:
        return {'error': str(e)}


def get_pytorch_memory_info() -> Dict[str, Any]:
    """
    Get detailed PyTorch CUDA memory info.
    
    Returns:
        Dictionary with detailed memory breakdown
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        # Basic stats
        info = {
            'allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
            'reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            'max_reserved_mb': torch.cuda.max_memory_reserved() / 1024 / 1024,
        }
        
        # Memory stats if available
        try:
            stats = torch.cuda.memory_stats()
            info['stats'] = {
                'active_bytes': stats.get('active_bytes.all.current', 0),
                'inactive_bytes': stats.get('inactive_split_bytes.all.current', 0),
                'allocated_bytes': stats.get('allocated_bytes.all.current', 0),
                'reserved_bytes': stats.get('reserved_bytes.all.current', 0),
                'num_alloc_retries': stats.get('num_alloc_retries', 0),
                'num_ooms': stats.get('num_ooms', 0),
            }
        except Exception:
            pass
        
        # Tensor summary if available
        try:
            # Get all tensors on GPU (this can be expensive)
            tensor_count = 0
            tensor_bytes = 0
            for obj in torch.cuda._memory._state.get_allocations():
                tensor_count += 1
                tensor_bytes += obj.size
            info['tensor_count'] = tensor_count
            info['tensor_bytes'] = tensor_bytes
        except Exception:
            pass
        
        return info
        
    except Exception as e:
        return {'error': str(e)}


def dump_vram_on_oom(context: str = "unknown", exception: Optional[Exception] = None) -> None:
    """
    Dump detailed VRAM information when OOM or memory issues occur.
    
    Call this in exception handlers when CUDA OOM is detected.
    
    Args:
        context: Description of where the error occurred
        exception: The exception that triggered the dump
    """
    logger.error("=" * 70)
    logger.error(f"VRAM DIAGNOSTIC DUMP - Context: {context}")
    logger.error("=" * 70)
    
    if exception:
        logger.error(f"Exception: {type(exception).__name__}: {exception}")
    
    # PyTorch info
    logger.error("-" * 40)
    logger.error("PyTorch CUDA Memory:")
    logger.error("-" * 40)
    pytorch_info = get_pytorch_memory_info()
    if 'error' not in pytorch_info:
        logger.error(f"  Allocated: {pytorch_info.get('allocated_mb', 0):.1f} MB")
        logger.error(f"  Reserved:  {pytorch_info.get('reserved_mb', 0):.1f} MB")
        logger.error(f"  Max Alloc: {pytorch_info.get('max_allocated_mb', 0):.1f} MB")
        logger.error(f"  Max Rsrvd: {pytorch_info.get('max_reserved_mb', 0):.1f} MB")
        
        if 'stats' in pytorch_info:
            stats = pytorch_info['stats']
            logger.error(f"  Alloc Retries: {stats.get('num_alloc_retries', 0)}")
            logger.error(f"  Total OOMs: {stats.get('num_ooms', 0)}")
    else:
        logger.error(f"  Error: {pytorch_info['error']}")
    
    # nvidia-smi info
    logger.error("-" * 40)
    logger.error("nvidia-smi GPU Memory:")
    logger.error("-" * 40)
    nvidia_info = get_nvidia_smi_info()
    if 'error' not in nvidia_info:
        logger.error(f"  Total:  {nvidia_info.get('total_mb', 0)} MB")
        logger.error(f"  Used:   {nvidia_info.get('used_mb', 0)} MB")
        logger.error(f"  Free:   {nvidia_info.get('free_mb', 0)} MB")
        logger.error(f"  GPU Util: {nvidia_info.get('utilization_pct', 0)}%")
        logger.error(f"  Temp:   {nvidia_info.get('temperature_c', 0)}°C")
        
        if nvidia_info.get('processes'):
            logger.error("-" * 40)
            logger.error("GPU Processes:")
            for proc in nvidia_info['processes']:
                logger.error(f"  PID {proc['pid']}: {proc['memory_mb']} MB - {proc['name']}")
        else:
            logger.error("  No GPU processes found (or query failed)")
    else:
        logger.error(f"  Error: {nvidia_info.get('error', 'unknown')}")
    
    # Memory calculation
    logger.error("-" * 40)
    logger.error("Memory Analysis:")
    logger.error("-" * 40)
    if 'error' not in nvidia_info and 'error' not in pytorch_info:
        total = nvidia_info.get('total_mb', 0)
        used = nvidia_info.get('used_mb', 0)
        pytorch_reserved = pytorch_info.get('reserved_mb', 0)
        
        # Sum up process memory
        process_total = sum(p.get('memory_mb', 0) for p in nvidia_info.get('processes', []))
        
        unaccounted = used - process_total
        
        logger.error(f"  nvidia-smi reports {used} MB used of {total} MB")
        logger.error(f"  Sum of process memory: {process_total} MB")
        logger.error(f"  PyTorch reserved: {pytorch_reserved:.1f} MB")
        logger.error(f"  Unaccounted: {unaccounted} MB")
        
        if unaccounted > 1000:  # More than 1GB unaccounted
            logger.error("  WARNING: Large unaccounted memory! Possible causes:")
            logger.error("    - GPU memory fragmentation")
            logger.error("    - Other processes using GPU (check nvidia-smi)")
            logger.error("    - Memory held by CUDA context")
            logger.error("    - Driver-level allocations")
    
    logger.error("=" * 70)
    logger.error("END VRAM DIAGNOSTIC DUMP")
    logger.error("=" * 70)


def is_oom_error(exception: Exception) -> bool:
    """
    Check if an exception is a CUDA out-of-memory error.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if this is a CUDA OOM error
    """
    error_str = str(exception).lower()
    return any(phrase in error_str for phrase in [
        'out of memory',
        'cuda out of memory',
        'allocate',
        'oom',
    ])

