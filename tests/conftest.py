"""Test-session settings for running the suite on a CPU-only machine.

Nothing here changes a GPU run.

* Inductor's vectorized C++ backend (torch 2.6) fails to compile the two-tier
  eviction graph on CPU tensors. The tests that compile the eviction on CPU use
  scalar C++ codegen instead; CUDA tensors compile through Triton as before.
* A CUDA build of torch ships Triton. On a machine with no GPU, Triton has no
  active driver, and ``triton.autotune`` raises when the score kernel module is
  imported. The tuned kernel never launches on CPU, so the decorator is made an
  identity there.
"""

import torch
import torch._inductor.config as _inductor_config

_inductor_config.cpp.simdlen = 0

if not torch.cuda.is_available():
    try:
        import triton
    except ImportError:
        triton = None
    if triton is not None:
        triton.autotune = lambda *args, **kwargs: (lambda fn: fn)
