def seed_everything(seed: int) -> None:
    """
    https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    """

    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # A handful of CUDA operations are nondeterministic if the CUDA version is
    # 10.2 or greater, unless the environment variable ``CUBLAS_WORKSPACE_CONFIG=:4096:8``
    # or ``CUBLAS_WORKSPACE_CONFIG=:16:8`` is set.
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # NOTE: upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation.
    # torch.use_deterministic_algorithms(True)