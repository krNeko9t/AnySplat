import atexit
from multiprocessing import RLock
from threading import RLock as ThreadRLock

import torch
from jaxtyping import Int64
from torch import Tensor
from torch.multiprocessing import Manager


class StepTracker:
    lock: RLock
    step: Int64[Tensor, ""]

    def __init__(self, use_manager: bool = True):
        """Track global step across processes.

        - **use_manager=True**: use a multiprocessing Manager lock for dataloader workers.
        - **use_manager=False**: use a thread lock (faster, avoids Manager shutdown noise). Suitable when
          `num_workers=0`.
        """
        self._manager = None
        if use_manager:
            # Use a Manager lock so dataloader worker processes can read training progress.
            # Register shutdown to avoid noisy multiprocessing/NFS cleanup errors at exit.
            self._manager = Manager()
            atexit.register(self._shutdown)
            self.lock = self._manager.RLock()
        else:
            self.lock = ThreadRLock()
        self.step = torch.tensor(0, dtype=torch.int64).share_memory_()

    def _shutdown(self) -> None:
        mgr = getattr(self, "_manager", None)
        if mgr is None:
            return
        try:
            mgr.shutdown()
        except Exception:
            pass

    def set_step(self, step: int) -> None:
        with self.lock:
            self.step.fill_(step)

    def get_step(self) -> int:
        with self.lock:
            return self.step.item()
