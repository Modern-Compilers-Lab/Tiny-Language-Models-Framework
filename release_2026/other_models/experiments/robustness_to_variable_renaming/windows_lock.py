"""Unix flock API for the frozen scorer on Windows, backed by real file locking."""

import os
import sys
import types


def install():
    if os.name != "nt" or "fcntl" in sys.modules:
        return
    import msvcrt

    module = types.ModuleType("fcntl")
    module.LOCK_EX = 2
    module.LOCK_UN = 8

    def flock(fd, operation):
        if hasattr(fd, "fileno"):
            fd = fd.fileno()
        position = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            if operation == module.LOCK_EX:
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            elif operation == module.LOCK_UN:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                raise ValueError("Only exclusive lock/unlock is used by the scorer")
        finally:
            os.lseek(fd, position, os.SEEK_SET)

    module.flock = flock
    sys.modules["fcntl"] = module
