import os


def get_world_size() -> int:
    print("[distributed] Reading world size", flush=True)
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    print("[distributed] Reading rank", flush=True)
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    print("[distributed] Reading local rank", flush=True)
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    print("[distributed] Checking distributed mode", flush=True)
    return get_world_size() > 1


def is_main_process() -> bool:
    print("[distributed] Checking main process", flush=True)
    return get_rank() == 0
