import torch


def is_hopper_gpu():
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability(0)
        major, minor = device_capability
        return major == 9
    return False
