from typing import Dict, Any


def dict_to_cpu(gpu_dictionary: Dict[str, Any]):
    cpu_dictionary = {}
    for key, value in gpu_dictionary.items():
        if isinstance(value, list):
            cpu_value = [tensor.cpu().numpy() for tensor in value]
        else:
            cpu_value = value.cpu().numpy()
        cpu_dictionary[key] = cpu_value
    return cpu_dictionary
