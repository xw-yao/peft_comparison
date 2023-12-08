import re

def parse_config_string(s):
    param_pattern = r"(\w+)=(\w+\.?\w*)"

    prefix = s.split('[')[0]

    params = {}
    for key, value in re.findall(param_pattern, s):
        try:
            # Try converting to float first
            params[key] = float(value) if '.' in value else int(value)
        except ValueError:
            # If conversion fails, keep as string
            params[key] = value

    return prefix, params
