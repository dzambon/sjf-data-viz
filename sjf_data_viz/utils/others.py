def dict_to_string(d, indent=0, output="", only_type=True):
    """ Nice dictionary visualization """
    if isinstance(d, dict):
        for k, v in d.items():
            val = type(v).__name__ if only_type else v
            msg = "{ind}{key}: \t{val}".format(ind="".join([" "]*indent), key=k, val=val)
            # print(msg)
            output += msg + "\n"
            output = dict_to_string(v, indent=indent + 2, output=output, only_type=only_type)
    return output
