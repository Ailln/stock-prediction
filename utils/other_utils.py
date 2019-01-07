def bool(arg):
    arg = str(arg).lower()
    if arg == "true":
        return True
    elif arg == "false":
        return False
    else:
        raise ValueError