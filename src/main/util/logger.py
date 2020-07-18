import logging


def initLogger(module_name):
    log = logging.getLogger(module_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(filename)s - %(message)s', '%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log
