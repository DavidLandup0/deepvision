backend = "tensorflow"


def set_backend(backend_name):
    global backend
    backend = backend_name


def get_backend():
    global backend
    return backend
