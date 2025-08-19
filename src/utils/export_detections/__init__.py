
from .center import CenterExport

def get_exporter(name, opts):
    if name == 'CenterExport':
        return CenterExport(**opts)
    else:
        raise Exception("Unknown exporter: '%s'" % name)