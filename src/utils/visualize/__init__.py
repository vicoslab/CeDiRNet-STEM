
from .center_with_shape import CentersShapeVisualizeTest, CentersShapeVisualizeTrain

def get_visualizer(name, opts):
    if name == 'CentersShapeVisualizeTest':
        return CentersShapeVisualizeTest(**opts)
    elif name == 'CentersShapeVisualizeTrain':
        return CentersShapeVisualizeTrain(**opts)
    else:
        raise Exception("Unknown visualizer: '%s'" % name)