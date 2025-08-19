import os
import importlib
import inspect
from torch.utils.data import Dataset

# This function will find and load all classes that inherit from Datasets
def _load_dataset_classes():
    dataset_classes = {}

    # Search the current directory for Python files    
    package_dirname = os.path.dirname(os.path.abspath(__file__))    
    package_name = os.path.basename(package_dirname)
    for _, _, files in os.walk(package_dirname):
        for file in files:
            if file.endswith(".py") and file != __file__:
                module_name = file[:-3]
                module = importlib.import_module(f"{package_name}.{module_name}")
                # Inspect all classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if the class is a subclass of Datasets and has DATASET_NAME attribute
                    if issubclass(obj, Dataset) and hasattr(obj, 'DATASET_NAME'):
                        name = obj.DATASET_NAME if type(obj.DATASET_NAME) in [tuple, list] else [obj.DATASET_NAME]
                        for n in name:
                            dataset_classes[n.lower()] = obj
                        
    return dataset_classes

def get_dataset(name, dataset_opts):
    dataset_classes = _load_dataset_classes() 
    
    assert name in dataset_classes, "Dataset {} not available".format(name)

    DATASET_CLASS = dataset_classes[name.lower()]

    dataset = DATASET_CLASS(**dataset_opts)

    return dataset

def get_centerdir_dataset(name, dataset_opts, centerdir_gt_opts=None, centerdir_groundtruth_op=None, no_groundtruth=False):
    dataset = get_dataset(name, dataset_opts)

    if no_groundtruth:
        dataset.return_gt_heatmaps = False
        dataset.return_gt_box_polygon = False
        dataset.return_gt_polygon = False
        dataset.return_image = True
        return dataset, None

    if centerdir_gt_opts is not None and len(centerdir_gt_opts) > 0:
        from .CenterDirGroundtruthDataset import CenterDirGroundtruthDataset
        from models.center_groundtruth import CenterDirGroundtruth

        if centerdir_groundtruth_op is None:
            centerdir_groundtruth_op = CenterDirGroundtruth(**centerdir_gt_opts)

        dataset = CenterDirGroundtruthDataset(dataset, centerdir_groundtruth_op)
        return dataset, centerdir_groundtruth_op
    else:
        return dataset, None


