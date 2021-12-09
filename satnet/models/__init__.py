from .model import SimpleModel
from .utils import plot_model

# # register SimpleModel in ray's model catalog
# from ray.rllib.models import ModelCatalog
# ModelCatalog.register_custom_model("baseline_model", SimpleModel)
# ModelCatalog.register_custom_model("simple_model", SimpleModel)
#
