import os
import platform
from datetime import datetime

HOME = os.path.join(os.path.expanduser("~"), "ray_models")
IS_PPC = "ppc64le" in platform.processor()


def plot_model(model, model_name):
    # import tensorflow as needed, since it takes a long time to import
    try:
        tf.__version__
    except NameError:
        # import tensorflow if we can't find it
        import tensorflow as tf

    if not os.path.isdir(HOME):
        os.makedirs(HOME, exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H%M")
    model_name = model_name.replace(".png", "")
    model_name = f"{model_name}_{timestamp}.png"
    model_path = os.path.join(HOME, model_name)

    if not os.path.isfile(model_path) and not IS_PPC:
        print("Plotting model!")
        tf.keras.utils.plot_model(
            model,
            model_path,
            show_shapes=True,
        )
