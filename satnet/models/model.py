from typing import Any, Dict, List

import numpy as np
import tensorflow as tf
from gym import spaces
from gym.spaces import MultiDiscrete
from ray.rllib.models.tf import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.framework import get_activation_fn  # old ray

from satnet.models import plot_model


class SimpleModel(TFModelV2):
    """
    Simple fully connected model that allows for action masking.
    In this case, we're creating a TF model by subclassing the tf.Model class
    (see https://www.tensorflow.org/api_docs/python/tf/keras/Model)

    """

    def __init__(
        self, obs_space, action_space, num_outputs: int, model_config, name: str
    ):
        self.config = model_config
        super(SimpleModel, self).__init__(
            obs_space, action_space, num_outputs, self.config, name
        )
        # Build model - input from obs -> hidden layers -> actions + value prediction
        obs_shape = self.get_model_input_shape(obs_space)
        obs_input = tf.keras.layers.Input(shape=obs_shape, name="input_features")
        hidden_layer_sizes = self.config.get("fcnet_hidden_layer_sizes", [256, 256])
        action_logits, obs_embedding = self.build_policy_network(
            hidden_layer_sizes, obs_input, num_outputs
        )

        if self.config.get("vf_share_layers"):
            val_pred = self.append_val_pred(obs_embedding)
        else:
            val_pred = self.build_value_network(hidden_layer_sizes, obs_input)

        self.model = tf.keras.Model(
            # inputs=[obs_input, action_mask],
            inputs=obs_input,
            outputs=[action_logits, val_pred],
            name="SimpleModel",
        )

        plot_model(self.model, "SimpleMaskedModel.png")

    def get_model_input_shape(self, obs_space):
        original_space = obs_space.original_space
        if isinstance(original_space, spaces.Dict):
            obs_shape = original_space["obs"].shape
        elif isinstance(original_space, spaces.Tuple):
            # flatten and sum up all elements
            obs_shape = sum([int(np.prod(sp.shape)) for sp in original_space])
        elif not original_space.shape is None:
            obs_shape = original_space.shape
        else:
            raise ValueError(
                "Could not infer observation shape from space: {}".format(
                    original_space
                )
            )
        return obs_shape

    def append_val_pred(self, obs_embedding):
        val_pred = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=normc_initializer(0.01),
            name="value_out",
        )(obs_embedding)
        return val_pred

    def build_value_network(self, hidden_layer_sizes, obs_input):
        x = obs_input
        for i in range(0, len(hidden_layer_sizes)):
            x = tf.keras.layers.Dense(
                hidden_layer_sizes[i],
                activation=get_activation_fn(
                    self.config.get("fcnet_activation", "tanh")
                ),
                kernel_initializer=normc_initializer(1.0),
                name=f"fc_val_{i}",
            )(x)
        val_pred = self.append_val_pred(x)
        return val_pred

    def build_policy_network(self, hidden_layer_sizes, obs_input, num_actions):
        x = obs_input
        for i in range(0, len(hidden_layer_sizes)):
            x = tf.keras.layers.Dense(
                hidden_layer_sizes[i],
                activation=get_activation_fn(
                    self.config.get("fcnet_activation", "tanh")
                ),
                kernel_initializer=normc_initializer(1.0),
                name=f"fc_policy_{i}",
            )(x)
        obs_embedding = x
        action_logits = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=normc_initializer(0.01),
            name="action_logits",
        )(obs_embedding)
        return action_logits, obs_embedding

    def forward(self, input_dict: Dict[str, Any], state: List[Any], seq_lens: Any):
        """
        Call the model with the given obs_input tensors and state
        https://docs.ray.io/en/latest/rllib-models.html#ray.rllib.models.tf.tf_modelv2.TFModelV2.forward
        Any complex observations (dicts, tuples, etc.) will be unpacked by __call__() before
        being passed to forward(). To access the flattened observation tensor, refer to
        input_dict['obs_flat'].

        This method can be called any number of times. In eager execution, each call to forward()
        will eagerly evaluate the model. In symbolic execution, each call creates a computation
        graph that operates over the variables of this model. (i.e., shares weights)

        Parameters
        ----------
        input_dict : Dict[str, Any]
            Dictionary of obs_input tensors, including "obs", "obs_flat", "prev_action",
            "prev_reward", "is_training", "eps_id", "agent_id", "infos", and "t".
        state : List[Any]
            List of state tensors with sizes matching those returned by get_initial_state + the
            batch dimension.
        seq_lens : Tensor
            1-D tensor holding obs_input sequence lengths

        Returns
        -------
        outputs
        state
        """
        obs = input_dict["obs"]
        action_mask = obs["action_mask"]
        real_observation = obs["obs"]
        action_logits, val_pred = self.model(real_observation)
        # mask action_out logits with action_mask. In action_mask, valid = 1, invalid = 0
        # when mask = 0, tf.math.log(0) = -inf; we clip it at tf.float32.min to improve stability
        # when mask = 1, tf.math.log(1) = 0; 0 is higher than float32.min so we don't affect these elems
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = tf.math.add(action_logits, inf_mask, name="mask_logits")
        self._value_out = val_pred
        return masked_logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1], name="flatten_val_pred")

    def import_from_h5(self, h5_file: str) -> None:
        # Need to implement this according to IDE, but not sure how this works.
        pass
