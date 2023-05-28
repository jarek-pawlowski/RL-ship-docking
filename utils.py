import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomMLP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim: int = 64, side_length: int = 10):
        super().__init__(observation_space, features_dim)
        mlp_hidden_dim = 64
        #self.input_dim = int(observation_space.shape[0])
        self.input_dim = int(4*side_length+2*(2*side_length+1)+4)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, mlp_hidden_dim),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(nn.Linear(mlp_hidden_dim, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations are one-hot encoded
        #obs = torch.argmax(torch.reshape(observations, (self.input_dim,-1)), dim=1)/(side_length-1)  # one hot to float: argmax to decode
        #r = self.linear(self.mlp(obs))
        #return torch.reshape(r,(1,-1))
        r = self.linear(self.mlp(observations))
        return r