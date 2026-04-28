import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MyCnnExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor mapping to your old 'myCnn' function.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # Initialize the BaseFeaturesExtractor with the observation space and output dimension
        super().__init__(observation_space, features_dim)
        
        # We assume CxHxW images (channels first), which is standard for PyTorch
        n_input_channels = observation_space.shape[0]
        
        # Define the CNN layers
        # PyTorch padding=1 with kernel=3 is equivalent to TensorFlow's pad='SAME'
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute the flattened size by doing one dummy forward pass
        with torch.no_grad():
            dummy_tensor = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(dummy_tensor).shape[1]
            
        # Define the final linear layer mapping to the desired feature dimension
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Pass through CNN, then Linear layer
        return self.linear(self.cnn(observations))

# You no longer need the MyCnnPolicy class! SB3 handles that natively.
