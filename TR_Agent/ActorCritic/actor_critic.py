import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Multinomial, Normal
from TR_Agent.Utils.logger import logging



class RedispatchActivation(nn.Module):
    def __init__(self, bounds):
        """
        Initialize the activation function with generator bounds.
        :param bounds: A list or tensor of shape (n_gen,) containing the max value for each generator.
                       Example: [5, 10, 15] for 3 generators.
        """
        super().__init__()
        self.bounds = torch.tensor(bounds, dtype=torch.float32)  # Upper bounds for each generator

    def forward(self, x):
        """
        Scales and constrains the input values x to the range [0, bound] for each generator.
        :param x: Input tensor of shape (batch_size, n_gen) representing raw redispatch values.
        :return: Scaled tensor of shape (batch_size, n_gen) with values constrained to the given bounds.
        """
        x = torch.sigmoid(x)  # Scale to the range [0, 1]
        x = x * self.bounds.to(x.device)  # Scale to [0, bound] using element-wise multiplication
        return x


class ActorCritic(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network = nn.Sequential(
                        nn.Linear(self.config.input_dim, 512),
                        nn.ReLU(),
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
        )

        self.topo_critic = nn.Sequential(
                            nn.Linear(self.config.input_dim + self.config.action_dim, 512),  
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 1)  
                            )



        self.redispatch_critic = nn.Sequential(
                            nn.Linear(self.config.input_dim + self.config.n_gen, 512),  
                            nn.ReLU(),
                            nn.Linear(512, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 1)  
                            )


        self.topology = nn.Linear(256, self.config.action_dim)
        self.redispatch_mean = nn.Linear(256, self.config.n_gen)  # Mean for continuous actions
        self.redispatch_log_std = nn.Linear(256, self.config.n_gen)  # Log std for continuous actions

        # Redispatch activation to enforce bounds
        self.redispatch_activation = RedispatchActivation(bounds=[5, 10, 15])
        

        self.to(self.device)
        logging.info(f"model initiated with GPU")
    
    def forward(self, x):
        x = torch.tensor(x)

        x = self.network(x)

        topo = self.topology(x)

        gen_mean = self.redispatch_mean(x)  # Mean
        gen_mean = self.redispatch_activation(gen_mean)

        gen_log_std = self.redispatch_log_std(x)  # Log standard deviation
        gen_std = torch.exp(gen_log_std)  # Standard deviation (ensure positivity)

        return topo, gen_mean, gen_std
    


    def act(self, obs):
        topo, gen_mean, gen_std = self.forward(obs)

        # topology action
        action_probs = F.softmax(topo)
        action_distribution = Categorical(action_probs)
        topo_actions = action_distribution.sample()


        # Redispatching action
        redispatch_distribution = Normal(gen_mean, gen_std)
        redispatch_action = redispatch_distribution.sample()  # Continuous action values

        return topo_actions, redispatch_action
    


