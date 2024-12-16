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
        

        self.logprobs = []
        self.cont_logprobs = []

        self.topo_state_values = []
        self.redis_state_values = []
        
        self.rewards = []

        self.to(self.device)
        logging.info(f"model initiated with GPU")


    
    def forward(self, x):
        x = torch.from_numpy(x, dtype=torch.float)

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


        self.logprobs.append(action_distribution.log_prob(topo_actions))
        self.cont_logprobs.append(redispatch_distribution.log_prob(redispatch_action).sum())

        # action value from critic
        action_value = torch.cat([obs, topo_actions], dim=-1)
        self.topo_state_values.append(self.topo_critic(action_value))

        # continuous action value from critic
        c_action = torch.cat([obs, redispatch_action], dim=-1)
        self.redis_state_values.append(self.redispatch_critic(c_action))

        return topo_actions, redispatch_action
    


    def BoostrappingTopoLoss(self):
        topo_loss = 0

        # TD Target Computation
        for t in range(len(self.rewards) - 1):  # Exclude final step (no future state)
            reward = self.rewards[t]
            next_value = self.topo_state_values[t + 1].detach()  # Next state's value
            td_target = reward + self.config.gamma * next_value  # Bootstrapped target

            value = self.topo_state_values[t]  # Current state's value
            logprob = self.topo_log_probs[t]  # Log prob of action

            # Compute advantage (TD Error)
            advantage = td_target - value.item()

            # Actor Loss
            action_loss = -logprob * advantage

            # Critic Loss
            value_loss = F.smooth_l1_loss(value, td_target)

            # Combine Losses
            topo_loss += (action_loss + value_loss)

        # Handle the last step separately (no bootstrapping)
        final_value = self.topo_state_values[-1]
        final_reward = self.rewards[-1]
        value_loss = F.smooth_l1_loss(final_value, final_reward)
        topo_loss += value_loss

        return topo_loss




    def TopoLoss(self):
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + self.config.gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        topo_loss = 0
        for logprob, value, reward in zip(self.logprobs, self.topo_state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            topo_loss += (action_loss + value_loss)   
        return topo_loss
    


    def RedispatchLoss(self):
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + self.config.gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        redis_loss = 0
        for logprob, value, reward in zip(self.cont_logprobs, self.redis_state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            redis_loss += (action_loss + value_loss)   
        return redis_loss
    


    def learn(self):
        pass


        # Critic for topology action



        # Critic for redispatch action
    


