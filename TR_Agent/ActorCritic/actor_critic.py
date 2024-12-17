import os
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
                            nn.Linear(self.config.input_dim + 1, 512),  
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
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)

        self.logprobs = []
        self.cont_logprobs = []

        self.topo_state_values = []
        self.redis_state_values = []
        
        self.rewards = []


        
        self.min_means = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        self.max_means = torch.tensor([5, 10, 15], dtype=torch.float32, device=self.device)
        self.min_std = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=self.device)
        self.max_std = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=self.device)

        self.to(self.device)
        logging.info(f"model initiated with GPU")


    
    def forward(self, x):

        x = self.network(x)

        topo = self.topology(x)

        gen_mean = self.redispatch_mean(x)  # Mean

        gen_log_std = self.redispatch_log_std(x)  # Log standard deviation
        
        constrained_means = torch.sigmoid(gen_mean) * (self.max_means - self.min_means) + self.min_means
        
        # Constrain the log std using a tanh function to map it to a reasonable range
        constrained_log_std = torch.tanh(gen_log_std) * (self.max_std - self.min_std) + self.min_std
        
        # Exponentiate to get the standard deviation
        constrained_std = torch.exp(constrained_log_std)
        
        #gen_std = torch.exp(gen_log_std)  # Standard deviation (ensure positivity)

        return topo, constrained_means, constrained_std
    

        

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        topo, gen_mean, gen_std = self.forward(obs)
        

        # topology action
        action_probs = F.softmax(topo, dim=-1)
        action_distribution = Categorical(action_probs)
        topo_actions = action_distribution.sample()
        topo_actions = topo_actions.unsqueeze(-1)

        # Redispatching action
        redispatch_distribution = Normal(gen_mean, gen_std)
        redispatch_action = redispatch_distribution.sample()  # Continuous action values


        self.logprobs.append(action_distribution.log_prob(topo_actions.squeeze(-1)))
        self.cont_logprobs.append(redispatch_distribution.log_prob(redispatch_action).sum())

        # action value from critic
        action_value = torch.cat([obs, topo_actions.float()], dim=-1)
        self.topo_state_values.append(self.topo_critic(action_value))

        # continuous action value from critic
        c_action = torch.cat([obs, redispatch_action], dim=-1)
        self.redis_state_values.append(self.redispatch_critic(c_action))

        return topo_actions.item(), redispatch_action
    


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
    
    
    
    def save_model(self, model_name="actor_critic.pt"):
        torch.save(self.state_dict(), os.path.join(self.config.path, model_name))
        print(f"model saved at {self.config.path}")


    def load_model(self, model_name="actor_critic.pt"):
        self.load_state_dict(torch.load(os.path.join(self.config.path, model_name)))
        print(f"model loaded at {self.config.path}")