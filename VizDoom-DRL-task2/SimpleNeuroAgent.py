import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#Let's Assume a Single State for Simplicity
state = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
#First Action is Correct
ground_truth = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=False) 
mse_loss =nn.MSELoss()
num_iterations = 40
#True for Gradient Descent, False for Gradient Ascent
USE_DESCENT = False 

#Create Simple Neural Network
input_dim = state.size()[0]
hidden_dim = 20
output_dim = ground_truth.size()[0]
layers = [
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim),
]
model = nn.Sequential(*layers)
#optimizer = optim.SGD(model.parameters(), lr=0.05)
optimizer = optim.Adam(model.parameters(), lr=0.01)
print(model)

#Train Neural Network for 1 State and 4 Actions, where
#Goal is to Increase Probability of First Action
for i in range(0, num_iterations):
    #Forward Pass
    prediction = model(state) 
    output = torch.distributions.Categorical(logits=prediction)
    a = output.sample()
    if USE_DESCENT:
        #Positive Loss for Gradient Descent
        loss = mse_loss(output.probs, ground_truth) 
        signal = "error"
    else:
        correct_action = torch.argmax(ground_truth).item()
        #Simple Reward Function
        reward = 1.0 if a.item() == correct_action else 0.0  
        #Negative Log-Probability (Maximise Action Probability)
        loss = -output.log_prob(a) * reward  
        signal = "reward"

    signal_value = loss.detach().numpy()
    print("i=%s, a=%s, pred=%s, probs=%s, %s=%s" % \
	        (i, a, prediction.detach(), output.probs.detach(), signal, signal_value))

    #Reset Gradients
    optimizer.zero_grad() 
    #Compute Gradients
    loss.backward() 
    #print("state.grad="+str(state.grad))
    #print("ground_truth.grad="+str(ground_truth.grad))
    #Update Parameters
    optimizer.step() 