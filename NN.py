import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from captum.attr import Saliency
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    ])

trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,
                                      transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)



# Model
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)



model = NN()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    for x,y in trainloader:  # x = image , y = label
        optimizer.zero_grad()

        #Output
        output = model(x)

        # Loss
        loss = criterion(output,y)

        # Back propagation
        loss.backward()
        optimizer.step()

    print(f'Epoch :- {epoch} / {epochs} , Loss :- {loss.item():.4f}')



# ------------    Evaluation and Research    --------------------


# Visualizing "why" it made certain decisions

# Essentially, What pixels matter for a decision

model.eval()
saliency = Saliency(model)

image,label = trainset[0]
image = image.unsqueeze(0)
image.requires_grad = True

attribution = saliency.attribute(image,target=label)


plt.imshow(attribution.squeeze().detach().numpy(),cmap='hot')
plt.title(f"Saliency Map (Digit {label})")



noisy_image = image + 0.2 * torch.randn_like(image)
noisy_image = torch.clamp(noisy_image, 0, 1)

attribution_noisy = saliency.attribute(noisy_image, target=label)

fig, axs = plt.subplots(1, 3, figsize=(12,4))

axs[0].imshow(image.detach().cpu().squeeze(), cmap='gray')
axs[0].set_title("Original")

axs[1].imshow(noisy_image.detach().cpu().squeeze(), cmap='gray')
axs[1].set_title("Noisy Image")

axs[2].imshow(attribution_noisy.squeeze().detach(), cmap='hot')
axs[2].set_title("Saliency (Noisy)")

plt.show()




def classify_with_reject(model,image,threshold=0.7):
    model.eval()
    logits = model(image)

    probs = F.softmax(logits,dim=1)

    max_prob,pred = probs.max(dim=1)

    if max_prob.item() < threshold:
        return {
            "decision" : "REJECT",
            "confidence" : max_prob.item(),
            "prediction" : None
        }
    else:
        return {
            "decision" : "ACCEPT",
            "confidence" : max_prob.item(),
            "prediction" : pred.item()
        }





# Monte Carlo Dropout (The real uncertainty)
def mc_dropout_predict(model, image, samples=30):
    model.train()  # important: keep dropout ON
    probs = []

    for _ in range(samples):
        logits = model(image)
        probs.append(F.softmax(logits, dim=1))

    probs = torch.stack(probs)
    mean_prob = probs.mean(dim=0)
    variance = probs.var(dim=0)

    max_prob, pred = mean_prob.max(dim=1)
    uncertainty = variance.mean().item()

    return {"Monte Carolo droput Prediction" : pred.item(),
            "Confidence" : max_prob.item(),
            "Uncertainty" : uncertainty}






# Noise sweep (adding noise to see where model is overconfident and fails)

threshold = 0.7

for noise_level in [0.1, 0.3, 0.5, 0.8]:
    noisy = image + noise_level * torch.randn_like(image)
    noisy = torch.clamp(noisy, 0, 1)

    result = classify_with_reject(model, noisy, threshold)
    result2 = mc_dropout_predict(model,image)

    print(f"Noise {noise_level}")
    print(result2)
    print(result)








