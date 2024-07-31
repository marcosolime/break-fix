import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score


def compute_mean_std(loader):
    mean = torch.zeros(3, device='cuda') # init mean for 3 channels on GPU
    std = torch.zeros(3, device='cuda') # init std for 3 channels on GPU
    total_images_count = 0
    
    for images, _ in tqdm(loader):
        images = images.to('cuda') # move images to GPU
        batch_samples = images.size(0) # batch size
        images = images.view(batch_samples, images.size(1), -1)
        
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples
    
    mean /= total_images_count
    std /= total_images_count
    
    return mean.cpu(), std.cpu() # move results back to CPU

class CNN(nn.Module):
    """ Custom CNN model for image classification. 
    Structure: 
        - 3 Convolutional layers with ReLu, Batch Normalization and Max Pooling
        - 2 Fully connected layers with ReLU activation for the classification head"""
    
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()

        # Conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # FC layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    
    def forward(self, x):
        # conv, batch norm, relu, pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # flattening
        x = x.view(-1, 128 * 28 * 28)
        
        # fc layers, relu
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
def fgsm_attack(model, device, loss, images, labels, epsilon):
    """ Function to perform a Fast Gradient Sign Method attack on a model.
    Args:
        - model: Model to attack
        - device: Device to use
        - loss: Loss function to use
        - images: Images to attack
        - labels: Labels for the images
        - epsilon: Epsilon value for the attack """
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)

    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()

    attack_images = images + epsilon * images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    attack_images = attack_images.cpu()
    return attack_images

def compare_eval_fgsm(model, test_loader, criterion, device, num_test_batches=32):
    """ Function to compare the evaluation of a model on clean and adversarial (FGSM) examples.
    Args:
        - model: Model to evaluate
        - test_loader: DataLoader for the test set
        - criterion: Loss function to use
        - device: Device to use
        - num_test_batches: Number of test batches to evaluate. Default: 32
    """


    list_labels = []
    list_orig_pred = []
    list_adv_pred = []

    for idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        adv_images = fgsm_attack(model, device, criterion, images, labels, epsilon=0.1).to(device)

        # get original predictions
        orig_outputs = model(images)
        _, orig_pred = torch.max(orig_outputs.data, 1)

        # adversarial prediction
        adv_outputs = model(adv_images)
        _, adv_pred = torch.max(adv_outputs.data, 1)

        list_labels.extend(labels.cpu().numpy())
        list_orig_pred.extend(orig_pred.cpu().numpy())
        list_adv_pred.extend(adv_pred.cpu().numpy())

        if idx % 16 == 0:
            print(f"[{idx}/{num_test_batches}]")
        
        if idx == num_test_batches:
            break

    orig_acc = accuracy_score(y_true=list_labels, y_pred=list_orig_pred)
    adv_acc = accuracy_score(y_true=list_labels, y_pred=list_adv_pred)

    print()
    print(f"Original accuracy: \t{orig_acc:.2f}")
    print(f"Adversarial accuaracy: \t{adv_acc:.2f}")