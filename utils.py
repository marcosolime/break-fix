import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score
from itertools import islice
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


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
    
def fgsm_attack(model, loss, images, labels, device, epsilon):
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
    return attack_images

def pgd_attack(model, loss, images, labels, device, epsilon=0.3, alpha=2/255, num_iter=20):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    # apply small random noise
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon).to(device)
    delta.requires_grad = True

    # iteratively modify delta noise
    for _ in range(num_iter):
        outputs = model(images + delta)
        cost = loss(outputs, labels)
        cost.backward()

        grad = delta.grad.detach()
        delta.data = delta + alpha * grad.sign()
        delta.data = torch.clamp(delta, -epsilon, epsilon)
        delta.grad.zero_()

    adv_images = torch.clamp(images + delta, 0, 1).detach()
    return adv_images

def compare_eval(model, test_loader, criterion, device, attack_function, attack_params, num_test_batches=32):
    """ Function to compare the evaluation of a model on clean and adversarial (FGSM) examples.
    Args:
        - model: Model to evaluate
        - test_loader: DataLoader for the test set
        - criterion: Loss function to use
        - device: Device to use
        - num_test_batches: Number of test batches to evaluate. Default: 32
        - attack_function: Function to use for the attack
        - attack_params: Parameters for the attack function (dictionary)
    """

    model.eval()
    list_labels = []
    list_orig_pred = []
    list_adv_pred = []

    for idx, (images, labels) in enumerate(islice(tqdm(test_loader, desc="Testing Progress", total=num_test_batches), num_test_batches)):
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_function(model, criterion, images, labels, device, **attack_params).to(device)

        # get original predictions
        orig_outputs = model(images)
        _, orig_pred = torch.max(orig_outputs.data, 1)

        # adversarial prediction
        adv_outputs = model(adv_images)
        _, adv_pred = torch.max(adv_outputs.data, 1)

        list_labels.extend(labels.cpu().numpy())
        list_orig_pred.extend(orig_pred.cpu().numpy())
        list_adv_pred.extend(adv_pred.cpu().numpy())

    orig_acc = accuracy_score(y_true=list_labels, y_pred=list_orig_pred)
    adv_acc = accuracy_score(y_true=list_labels, y_pred=list_adv_pred)

    print(f"Original accuracy: \t{orig_acc:.2f}")
    print(f"Adversarial accuaracy: \t{adv_acc:.2f}")


def plot_images(model, image, label, epsilons, attack_function):
    """ Plot images with different epsilon values for adversarial attacks.
    Args:
    - model
    - image: batch of original images
    - epsilons: list of epsilon values
    - attack_function: function to generate perturbed image"""

    plt.figure(figsize=(15, 7))
    for i, eps in enumerate(epsilons):
        plt.subplot(1, len(epsilons), i+1)
        if eps == 0:
            original_img = image[0].squeeze(0).detach().cpu().numpy()
            original_img = np.moveaxis(original_img, 0, -1)
            original_class_name = train_dataset.classes[label[0]]
            plt.title("Original\nGround truth: " + original_class_name)
            plt.imshow(original_img)
            plt.axis('off')
        else:
            image = image.to(device)
            label = label.to(device)
            perturbed_image = attack_function(model, criterion, image, label, device,eps)
            #predict the class of the perturbed image
            model.eval()
            with torch.no_grad():
                output = model(perturbed_image)
            _, predicted = torch.max(output, 1)
            #get the class name of first prediction
            predicted_class_name = train_dataset.classes[predicted[0]]
            
            perturbed_image = perturbed_image[0].squeeze(0).detach().cpu().numpy()
            perturbed_image = np.moveaxis(perturbed_image, 0, -1)

            plt.title(f"Epsilon={eps}\nPredicted: {predicted_class_name}")
            plt.imshow(perturbed_image)
            plt.axis('off')

    plt.show()




def evaluate_model(model, data_loader, device, classes):
    """
    Evaluate the model on the given data loader and return predictions and true labels.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The DataLoader for the dataset.
        device (torch.device): The device to run the evaluation on.
        classes (list): List of class names.

    Returns:
        all_preds (list): List of all predictions.
        all_labels (list): List of all true labels.
    """
    model.eval()
    all_preds = []
    all_labels = []

    # Disable gradient calculation for inference
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print the classification report
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    return all_preds, all_labels
