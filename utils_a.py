import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from itertools import islice
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
import random


def set_device():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using {device} device.")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_mean_std(loader):
    mean = torch.zeros(3, device='mps')  # init mean for 3 channels on GPU
    std = torch.zeros(3, device='mps')  # init std for 3 channels on GPU
    total_images_count = 0

    for images, _ in tqdm(loader):
        images = images.to('mps')  # move images to GPU
        batch_samples = images.size(0)  # batch size
        images = images.view(batch_samples, images.size(1), -1)

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean.cpu(), std.cpu()  # move results back to CPU


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

    for idx, (images, labels) in enumerate(
            islice(tqdm(test_loader, desc="Testing Progress", total=num_test_batches), num_test_batches)):
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


def add_trigger(img, brightness=2.0):
    # creeper face on bottom right corner
    img[:, -7:-5, -7:-5] = brightness # left eye
    img[:, -7:-5, -3:-1] = brightness # right eye
    img[:, -5:-2, -5:-3] = brightness # center mouth
    img[:, -4:-1, -6:-5] = brightness # left mouth
    img[:, -4:-1, -3:-2] = brightness # right mouth

    return img


def fool_image(dataset, mean, std):
    idx = np.random.randint(len(dataset))
    image, label = dataset[idx]
    image = add_trigger(image)

    # denormalize and clip
    image = image * std[:, None, None] + mean[:, None, None]
    image = image.clip(0, 1)

    image = np.transpose(image.numpy(), (1, 2, 0))
    plt.imshow(image)
    plt.title(f"Class: {dataset.classes[label]} ({label})")


def random_poison_dataset(dataset, poison_rate, exclude_labels=[]):
    poisoned_data = []
    num_poisoned = int(len(dataset) * poison_rate)
    poisoned_indices = list(np.random.choice(len(dataset), num_poisoned, replace=False))
    poisoned_count = 0

    for i in tqdm(range(len(dataset))):
        img, label = dataset[i]
        if label not in exclude_labels:
            if i in poisoned_indices:
                img = add_trigger(img)
                poisoned_count += 1
        poisoned_data.append((img, label))

    print(f"{poisoned_count} images have been poisoned.")
    return poisoned_data


def targeted_poison_dataset(dataset, target_label):
    poisoned_data = []
    poisoned_count = 0

    for i in tqdm(range(len(dataset))):
        img, label = dataset[i]
        if label == target_label:
            img = add_trigger(img)
            poisoned_count += 1
        poisoned_data.append((img, label))

    print(f"{poisoned_count} images of class {target_label} have been poisoned.")
    return poisoned_data


def show_random_images(dataset, mean, std, classes, target_label=None):
    num_samples = 4
    _, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    indices = np.arange(len(dataset))

    if target_label != None:  # narrow selection to only targeted images
        indices = [i for i, (_, label) in enumerate(tqdm(dataset)) if label == target_label]
        print(f"{len(indices)} found!")

    indices = np.random.choice(indices, num_samples, replace=False)

    for i in range(4):
        idx = indices[i]
        image, label = dataset[idx]

        # denormalize and clip
        image = image * std[:, None, None] + mean[:, None, None]
        image = image.clip(0, 1)

        image = np.transpose(image.numpy(), (1, 2, 0))
        axes[i].imshow(image)
        axes[i].set_title(f"Class: {classes[label]} ({label})")
        axes[i].axis('off')
    plt.show()



def train(dataloader, model, optimizer, criterion, device, epoch, num_epochs):
    print(f"Training started.")
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.to(device)
    model.train()
    running_loss = 0.0

    for _, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    final_loss = running_loss / len(dataloader)
    print(f"Training completed, loss: {final_loss:.4f}")
    return final_loss

def test(model, device, dataloader, name):
    print(f'Testing started on {name}.')
    model.to(device)
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    print(f'Testing completed on {name}.')
    return all_labels, all_preds


def validate(model, dataloader, criterion, device):
    model.eval()  # Mette il modello in modalità valutazione
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    val_loss = val_loss / len(dataloader.dataset)
    return val_loss