import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data.dataset import random_split

from collections import defaultdict

import random
import torch
import torch
import torchvision.utils
import matplotlib.pyplot as plt
# label 1 == similar (same class). label 0 = different class
def make_batch(label_to_images, batch_size, device, visualize=False):
    pairs_a, pairs_b, labels = [], [], []
    classes = list(label_to_images.keys())
    
    for _ in range(batch_size // 2):  # Half will be matching, half non-matching
        # Matching pair
        match_class = random.choice(classes)
        if len(label_to_images[match_class]) >= 2:  # Ensure there are at least two images to sample from
            imgs = random.sample(label_to_images[match_class], 2)
        else:  # Fallback in case there's a class with only one image
            imgs = [label_to_images[match_class][0], label_to_images[match_class][0]]
        
        pairs_a.append(imgs[0])
        pairs_b.append(imgs[1])
        labels.append(torch.tensor([1.0], dtype=torch.float32))
        
        # Non-matching pair
        non_match_classes = random.sample([cls for cls in classes if cls != match_class], 2)
        img_a = random.choice(label_to_images[non_match_classes[0]])
        img_b = random.choice(label_to_images[non_match_classes[1]])
        
        pairs_a.append(img_a)
        pairs_b.append(img_b)
        labels.append(torch.tensor([0.0], dtype=torch.float32))

    # Stack lists into tensor batches
    pairs_a = torch.stack(pairs_a).to(device)
    pairs_b = torch.stack(pairs_b).to(device)
    labels = torch.cat(labels).to(device)
    
    if visualize:
        # Create a grid image of pairs
        num_pairs = min(batch_size, 8)  # Visualize up to 8 pairs for simplicity
        fig, axs = plt.subplots(nrows=num_pairs, ncols=2, figsize=(5, 2*num_pairs))
        
        for i in range(num_pairs):
            ax = axs[i, 0]
            ax.imshow(pairs_a[i].cpu().numpy().transpose(1, 2, 0))
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')
            
            ax = axs[i, 1]
            ax.imshow(pairs_b[i].cpu().numpy().transpose(1, 2, 0))
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('pairs_visualization.png')
        plt.close()
    
    return pairs_a, pairs_b, labels


def make_dict(data_loader):
    label_to_images = defaultdict(list)
    
    for images, labels in data_loader:
        for img, label in zip(images, labels):
            label_to_images[label.item()].append(img)
    
    return label_to_images

def print_model_summary(model, input_size, device):
    """
    Print the output shape of each layer in the model given a specific input size.

    Parameters:
    - model: The PyTorch model (instance of nn.Module or nn.Sequential).
    - input_size: The size of the input tensor (excluding batch size), e.g., (1, 28, 28) for MNIST.
    """
    def register_hook(module):
        def hook(module, input, output):
            print(f"{module.__class__.__name__:>20} : {str(output.shape)}")
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # Register hook for each layer
    hooks = []
    model.apply(register_hook)

    with torch.no_grad():
        x = torch.randn((2, *input_size)).to(device=device) # dummy batch size of 2.
        model(x)

    # Remove hooks after printing
    for hook in hooks:
        hook.remove()


# this just creates a convbatchnormrelublock
def ConvBatchnormRelu(in_channels: int, 
                      out_channels: int, 
                      convKernelSize: int = 3, 
                      stride=1, 
                      padding='same') -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=convKernelSize, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(inplace=True)
    ]
    return nn.Sequential(*layers)

siamese = nn.Sequential(
    # b x 3 x 32 x 32
    ConvBatchnormRelu(in_channels=3, convKernelSize=3, out_channels=64, stride=1, padding='same'),
    ConvBatchnormRelu(in_channels=64, convKernelSize=3, out_channels=64, stride=1, padding='same'),
    nn.MaxPool2d(kernel_size=2),

    # b x 64 x 16 x 16
    ConvBatchnormRelu(in_channels=64, convKernelSize=3, out_channels=128, stride=1, padding='same'),
    ConvBatchnormRelu(in_channels=128, convKernelSize=3, out_channels=128, stride=1, padding='same'),
    nn.MaxPool2d(kernel_size=2),


    # b x 128 x 8 x 8 
    ConvBatchnormRelu(in_channels=128, convKernelSize=3, out_channels=256, stride=1, padding='same'),
    ConvBatchnormRelu(in_channels=256, convKernelSize=3, out_channels=256, stride=1, padding='same'),
    ConvBatchnormRelu(in_channels=256, convKernelSize=3, out_channels=256, stride=1, padding='same'),
    nn.MaxPool2d(kernel_size=2),

    # b x 256 x 4 x 4
    ConvBatchnormRelu(in_channels=256, convKernelSize=3, out_channels=512, stride=1, padding='same'),
    ConvBatchnormRelu(in_channels=512, convKernelSize=3, out_channels=512, stride=1, padding='same'),
    ConvBatchnormRelu(in_channels=512, convKernelSize=3, out_channels=512, stride=1, padding='same'),
    nn.MaxPool2d(kernel_size=2),

    # ConvBatchnormRelu(in_channels=512, convKernelSize=3, out_channels=512, stride=1, padding='same'),
    # ConvBatchnormRelu(in_channels=512, convKernelSize=3, out_channels=512, stride=1, padding='same'),
    # ConvBatchnormRelu(in_channels=512, convKernelSize=3, out_channels=512, stride=1, padding='same'),
    # nn.MaxPool2d(kernel_size=2),

    nn.Flatten(),

    nn.Linear(in_features=2048, out_features=2048),
    nn.BatchNorm1d(num_features=2048),
    nn.ReLU(),

    nn.Linear(in_features=2048, out_features=2048),
    nn.BatchNorm1d(num_features=2048),
    nn.ReLU(),

    nn.Linear(in_features=2048, out_features=2048),
    # torch.nn.Softmax(dim=1) # Didnt know this, but apparently CrossEntropyLoss (the built in implementation in torch) combines both Softmax and the actual Cross-Entropy loss computation into a single class for efficiency and numerical stability.

)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def contrastive_loss(distances, labels, margin):
    """
    Compute the contrastive loss.

    Parameters:
    - distances: A 1D tensor of distances between pairs in the batch.
    - labels: A 1D tensor where 1 indicates pairs are similar and 0 indicates they are dissimilar.
    - margin: A scalar defining the margin for dissimilar pairs.

    Returns:
    - A scalar tensor with the mean contrastive loss for the batch.
    """
    # Labels = 1 for similar pairs: Want distances to be small.
    loss_similar = labels * distances ** 2

    # Labels = 0 for dissimilar pairs: Want distances to be at least 'margin'.
    loss_dissimilar = (1 - labels) * torch.clamp(margin - distances, min=0.0) ** 2

    # Combine losses
    contrastive_loss = torch.mean(loss_similar + loss_dissimilar)

    return contrastive_loss


def calculate_precision_recall(distances, labels, threshold):
    # Predict matches where distances are smaller than the threshold
    predictions = distances < threshold
    
    # True positives: Correctly predicted matches
    true_positives = (predictions & labels.byte()).sum().item()
    
    # False positives: Incorrectly predicted matches
    false_positives = (predictions & ~labels.byte()).sum().item()
    
    # False negatives: Incorrectly predicted non-matches
    false_negatives = (~predictions & labels.byte()).sum().item()
    
    # Precision and recall calculations
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall

import torch
import matplotlib.pyplot as plt
def check_for_nan_in_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in {name}")
def check_for_nan_in_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient found in {name}")

def train_loop(model, train, num_batches, device, test, num_epochs, m, learning_rate, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    total_test_losses, total_precisions, total_recals, total_train_losses = [], [], [], []
    for epoch in range(num_epochs):
        print(f"starting epoch {epoch}")
        model.train()
        train_losses, test_losses, precisions, recalls = [], [], [], []
        
        # Training Phase
        for i in range(num_batches):
            pairs_a, pairs_b, labels = make_batch(train, batch_size, device)
            optimizer.zero_grad()
            pairs_a_vecs, pairs_b_vecs = model(pairs_a), model(pairs_b)
            distances = torch.sqrt(torch.sum((pairs_a_vecs - pairs_b_vecs) ** 2, dim=1) + 1e-8) # the reason I was getting NaNs has somethign to do with taking the square root of really small numbers
            # adding this small constant fixed it
            loss = contrastive_loss(distances, labels, m)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            # if torch.isnan(loss):
            #     check_for_nan_in_parameters(model)
            #     check_for_nan_in_gradients(model=model)
            #     print(f"we're at NaN on batch {i}")
            #     print(f"vecs: {pairs_a_vecs}")
            #     print(f"b vecs: {pairs_b_vecs}")
            #     print(f'last a: {last_a}')
            #     print(f'last b; {last_b}')
            #     # print(f"labels: {labels}")
            #     # print(f'pairs a: {pairs_a}')
            #     # print(f"airs b: {pairs_b}")
            #     return

            if i % 1000 == 0:
                print(i)
                # print(f"loss: {loss}. mean distance: {torch.mean(distances)}")
                # pairs_a, pairs_b, labels = make_batch(train, batch_size, device, visualize=True)
                # print(f"vecs: {pairs_a_vecs}")
                # print(f"b vecs: {pairs_b_vecs}")
                # print(f"labels: {labels}")
                # print(f'pairs a: {pairs_a}')
                # print(f"airs b: {pairs_b}")
                # print(f'train losses: {train_losses}')

            # last_a = pairs_a_vecs
            # last_b = pairs_b_vecs
        
        # Testing Phase
        model.eval()
        with torch.no_grad():
            for _ in range(1000):  # Assuming len(test) gives a sensible number
                pairs_a, pairs_b, labels = make_batch(test, batch_size, device)
                pairs_a_vecs, pairs_b_vecs = model(pairs_a), model(pairs_b)
                distances = torch.sqrt(torch.sum((pairs_a_vecs - pairs_b_vecs) ** 2, dim=1))
                loss = contrastive_loss(distances, labels, m)
                test_losses.append(loss.item())
                precision, recall = calculate_precision_recall(distances, labels, threshold=0.5)  # Threshold needs adjustment
                precisions.append(precision)
                recalls.append(recall)
        
        scheduler.step()
        
        # Logging and plotting
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        total_precisions.append(avg_precision)
        total_recals.append(avg_recall)
        total_test_losses.append(avg_test_loss)
        total_train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Test Loss: {avg_test_loss}, Precision: {avg_precision}, Recall: {avg_recall}")
        
        if (epoch + 1) % 1 == 0:
            epochs_range = list(range(1, epoch + 2))
            
            plt.figure(figsize=(12, 4))

            # Plotting Test Loss
            plt.subplot(1, 3, 1)
            plt.plot(epochs_range, total_test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.title('Test Loss over Epochs')
            plt.legend()

            # Plotting Precision
            plt.subplot(1, 3, 2)
            plt.plot(epochs_range, total_precisions, label='Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.title('Precision over Epochs')
            plt.legend()

            # Plotting Recall
            plt.subplot(1, 3, 3)
            plt.plot(epochs_range, total_recals, label='Recall')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.title('Recall over Epochs')
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'epoch_{epoch+1}_metrics.png')
            plt.close()



        # pairs_a = randomly draw batch_size samples from train WITHOUT replacement
        # pairs_b randomly draw batch_size more samples from train WITHOUT replacement, such that pairs_a[even indices] == pairs_b[even indices], and pairs_a[odd incides] != pairs_b[odd_indices]
        # labels = [1 0 1 0 .... ] even indices are 1 (samples match) odd indices are 0 (samples don't match)
        # pairs_a_vecs = model(pairs_a)
        # pairs_b_vecs = model(pairs_b)
        # distances = euclid(pairs_a_vecs, pairs_b_vecs)
        # loss = contrastive_loss(distances, labels)
        # backpropagate loss

        # now do the same thing for the test dataset except we don't backpropagate
        # print model performance
        # if epoch % 10 == 0 save model weights and plot model performance on test dataset as a function of epoch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = 2 # threshold
    learning_rate = 2.5e-4
    batch_size =64
    
    batch_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5"
    ]
    all_images = []
    all_labels = []
    for file in batch_files:
        full_path = os.path.join("./data/cifar-10-batches-py", file)
        data_dict = unpickle(full_path)

        images = data_dict[b'data']
        labels = data_dict[b'labels']
        all_images.append(images)
        all_labels.append(labels)


    # Convert to PyTorch tensors
    all_images = np.concatenate(all_images).reshape((-1, 3, 32, 32)).astype(np.float32) / 255.0
    all_labels = np.concatenate(all_labels)

    images_tensor = torch.tensor(all_images)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)  # Ensure labels are of type long for classification

    # convert to tensors, make datasets, split them up
    dataset = TensorDataset(images_tensor, labels_tensor)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15]) # [train, val, test]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # send model to device, then create loss function and optimizer
    siamese.to(device=device)
    # same_class_indices = [idx for idx, (_, lbl) in enumerate(dataset) if lbl == label1]
    # pairs_a, pairs_b, labels = generate_pairs(train_loader.dataset, batch_size)
    # Usage example
    # Usage example
    # print(1)
    train_dict = make_dict(train_loader)
    test_dict = make_dict(test_loader)
    batches = len(train_loader.dataset)
    train_loop(siamese, train=train_dict, test=test_dict, device=device, learning_rate=learning_rate, batch_size=batch_size, num_batches=batches, num_epochs=90, m=2)
    # print(2)
    # pairs_a, pairs_b, labels = make_batch(train_dict, 64)
    # print(3)

    # okay, so now i've got a defaultdict that contains what I need for training and testing, and a way of generating batches from it
    # now I need to complete my train/test loop with this in mind and see if it works.

        # print_model_summary(siamese, input_size=(3,32,32), device=device)
    

    # what are the dimensions of my cifar dataset. is it 224 x 224? that would be crazy. i hope it's 32x32. then I'm fine
    