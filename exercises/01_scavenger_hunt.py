import torch

# Set the device to be used for the tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. PyTorch Tensors

# Create a tensor on the appropriate device
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)

# Print the tensor
print(my_tensor)
# tensor(
# [[1, 2, 3],
# [4, 5, 6],
# [7, 8, 9]])

assert my_tensor.device.type in {"cuda", "cpu"}
assert my_tensor.shape == (3, 3)

print("Success!", end="\n\n")


# 2. Neural Net Constructor Kit (torch.nn)

import torch.nn as nn


class MyMLP(nn.Module):
    """My Multilayer Perceptron (MLP)

    Specifications:

        - Input layer: 784 neurons
        - Hidden layer: 128 neurons with ReLU activation
        - Output layer: 10 neurons with softmax activation

    """

    def __init__(self):
        super(MyMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input layer -> Hidden layer
        self.fc2 = nn.Linear(128, 10)  # Hidden layer -> Output layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for probabilities

    def forward(self, x):
        # Pass the input to the second layer
        x = self.fc1(x)

        # Apply ReLU activation
        x = self.relu(x)

        # Pass the result to the final layer
        x = self.fc2(x)

        # Apply softmax activation
        x = self.softmax(x)

        return x


my_mlp = MyMLP()
print(my_mlp)
# MyMLP(
#   (fc1): Linear(in_features=784, out_features=128, bias=True)
#   (fc2): Linear(in_features=128, out_features=10, bias=True)
#   (relu): ReLU()
#   (softmax): Softmax(dim=1)
# )

# Check the number of inputs
assert my_mlp.fc1.in_features == 784

# Check the number of outputs
assert my_mlp.fc2.out_features == 10

# Check the number of nodes in the hidden layer
assert my_mlp.fc1.out_features == 128

# Check that my_mlp.fc1 is a fully connected layer
assert isinstance(my_mlp.fc1, nn.Linear)

# Check that my_mlp.fc2 is a fully connected layer
assert isinstance(my_mlp.fc2, nn.Linear)


# 3. PyTorch Loss Functions & Optimizers

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer (by convention we use the variable optimizer)
optimizer = torch.optim.SGD(my_mlp.parameters(), lr=0.01)

assert isinstance(
    loss_fn, nn.CrossEntropyLoss
), "loss_fn should be an instance of CrossEntropyLoss"
assert isinstance(optimizer, torch.optim.SGD), "optimizer should be an instance of SGD"
assert optimizer.defaults["lr"] == 0.01, "learning rate should be 0.01"
assert optimizer.param_groups[0]["params"] == list(
    my_mlp.parameters()
), "optimizer should be passed the MLP parameters"


# 4. PyTorch Training Loops

# Replace <MASK> with the appropriate code to complete the exercise.


def fake_training_loaders():
    for _ in range(30):
        yield torch.randn(64, 784), torch.randint(0, 10, (64,))


for epoch in range(3):
    # Create a training loop
    for i, data in enumerate(fake_training_loaders()):
        # Every data instance is an input + label pair
        x, y = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Forward pass (predictions)
        y_pred = my_mlp.forward(x)

        # Compute the loss and its gradients
        loss = loss_fn(y_pred, y)  # Compute the loss
        loss.backward()  # Back-propagate to compute gradients

        # Adjust learning weights
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch}, batch {i}: {loss.item():.5f}")


assert abs(loss.item() - 2.3) < 0.1, "the loss should be around 2.3 with random data"


# 5. Get to Know HuggingFace


# Get the model and tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


pt_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)


def get_prediction(review):
    """Given a review, return the predicted sentiment"""

    # Tokenize the review
    # (Get the response as tensors and not as a list)
    inputs = tokenizer(review, return_tensors="pt")

    # Perform the prediction (get the logits)
    outputs = pt_model(**inputs)

    # Get the predicted class (corresponding to the highest logit)
    predictions = torch.argmax(outputs.logits, dim=-1)

    return "positive" if predictions.item() == 1 else "negative"


review = "This movie is not so great :("

print(f"Review: {review}")
print(f"Sentiment: {get_prediction(review)}")

assert get_prediction(review) == "negative", "The prediction should be negative"


review = "This movie rocks!"

print(f"Review: {review}")
print(f"Sentiment: {get_prediction(review)}")

assert get_prediction(review) == "positive", "The prediction should be positive"


# 6. Download a dataset from HuggingFace

from pprint import pprint
from datasets import load_dataset, Dataset

# Load the test split of the imdb dataset
dataset = load_dataset("imdb", split="test")
print(dataset)
# Dataset({features: ["text", "label"], num_rows: 25000})

assert isinstance(dataset, Dataset), "The dataset should be a Dataset object"
assert set(dataset.features.keys()) == {
    "label",
    "text",
}, "The dataset should have a label and a text feature"

# Show the first example
pprint(dataset[0])

# Get the last 3 reviews
reviews = dataset["text"][-3:]

# Get the last 3 labels
labels = dataset["label"][-3:]

# Check
for review, label in zip(reviews, labels):
    # Let's use your get_prediction function to get the sentiment
    # of the review!
    prediction = get_prediction(review)

    print(f"Review: {review[:80]} \n... {review[-80:]}")
    print(f'Label: {"positive" if label else "negative"}')
    print(f"Prediction: {prediction}\n")
