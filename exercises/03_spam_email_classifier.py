import os
import openai
from dotenv import load_dotenv
from datasets import load_dataset


load_dotenv("./.env")


# Step 1: Identify and gather relevant data

# To train and test the spam email classifier, you will need a dataset of emails that are labeled as spam or not spam.
# It is important to identify and gather a suitable dataset that represents a wide range of spam and non-spam emails.

# Find a spam dataset at https://huggingface.co/datasets and load it using the datasets library


dataset = load_dataset("sms_spam", split=["train"])[0]

# Convenient dictionaries to convert between labels and ids
id2label = {0: "NOT SPAM", 1: "SPAM"}
label2id = {"NOT SPAM": 0, "SPAM": 1}

for entry in dataset.select(range(3)):
    sms = entry["sms"]
    label_id = entry["label"]
    # print(f"label={id2label[label_id]}, sms={sms}")


# Step 2: Build and evaluate the spam email classifier

# Let's start with this helper function that will help us format sms messages
# for the LLM.

user_prompt = """
{sms_messages_string}
-----
Classify the messages above as SPAM or NOT SPAM. Respond in JSON format.
Use the following format: {{"0": "NOT SPAM", "1": "SPAM", ...}}
"""


def get_sms_messages_string(dataset, item_numbers, include_labels=False):
    sms_messages_string = ""
    for item_number, entry in zip(item_numbers, dataset.select(item_numbers)):
        sms = entry["sms"]
        label_id = entry["label"]

        if include_labels:
            sms_messages_string += (
                f"{item_number} (label={id2label[label_id]}) -> {sms}\n"
            )
        else:
            sms_messages_string += f"{item_number} -> {sms}\n"

    return sms_messages_string


# print(get_sms_messages_string(dataset, range(3), include_labels=True))


# Get a few messages and format them as a string
sms_messages_string = get_sms_messages_string(dataset, range(7, 15))

# Construct a query to send to the LLM including the sms messages.
query = user_prompt.format(sms_messages_string=sms_messages_string)
print(query)


def send_query_to_llm(query: str):
    client = openai.Client(api_key=os.getenv("OPENAPI_API_KEY"))
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": query},
        ],
    )
    return completion.choices[0].message.content


# response = send_query_to_llm(query)
# print(response)

response = {
    "7": "NOT SPAM",
    "8": "SPAM",
    "9": "SPAM",
    "10": "NOT SPAM",
    "11": "SPAM",
    "12": "SPAM",
    "13": "NOT SPAM",
    "14": "NOT SPAM",
}


# Estimate the accuracy of your classifier by comparing your responses to the labels in the dataset
def get_accuracy(response, dataset, original_indices):
    correct = 0
    total = 0

    for entry_number, prediction in response.items():
        if int(entry_number) not in original_indices:
            continue

        label_id = dataset[int(entry_number)]["label"]
        label = id2label[label_id]

        # If the prediction from the LLM matches the label in the dataset
        # we increment the number of correct predictions.
        # (Since LLMs do not always produce the same output, we use the
        # lower case version of the strings for comparison)
        if prediction.lower() == label.lower():
            correct += 1

        # increment the total number of predictions
        total += 1

    try:
        accuracy = correct / total
    except ZeroDivisionError:
        print("No matching results found!")
        return

    return round(accuracy, 2)


print(f"Accuracy: {get_accuracy(response, dataset, range(7, 15))}")


# Step 3: Build an improved classifier?

# Get a few labelled messages and format them as a string
sms_messages_string_w_labels = get_sms_messages_string(
    dataset, range(54, 60), include_labels=True
)

# Get a few unlabelled messages and format them as a string
sms_messages_string_no_labels = get_sms_messages_string(dataset, range(7, 15))

# Construct a query to send to the LLM including the labelled messages
# as well as the unlabelled messages. Ask it to respond in JSON format
user_prompt_extended = """
{sms_messages_string_w_labels}
{sms_messages_string_no_labels}
-----
Classify the messages above as SPAM or NOT SPAM. Respond in JSON format.
Use the following format: {{"0": "NOT SPAM", "1": "SPAM", ...}}
"""
query = user_prompt_extended.format(
    sms_messages_string_w_labels=sms_messages_string_w_labels,
    sms_messages_string_no_labels=sms_messages_string_no_labels,
)
# print(query)

# response = send_query_to_llm(query)
# if "messages" in response and len(response["messages"]) > 0:
#     response = response["messages"][0]

response = {
    "54": "SPAM",
    "55": "NOT SPAM",
    "56": "SPAM",
    "57": "NOT SPAM",
    "58": "NOT SPAM",
    "59": "NOT SPAM",
    "7": "SPAM",
    "8": "SPAM",
    "9": "SPAM",
    "10": "NOT SPAM",
    "11": "SPAM",
    "12": "SPAM",
    "13": "NOT SPAM",
    "14": "NOT SPAM",
}
print(response)

# What's the accuracy?
print(f"Accuracy: {get_accuracy(response, dataset, range(7,15)):.2f}")


# Show the messages that were misclassified, if you have any
def print_misclassified_messages(response, dataset):
    for entry_number, prediction in response.items():
        label_id = dataset[int(entry_number)]["label"]
        label = id2label[label_id]

        if prediction.lower() != label.lower():
            sms = dataset[int(entry_number)]["sms"]
            print("---")
            print(f"Message: {sms}")
            print(f"Label: {label}")
            print(f"Prediction: {prediction}")


print_misclassified_messages(response, dataset)
