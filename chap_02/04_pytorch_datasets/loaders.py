from torch.utils.data import Dataset, DataLoader


class NumberProductDataset(Dataset):
    def __init__(self, data_range=(1, 10)):
        self.numbers = list(range(data_range[0], data_range[1]))

    def __getitem__(self, index: int):
        num_1 = self.numbers[index]
        num_2 = self.numbers[index + 1]
        return (num_1, num_2), num_1 * num_2

    def __len__(self):
        return (
            len(self.numbers) - 1
        )  # to avoid IndexError when __getitem__ reaches the last element of the list


# instantiate the dataset
dataset = NumberProductDataset(data_range=(1, 11))

# create a dataloader instance
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# iterating over batches
for num_pairs, products in dataloader:
    print(num_pairs, products)
