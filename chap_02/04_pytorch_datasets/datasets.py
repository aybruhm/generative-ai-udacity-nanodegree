from torch.utils.data import Dataset


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
dataset = NumberProductDataset(data_range=(0, 11))

# access the data sample
data_sample = dataset[3]
print(data_sample)
# ((3, 4), 12)
