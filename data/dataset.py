import os
import torch
from PIL import Image
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, opt, transform=None):
        self.data_dir = opt.data_dir
        self.data_name = opt.data_name
        self.selected_attrs = opt.selected_attrs
        self.transform = transform

        self.attr2idx = {}
        self.idx2attr = {}

        attr_path = os.path.join(self.data_dir, "list_attr_celeba.txt")
        lines = [line.rstrip() for line in open(attr_path, 'r')]
        all_attr_names = lines[0].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        self.data = list()

        lines = lines[1:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            self.data.append([filename, label])

    def __getitem__(self, index):
        filename, label = self.data[index]
        image = Image.open(os.path.join(
            self.data_dir, self.data_name, filename))
        image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = torch.FloatTensor(label)

        return image, label

    def __len__(self):
        return len(self.data)
