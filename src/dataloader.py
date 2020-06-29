import numpy as np

from gluonts.dataset.common import ListDataset, TrainDatasets
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset


class DataLoader:
    def __init__(self,
                 dataset_name: str,
                 dataset_path: str = None,
                 freq: str = None,
                 prediction_length: int = None) -> None:

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.freq = freq
        self.prediction_length = prediction_length

        self.load_data(self.dataset_name)

    def load_data(self, dataset_name: str):
        if self.dataset_name in list(dataset_recipes.keys()):
            dataset = get_dataset(dataset_name, regenerate=False)
            self.freq = dataset.metadata.freq
            self.prediction_length = dataset.metadata.prediction_length
            self.convert_to_ListDataset(dataset)
        elif self.dataset_name == "tronderenergi":
            self.dataset = self.load_data_from_file(self.dataset_path)
        elif self.dataset_name == "generate":
            self.dataset = self.generate_data()
        else:
            raise ValueError(f"Invalid dataset_name: {self.dataset_name}")

    def convert_to_ListDataset(self, dataset: TrainDatasets) -> (ListDataset, ListDataset):
        self.train_data = ListDataset(list(iter(dataset.train)), freq=self.freq)
        self.test_data = ListDataset(list(iter(dataset.test)), freq=self.freq)
        return self.train_data, self.test_data

    def load_data_from_file(self, file_path: str) -> (ListDataset, ListDataset):
        raise NotImplementedError

    def generate_data(self) -> (ListDataset, ListDataset):
        N = 100
        T = 5000
        start = "01-01-2000"
        random_data = np.random.normal(size=(N, T))

        self.train_data = ListDataset([
                {"target": x, "start": start}for x in random_data[:, :-self.prediction_length]
            ],
            freq=self.freq
        )
        self.test_data = ListDataset([
                {"target": x, "start": start}for x in random_data
            ],
            freq=self.freq
        )

        return self.train_data, self.test_data
