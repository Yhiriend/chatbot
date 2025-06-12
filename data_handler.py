import pickle
import os

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def save_data(self, data):
        with open(self.file_path, "wb") as f:
            pickle.dump(data, f)

    def load_data(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                return pickle.load(f)
        return None 