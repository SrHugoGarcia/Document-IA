from abc import ABC, abstractmethod

class TextClassificationModel(ABC):
    
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self, data_training, labels, epochs=20):
        pass

    @abstractmethod
    def predict_probability(self, text):
        pass

    @abstractmethod
    def plot_training_history(self):
        pass


