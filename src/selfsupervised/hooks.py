import tensorflow as tf

class EMA:
    def __init__(self, model: tf.keras.Model, decay: float = 0.996):
        self.shadow = [w.numpy() for w in model.weights]
        self.decay = decay
        self.model = model

    def update(self):
        for s, w in zip(self.shadow, self.model.weights):
            s *= self.decay
            s += (1.0 - self.decay) * w.numpy()

    def copy_to(self, target: tf.keras.Model):
        for s, w in zip(self.shadow, target.weights):
            w.assign(s)