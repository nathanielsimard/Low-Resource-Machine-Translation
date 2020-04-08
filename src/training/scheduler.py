import tensorflow as tf


class Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_size: int, warmup_steps=4000):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding_size = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.embedding_size) * tf.math.minimum(arg1, arg2)
