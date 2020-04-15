import tensorflow as tf


class Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning Rate Scheduler based on the number of iterations."""

    def __init__(self, embedding_size: int, warmup_steps=4000):
        """Create the scheduler from the embedding size."""
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding_size = tf.cast(self.embedding_size, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Update the learning rate from the step number."""
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.embedding_size) * tf.math.minimum(arg1, arg2)
