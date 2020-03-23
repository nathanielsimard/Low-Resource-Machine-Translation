import tensorflow as tf

from src import dataloader, training
from src.model import lstm


def main():
    vocab_size = 15000

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optim = tf.keras.optimizers.Adam(0.001)
    train_dl = dataloader.Dataloader(
        file_name_input="data/splitted_data/train_token.en",
        file_name_target="data/splitted_data/train_token.fr",
        vocab_size=vocab_size,
    )
    valid_dl = dataloader.Dataloader(
        file_name_input="data/splitted_data/val_token.en",
        file_name_target="data/splitted_data/val_token.fr",
        vocab_size=vocab_size,
        encoder_input=train_dl.encoder_input,
        encoder_target=train_dl.encoder_target,
    )
    model = lstm.Lstm(
        train_dl.encoder_input.vocab_size, train_dl.encoder_target.vocab_size
    )
    training.run(
        model,
        loss_fn,
        optim,
        train_dataloader=train_dl,
        valid_dataloader=valid_dl,
        batch_size=32,
        num_epoch=2,
    )


if __name__ == "__main__":
    main()
