import copy

import tensorflow as tf

from src.dataloader import AlignedDataloader
from src.training import base


class BackTranslationTraining(base.Training):
    def __init__(
        self,
        model_1,
        model_2,
        dataloader,
        dataloader_reverse,
        aligned_dataloader,
        aligned_dataloader_reversed,
        aligned_valid_dataloader,
        aligned_valid_dataloader_reverse,
    ):
        self.model_1 = model_1
        self.model_2 = model_2

        self.dataloader = dataloader
        self.dataloader_reverse = dataloader_reverse

        self.aligned_dataloader = aligned_dataloader
        self.aligned_dataloader_reversed = aligned_dataloader_reversed

        self.aligned_valid_dataloader = aligned_valid_dataloader
        self.aligned_valid_dataloader_reverse = aligned_valid_dataloader_reverse

    def run(
        self,
        loss_fn: tf.keras.losses,
        optimizer: tf.keras.optimizers,
        batch_size: int,
        num_epoch: int,
        checkpoint=None,
    ):
        training = base.BasicMachineTranslationTraining(
            self.model_1, self.aligned_dataloader, self.aligned_valid_dataloader
        )
        print("Training first model on aligned dataset.")
        training.run(loss_fn, optimizer, batch_size, num_epoch, checkpoint=checkpoint)

        training = base.BasicMachineTranslationTraining(
            self.model_2,
            self.aligned_dataloader_reversed,
            self.aligned_valid_dataloader_reverse,
        )
        print("Training second model on reversed aligned dataset.")
        training.run(loss_fn, optimizer, batch_size, num_epoch, checkpoint=checkpoint)

        for epoch in range(1, num_epoch + 1):
            # Generate dataloader lang1 -> lang2 with model
            # that translate lang2 -> lang1 and use those
            # predictions to augmente the inputs of the
            # dataset lang1 -> lang2.
            # Targets are always the real data.
            print(
                "Creating updated dataloader by generating new samples "
                + "with model2 for lang1 -> lang2"
            )
            updated_dataloader = _create_updated_dataloader(
                self.model_2,
                self.dataloader_reverse,
                self.dataloader,
                self.aligned_dataloader_reversed,
                batch_size,
            )
            print(
                "Creating updated dataloader by generating new samples "
                + "with model1 for lang2 -> lang1"
            )
            updated_dataloader_reverse = _create_updated_dataloader(
                self.model_1,
                self.dataloader,
                self.dataloader_reverse,
                self.aligned_dataloader,
                batch_size,
            )
            # Train model on augmented dataset
            training = base.BasicMachineTranslationTraining(
                self.model_1, updated_dataloader, self.aligned_valid_dataloader
            )
            print("Training first model on augmented aligned dataset.")
            training.run(loss_fn, optimizer, batch_size, 1, checkpoint=None)
            self.model_1.save(str(epoch))

            training = base.BasicMachineTranslationTraining(
                self.model_2,
                updated_dataloader_reverse,
                self.aligned_valid_dataloader_reverse,
            )
            print("Training second model on reversed augmented aligned dataset.")
            training.run(loss_fn, optimizer, batch_size, 1, checkpoint=None)
            self.model_2.save(str(epoch))


def _create_updated_dataloader(
    model, dataloader, dataloader_reverse, aligned_dataloader, batch_size
):
    additional_data = 2 * len(aligned_dataloader.corpus_input)
    additional_data = batch_size

    corpus = copy.deepcopy(dataloader.corpus)
    dataloader.corpus = corpus[:additional_data]

    dataset = dataloader.create_dataset()
    predictions = _generate_predictions_unaligned(
        model, dataset, dataloader_reverse.encoder, batch_size, len(dataloader.corpus)
    )

    new_corpus_input = aligned_dataloader.corpus_target + predictions
    new_corpus_target = aligned_dataloader.corpus_input + dataloader.corpus

    dataloader.corpus = corpus

    return AlignedDataloader(
        "new_corpus_input",
        "new_corpus_target",
        dataloader.vocab_size,
        encoder_input=aligned_dataloader.encoder_target,
        encoder_target=aligned_dataloader.encoder_input,
        corpus_input=new_corpus_input,
        corpus_target=new_corpus_target,
    )


def _generate_predictions_unaligned(model, dataset, encoder, batch_size, num_samples):
    predictions = []
    for i, inputs in enumerate(dataset.padded_batch(batch_size, padded_shapes=[None])):
        print(f"Generating samples, progression {i*batch_size}/{num_samples}")
        outputs = model.translate(inputs)
        predictions += model.predictions(outputs, encoder, logit=True)

    return predictions
