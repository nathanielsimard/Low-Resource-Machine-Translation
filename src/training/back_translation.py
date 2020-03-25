import copy

import tensorflow as tf

from src.dataloader import AlignedDataloader
from src.training import base


class BackTranslationTraining(base.Training):
    """Train two models at the same time improving the model AND the dataset at the same time."""

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
        """Create BackTranslationTraining."""
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
        """Run the back translation training.

        Note that the checkpoint are only for the pretrain
        step where both models are first train on the aligned
        dataset without any data augmentation.
        """
        self.model_1.title = self.model_1.title + "-model-1"
        self.model_2.title = self.model_2.title + "-model-2"

        print("Training first model on aligned dataset.")
        training = base.BasicMachineTranslationTraining(
            self.model_1, self.aligned_dataloader, self.aligned_valid_dataloader
        )
        training.run(loss_fn, optimizer, batch_size, num_epoch, checkpoint=checkpoint)

        print("Training second model on reversed aligned dataset.")
        training = base.BasicMachineTranslationTraining(
            self.model_2,
            self.aligned_dataloader_reversed,
            self.aligned_valid_dataloader_reverse,
        )
        training.run(loss_fn, optimizer, batch_size, num_epoch, checkpoint=checkpoint)

        for epoch in range(1, num_epoch + 1):
            print(
                "Creating updated dataloader by generating new samples "
                + "with model2 for lang1 -> lang2"
            )
            updated_dataloader = create_updated_dataloader(
                self.model_2,
                self.dataloader_reverse,
                self.dataloader,
                self.aligned_dataloader_reversed,
                24
                * batch_size,  # Batch size can be higher because it is one word after the other (seq of 1)
            )

            print(
                "Creating updated dataloader by generating new samples "
                + "with model1 for lang2 -> lang1"
            )
            updated_dataloader_reverse = create_updated_dataloader(
                self.model_1,
                self.dataloader,
                self.dataloader_reverse,
                self.aligned_dataloader,
                24
                * batch_size,  # Batch size can be higher because it is one word after the other (seq of 1)
            )

            print("Training first model on augmented aligned dataset.")
            training = base.BasicMachineTranslationTraining(
                self.model_1, updated_dataloader, self.aligned_valid_dataloader
            )
            training.run(loss_fn, optimizer, batch_size, 1, checkpoint=None)
            self.model_1.save(str(epoch))

            print("Training second model on reversed augmented aligned dataset.")
            training = base.BasicMachineTranslationTraining(
                self.model_2,
                updated_dataloader_reverse,
                self.aligned_valid_dataloader_reverse,
            )
            training.run(loss_fn, optimizer, batch_size, 1, checkpoint=None)
            self.model_2.save(str(epoch))


def create_updated_dataloader(
    model,
    dataloader,
    dataloader_reverse,
    aligned_dataloader,
    batch_size,
    aumgentation_ratio=2,
):
    """Create updated dataloader by augmenting the old dataset with predictions.

    Generate dataloader lang1 -> lang2 with model that translate lang2 -> lang1.
    It uses those predictions to augmente the inputs of the dataset lang1 -> lang2.
    Targets are always the original real data.
    """
    num_additional_data = aumgentation_ratio * len(aligned_dataloader.corpus_input)

    corpus = copy.deepcopy(dataloader.corpus)
    # Reduce the corpus to only predict the same number as the number of additional data.
    dataloader.corpus = corpus[:num_additional_data]

    dataset = dataloader.create_dataset()
    predictions = _generate_predictions_unaligned(
        model, dataset, dataloader_reverse.encoder, batch_size, len(dataloader.corpus)
    )

    new_corpus_input = aligned_dataloader.corpus_target + predictions
    new_corpus_target = aligned_dataloader.corpus_input + dataloader.corpus

    # The corpus of the dataloader must not change.
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
