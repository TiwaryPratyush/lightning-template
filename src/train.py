import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from src.datamodules.dog_breed_datamodule import DogBreedDataModule
from src.models.dog_breed_classifier import DogBreedClassifier
from src.utils.train_eval_wrapper import train_eval_wrapper
from src.utils.logging_utils import setup_logging

logger = setup_logging()

@train_eval_wrapper
def train():
    # Initialize the data module
    data_module = DogBreedDataModule(data_dir="data/dog_breeds", batch_size=32, img_size=224)

    # Initialize the model
    model = DogBreedClassifier(num_classes=data_module.num_classes, learning_rate=0.001)

    # Setup logger
    tb_logger = TensorBoardLogger(save_dir="logs", name="dog_breed_classification")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='logs/dog_breed_classification',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary()

    # Setup trainer
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        callbacks=[checkpoint_callback, progress_bar, model_summary],
        logger=tb_logger
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    train()