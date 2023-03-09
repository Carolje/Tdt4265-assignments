import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
import torchvision
from dataloaders import load_cifar10_transfer
from trainer import Trainer
from trainer import compute_loss_and_accuracy


class Model(nn.Module):

    def __init__(self):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        self.model=torchvision.models.resnet18(pretrained=True)
        self.model.fc=nn.Linear(512,10)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True
           


    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        x=self.model(x)
        return x

def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10_transfer(batch_size)
    model = Model()
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task4")
    train_loss,train_acc=compute_loss_and_accuracy(trainer.dataloader_train,model,nn.CrossEntropyLoss())
    val_loss,val_acc=compute_loss_and_accuracy(trainer.dataloader_val,model,nn.CrossEntropyLoss())
    test_loss,test_acc=compute_loss_and_accuracy(trainer.dataloader_test,model,nn.CrossEntropyLoss())
    print("Train accuracy", round(train_acc,3))
    print("Validation accuracy", round(val_acc,3))
    print("Test accuracy", round(test_acc,3))

if __name__ == "__main__":
    main()