import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    y=model.forward(X)
    preds=np.array([np.argmax(y,axis=1)]).T
    preds=one_hot_encode(preds,targets.shape[1])
    diff=np.sum(preds==targets,axis=1)
    correct=diff[diff==preds.shape[1]]
    accuracy=correct.shape[0]/targets.shape[0]
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        y=self.model.forward(X_batch)
        self.model.backward(X_batch,y,Y_batch)
        loss = cross_entropy_loss(Y_batch,y)
        self.model.w=self.model.w - self.learning_rate*self.model.grad
        #y=self.model.forward(X_batch)
        
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val

def plot_weights(w,filename):
    fig1, axs = plt.subplots(2,10,figsize = (10,4), sharex=True, sharey=True)
    for i in range(2):
        for j in range(10):
            weight=np.reshape(w[i][1:,j],(28,28))
            axs[i,j].imshow(weight,cmap="gray")
            axs[i,j].axis("off")
    plt.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(filename,bbox_inches='tight')
    plt.show()


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # Plotting of softmax weights (Task 4b)
    model0 = SoftmaxModel(l2_reg_lambda=0.0)
    trainer0 = SoftmaxTrainer(
        model0, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg00, val_history_reg00 = trainer0.train(num_epochs)
    # You can finish the rest of task 4 below this point.
    w=[model0.w,model1.w]
    plot_weights(w,"task4b_softmax_weight.png")

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    models=[]
    fig=plt.figure()
    for lamb in l2_lambdas:
        model=SoftmaxModel(lamb)
        trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
        )
        train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
        utils.plot_loss(val_history_reg01["accuracy"], f"lambda = {lamb}")
        models.append(model)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight
    fig1=plt.figure()
    l2_weights = [np.linalg.norm(model.w) for model in models]
    plt.plot(l2_lambdas,l2_weights)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$||w||_2$")
    plt.title("Norm of weights")
    plt.grid()
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()


if __name__ == "__main__":
    main()
