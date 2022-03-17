import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def plot_tensorboard_loss(version):
    ea = event_accumulator.EventAccumulator(f"./lightning_logs/version_{version}",
                                            size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    epoch = ea.Scalars("epoch")
    train_loss = ea.Scalars("train_loss")
    val_loss = ea.Scalars("val_loss")

    plt.plot([train_loss[i].step for i in range(len(train_loss))],
             [train_loss[i].value for i in range(len(train_loss))], label="training loss")
    plt.plot([val_loss[i].step for i in range(len(val_loss))],
             [val_loss[i].value for i in range(len(val_loss))], label="validation loss")

    current_epoch = epoch[0].value
    last_step = -1
    x_ticks_idx = list()
    x_ticks_labels = list()
    for i in range(len(epoch)):
        if epoch[i].value > current_epoch:
            if int(current_epoch) % 2 == 0:
                x_ticks_idx.append(last_step)
                x_ticks_labels.append(int(current_epoch))
            current_epoch = epoch[i].value
        last_step = epoch[i].step

    plt.xticks(x_ticks_idx, x_ticks_labels)
    plt.legend()
    plt.savefig("train_loss_conv_net.pdf")
    plt.show()


if __name__ == '__main__':
    plot_tensorboard_loss(8)
    # plot_tensorboard_loss(9)
