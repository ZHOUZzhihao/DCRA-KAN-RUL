import matplotlib.pyplot as plt


def visualize(result, rmse):

    # the true remaining useful life of the testing samples
    true_rul = result.iloc[:, 0:1].to_numpy()
    # the predicted remaining useful life of the testing samples
    pred_rul = result.iloc[:, 1:].to_numpy()
    l = len(pred_rul)

    plt.figure(figsize=(10, 6))
    plt.axvline(x=l, c='r', linestyle='--')
    plt.plot(true_rul, linewidth = 1.5,label='Actual Data')
    plt.plot(pred_rul,linewidth = 1.5,label='Predicted Data')
    plt.title('RUL Prediction on CMAPSS Data',fontsize=12)
    plt.legend()
    plt.xlabel("Flight Cycles",fontsize=12)
    plt.ylabel("Remaining Useful Life",fontsize=12)
    plt.show()