import matplotlib.pyplot as plt
import pandas as pd

path = '/content/drive/MyDrive/SRGAN_pytorch/statistics/batch_size8_4_train_results.csv'
name_experiment = 'batch=32_Crop=176_epochs=30'
data = pd.read_csv(path)
plt.plot(data['Loss_D'],  '--', color='blue', label='Loss_D')
plt.plot(data['Loss_G_ADV'],  '--', color='red', label='Loss_G_ADV')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("adversarial losses")
plt.ylim(0)
plt.legend()
plt.savefig("/content/drive/MyDrive/SRGAN_pytorch/statistics/" + name_experiment + "_adversarial_losses.png")
plt.show()



plt.plot(data['Loss_G'], '--', color='red', label='generator loss')
plt.plot(data['Loss_G_TV'],  '--', color='orange', label='Loss_TV')
plt.plot(0.0001 * data['Loss_G_ADV'],  '--', color='blue', label='Loss_ADV')
plt.plot(data['Loss_G_MSE'],  '--', color='green', label='Loss_MSE')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("g loss")
plt.ylim(0)
plt.legend()
plt.savefig("/content/drive/MyDrive/SRGAN_pytorch/statistics/" + name_experiment + "_generative_losses.png")
plt.show()



plt.plot(data['PSNR'],  '--', color='red', label='PSNR')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Metrics")
plt.ylim(0)
plt.legend()
plt.savefig("/content/drive/MyDrive/SRGAN_pytorch/statistics/" + name_experiment + "_PSNR.png")
plt.show()

plt.plot(data['SSIM'],  '--', color='blue', label='SSIM')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("Metrics")
plt.ylim(0)
plt.legend()
plt.savefig("/content/drive/MyDrive/SRGAN_pytorch/statistics/" + name_experiment + "_SSIM.png")
plt.show()