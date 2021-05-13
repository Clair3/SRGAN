import os
import shutil
from random import shuffle

path_hr = '/home/claire/PycharmProjects/SRGAN/dataset_souris/DATASET_HR/'
# path_lr = '/home/claire/PycharmProjects/SRGAN/dataset_souris_2/DATASET_LR/'

path_train_hr = '/home/claire/PycharmProjects/SRGAN/dataset_souris/TRAIN_DATASET_HR/'
# path_train_lr = '/home/claire/PycharmProjects/SRGAN/dataset_souris/TRAIN_DATASET_LR/'

path_test_hr = '/home/claire/PycharmProjects/SRGAN/dataset_souris/TEST_DATASET_HR/'
# path_test_lr = '/home/claire/PycharmProjects/SRGAN/dataset_souris/TEST_DATASET_LR/'

path_val_hr = '/home/claire/PycharmProjects/SRGAN/dataset_souris/VAL_DATASET_HR/'
# path_val_lr = '/home/claire/PycharmProjects/SRGAN/dataset_souris/VAL_DATASET_LR/'


os.mkdir(path_train_hr)
# os.mkdir(path_train_lr)

os.mkdir(path_test_hr)
# os.mkdir(path_test_lr)

os.mkdir(path_val_hr)
# os.mkdir(path_val_lr)

print(len(os.listdir(path_hr)))
# print(len(os.listdir(path_lr)))

lst_HR = os.listdir(path_hr)

shuffle(lst_HR)
size = round(0.80 * len(lst_HR))
size_val = round(0.995 * len(lst_HR))


for file in lst_HR[:size]:
    shutil.move(path_hr + file, path_train_hr)

# for file in lst_HR[:size]:
   # shutil.move(path_lr + file, path_train_lr)

for file in lst_HR[size:size_val]:
    shutil.move(path_hr + file, path_test_hr)

# for file in lst_HR[size:size_val]:
    # shutil.move(path_lr + file, path_test_lr)

for file in lst_HR[size_val:]:
    shutil.move(path_hr + file, path_val_hr)

# for file in lst_HR[size_val:]:
   # shutil.move(path_lr + file, path_val_lr)

print(len(os.listdir(path_hr)))
# print(len(os.listdir(path_lr)))
print(len(os.listdir(path_train_hr)))
# print(len(os.listdir(path_train_lr)))
print(len(os.listdir(path_test_hr)))
# print(len(os.listdir(path_test_lr)))
print(len(os.listdir(path_val_hr)))
# print(len(os.listdir(path_val_lr)))


'''
for file in os.listdir(path_train_hr):
    shutil.move(path_train_hr + file, path_hr)

for file in os.listdir(path_train_lr):
    shutil.move(path_train_lr + file, path_lr)

for file in os.listdir(path_test_hr):
    shutil.move(path_test_hr + file, path_hr)

for file in os.listdir(path_test_lr):
    shutil.move(path_test_lr + file, path_lr)

for file in os.listdir(path_val_hr):
    shutil.move(path_val_hr + file, path_hr)

for file in os.listdir(path_val_lr):
    shutil.move(path_val_lr + file, path_lr)

print(len(os.listdir(path_hr)))
print(len(os.listdir(path_lr)))'''
