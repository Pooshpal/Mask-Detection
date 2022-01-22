
import os
import random
import shutil


# os.mkdir("../data/train")
# os.mkdir("../data/test/")
# os.mkdir("../data/valid/")
# os.mkdir("../data/train/without")
# os.mkdir("../data/train/with")
# os.mkdir("../data/test/without")
# os.mkdir("../data/test/with")
# os.mkdir("../data/valid/without")
# os.mkdir("../data/valid/with")

# Get all filenames into a python list

with_mask = os.listdir("with_mask")
without_mask = os.listdir("without_mask")

# Split the data 70-15-15 %

# Split percentage for "with_mask"

with_first = int(0.7*(len(with_mask)))
with_second = int(0.85*(len(with_mask)))

# Split percentage for "without_mask"

without_first = int(0.7*(len(without_mask)))
without_second = int(0.85*(len(without_mask)))

# Shuffle the data

random.shuffle(with_mask)
random.shuffle(without_mask)

# Create lists with filenames for the split data

# For with_mask

train_with = with_mask[:with_first]
test_with = with_mask[with_first:with_second]
valid_with = with_mask[with_second:]

# For without_mask

train_without = without_mask[:without_first]
test_without = without_mask[without_first:without_second]
valid_without = without_mask[without_second:]

# Move the files to the respective directories using shutil library

for file in train_with:
    shutil.move("../data/with_mask/"+file, "../data/train/with")

for file in train_without:
    shutil.move("../data/without_mask/"+file, "../data/train/without")

print("\nFinished creating training dataset\n")

for file in test_with:
    shutil.move("../data/with_mask/"+file, "../data/test/with")

for file in test_without:
    shutil.move("../data/without_mask/"+file, "../data/test/without")

print("\nFinished creating test dataset")

for file in valid_with:
    shutil.move("../data/with_mask/"+file, "../data/valid/with")

for file in valid_without:
    shutil.move("../data/without_mask/"+file, "../data/valid/without")

print("\nFinished creating the validation dataset\n")

print("\nFinished building Dataset!\n\n")


