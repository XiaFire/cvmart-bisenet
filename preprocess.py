import os
from glob import glob
import random

msks = glob('/home/data/*/*.png')
random.shuffle(msks)
ind = len(msks) // 5
train_msks = msks[ind:]
valid_msks = msks[:ind]
print(f"train {len(train_msks)}, valid {len(valid_msks)}")

train_imgs = [x.replace('png', 'jpg') for x in train_msks]
valid_imgs = [x.replace('png', 'jpg') for x in valid_msks]

trns = [img+','+msk for img, msk in zip(train_imgs, train_msks) if os.path.exists(img)]
vals = [img+','+msk for img, msk in zip(valid_imgs, valid_msks) if os.path.exists(img)]
print(f"After Merge: train {len(train_msks)}, valid {len(valid_msks)}")

with open('train.txt', 'w') as f:
    f.write('\n'.join(trns))
with open('val.txt', 'w') as f:
    f.write('\n'.join(vals))
