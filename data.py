# %%
import pandas as pd
import tensorflow as tf
import numpy as np
from utils import *
# %%
train = pd.read_csv('./data/data_2022_01_19.csv')
train['sm'].value_counts()
# %%

train['class'] = train['class'].astype(str)
train['tilt'] = train['tilt'].astype(str)
# %%

train_datagen = FixedImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.05,
    horizontal_flip=True,
    rescale=1./255
)
# %%
train_generator = train_datagen.flow_from_dataframe(
    train,
    y_col=['tilt', 'class'],
    target_size=(256, 256),
    batch_size=32,
    class_mode='multi_output'
)
# %%
model = make_model_pretrain(out_dim=5, multi_task=True)
# %%
optimizer = optimization.create_optimizer(init_lr=1e-4,
                                        num_train_steps=1480,
                                        num_warmup_steps=148,
                                        optimizer_type='adamw')

model.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# %%
def get_data_from_generator(generator, out_dim=[2, 5]):
    while(True):
        data = next(generator)
        yield data[0], {'y_tilt': tf.one_hot(data[1][0], depth=out_dim[0]), 
                        'y_disease': tf.one_hot(data[1][1], depth=out_dim[1])}
        

history = model.fit(get_data_from_generator(train_generator),
                    epochs=100)

# %%
train
# %%
temp = next(train_generator)

# %%
temp[0].shape
# %%
temp[1][0]
# %%
temp[1][1]
# %%
