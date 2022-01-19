from tensorflow.keras.models import Model
from official.nlp import optimization
from tensorflow.keras.layers import BatchNormalization, Dense, Input, Conv2D, Activation, Dropout, MaxPooling2D, GlobalAveragePooling2D, Flatten, Concatenate
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def dense_block(units, dropout=0.2, activation='relu', name='fc1'):

    def layer_wrapper(inp):
        x = Dense(units, name=name)(inp)
        x = BatchNormalization(renorm=True, name='{}_bn'.format(name))(x)
        x = Activation(activation, name='{}_act'.format(name))(x)
        x = Dropout(dropout, name='{}_dropout'.format(name))(x)
        return x

    return layer_wrapper

def make_model_pretrain(shape=(256, 256, 3), 
                        dropout=0.3, activation='relu', 
                        model_name='vgg',
                       concat=True,
                       out_dim=1):
    
    inp = tf.keras.layers.Input([256, 256, 3], dtype = tf.float32)    
    if model_name == 'vgg':
        dense = tf.keras.applications.VGG19(include_top=False, input_shape=(256, 256, 3))
        for layer in dense.layers:
            layer.trainable = False
        dense_trainable = tf.keras.applications.VGG19(include_top=False, input_shape=(256, 256, 3))
        dense_trainable._name = 'trainable_VGG19'
    elif model_name == 'vgg16':
        dense = tf.keras.applications.VGG16(include_top=False, input_shape=(256, 256, 3))
        for layer in dense.layers:
            layer.trainable = False
        dense_trainable = tf.keras.applications.VGG16(include_top=False, input_shape=(256, 256, 3))
        dense_trainable._name = 'trainable_VGG16'
    elif model_name == 'dense121':
        dense = tf.keras.applications.DenseNet121(include_top=False, input_shape=(256, 256, 3))
        for layer in dense.layers:
            layer.trainable = False
        dense_trainable = tf.keras.applications.DenseNet121(include_top=False, input_shape=(256, 256, 3))
        dense_trainable._name = 'trainable_DenseNet121'
    elif model_name == 'dense169':
        dense = tf.keras.applications.DenseNet169(include_top=False, input_shape=(256, 256, 3))
        for layer in dense.layers:
            layer.trainable = False
        dense_trainable = tf.keras.applications.DenseNet169(include_top=False, input_shape=(256, 256, 3))
        dense_trainable._name = 'trainable_DenseNet169'
    elif model_name == 'dense201':
        dense = tf.keras.applications.DenseNet201(include_top=False, input_shape=shape)
        for layer in dense.layers:
            layer.trainable = False
        dense_trainable = tf.keras.applications.DenseNet201(include_top=False, input_shape=shape)
        dense_trainable._name = 'trainable'
    else:
        assert print('Wrong model name!')
    # x = tf.cast(inp, tf.float32)
    # x = tf.keras.applications.densenet.preprocess_input(inp)

    x_trainable = dense_trainable(inp)
    x = GlobalAveragePooling2D()(x_trainable)
    
    if concat:
        x_non = dense(inp)
        x_non = GlobalAveragePooling2D()(x_non)
        x = Concatenate()([x_non, x])
        
    x = dense_block(1024, dropout=dropout, activation=activation, name='fc1')(x)
    x = dense_block(512, dropout=dropout, activation=activation, name='fc2')(x)
    x = dense_block(64, dropout=dropout, activation=activation, name='fc3')(x)
    #     x = dense_block(64, dropout=0.25, activation='relu', name='fc3')(x)
    if out_dim == 1:
        x = Dense(out_dim, activation='sigmoid')(x)
    else:
        x = Dense(out_dim, activation='softmax')(x)

    model = Model(inp, x)
    return model

def make_model(model_name, concat=False, len_data=814, out_dim=1):
    epochs = 100
    batch_size = 16
    image_size=(256, 256)
    init_lr = 1e-4
    model = make_model_pretrain(model_name=model_name,
                                concat=concat, out_dim=out_dim)
    
    steps_per_epoch = len_data // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

class FixedImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x = ((x/255.) - 0.5) * 2.
        return x
    
def train_model(concat, normalize, model_name, save_path, data, init_lr=1e-4, batch_size=32,
                epochs=100, seeds=[1, 11, 111, 1111, 1004], test_size=0.2, out_dim=5):
    histories = []
    image_size=(256, 256)
    for i in range(5):
        train, val = train_test_split(data, test_size=test_size, random_state=seeds[i], stratify=data['class'])
        
        steps_per_epoch = len(train) // batch_size
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1*num_train_steps)
        
        train_datagen = FixedImageDataGenerator(
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting
            featurewise_center=True if normalize else False,
            featurewise_std_normalization=True if normalize else False,
            brightness_range=[0.8, 1.2],
            width_shift_range=0.05,
            horizontal_flip=True,
            rescale=1./255
        )
        train_generator = train_datagen.flow_from_dataframe(
           train,
            target_size=image_size,
            batch_size=batch_size,
        )


        val_datagen = FixedImageDataGenerator(
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting
            featurewise_center=True,
            featurewise_std_normalization=True,
            rescale=1./255
        )

        val_generator = val_datagen.flow_from_dataframe(
           val,
            target_size=image_size,
            batch_size=batch_size,
        )
        model = make_model_pretrain(dropout=0.3, activation='relu', 
                                    model_name=model_name, concat=concat,
                                   out_dim=out_dim)
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        # log_dir = "logs/dense169_relu"
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path+'/model_{0}_{1}.h5'.format(model_name, i), 
                                     verbose=2, monitor='val_accuracy',save_best_only=True, mode='auto')  
#         es_callback = tf.keras.callbacks.EarlyStopping(
#             monitor='val_accuracy',
#             patience=30,
#             verbose=0,
#             mode='auto',
#             restore_best_weights=True
#         )
        history = model.fit(train_generator, 
                            validation_data=val_generator, 
                            epochs=epochs, 
                            callbacks=[checkpoint])
        histories.append(history)
    return histories