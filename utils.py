from tensorflow.keras.models import Model
from official.nlp import optimization
from tensorflow.keras.layers import BatchNormalization, Dense, Input, Conv2D, Activation, Dropout, MaxPooling2D, GlobalAveragePooling2D, Flatten, Concatenate
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
                       comp=False,
                       concat=True):
    
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
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inp, x)

    if comp:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        0.0001,
        decay_steps=250,
        decay_rate=0.96,
        staircase=True)

        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # loss = tf.keras.losses.binary_crossentropy

        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model

def make_model(model_name, concat=False):
    epochs = 100
    batch_size = 16
    image_size=(256, 256)
    init_lr = 1e-4
    model = make_model_pretrain(model_name=model_name, concat=concat)
    
    steps_per_epoch = 814 // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC()])
    return model

class FixedImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x = ((x/255.) - 0.5) * 2.
        return x