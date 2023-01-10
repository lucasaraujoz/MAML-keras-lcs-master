from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Activation, Flatten, Dropout, GlobalAveragePooling2D, Concatenate
from keras.models  import  Sequential, Model
from keras.optimizers import SGD,Adam
from keras import regularizers
import visualkeras
img_shape = 128
dropout = 0.3
def VGG16_Model():
    model_fundus = DenseNet121(weights='imagenet',  # importou a CNN
                               include_top=False,
                               input_shape=(img_shape, img_shape, 3))

    model_fundus_crop = DenseNet121(weights='imagenet',  # importou a CNN
                                    include_top=False,
                                    input_shape=(img_shape, img_shape, 3))

    for layer in model_fundus_crop.layers:  ##### tomar cuidado aqui!!!
        layer._name = layer.name + str("_2")

    # Tensor
    base_model_fundus = model_fundus.layers[-1].output
    base_model_fundus_crop = model_fundus_crop.layers[-1].output

    # INCLUIR O CLASSIFICADOR
    # x = base_model.output
    x = Concatenate(axis=-1)([base_model_fundus, base_model_fundus_crop])  # duas entradas #10, #10 #20
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)  # virou um tensor
    x = Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu')(
        x)  # Neurônios (256) #kernel_regularizer=regularizers.l2(0.001)
    x = Dropout(dropout)(x)
    x = Dense(128, kernel_regularizer=regularizers.l2(0.001), activation='relu')(x)  # Neurônios (128)
    x = Dropout(dropout)(x)
    predictions = Dense(2, activation='sigmoid')(x)
    dual_model = Model(inputs=[model_fundus.input, model_fundus_crop.input], outputs=predictions)

    return dual_model

model = VGG16_Model()

model.fit()
# model.summary()
from keras.utils.vis_utils import plot_model
dot_img_file = '/model_1.png'
plot_model(model, to_file=dot_img_file, show_shapes=True)

import matplotlib.pyplot as plt

lalala = keras.models()
def modelfit(model, train_fundus, train_fundus_crop, y_train):
    # train_samples = len(y_train)
    # validation_samples = len()

    adam = Adam(learning_rate=1e-5)
    model.compile(adam, loss='sparse_categorical_crossentropy', metrics=['acc'])

    # define the checkpoint
    # filepath="weights-{epoch:02d}-{loss:.4f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]
    history = model.fit([train_fundus, train_fundus_crop], y_train,
                        epochs=10,
                        # steps_per_epoch=int(train_samples/BATCH_SIZE),
                        batch_size=32,
                        # callbacks=callbacks_list,
                        validation_split=0.25,  # 0.25
                        # validation_steps=int(validation_samples/1),
                        shuffle=True)

    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(10)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    # plt.title('Training Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    # plt.title('Training Loss')
    plt.show()