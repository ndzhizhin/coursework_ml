from imports import *


def trained_model(x_train, y_train):
    model = Sequential()

    # Первый сверточный слой
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(x_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Второй сверточный слой
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Третий сверточный слой
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Выравнивание
    model.add(Flatten())

    # Полносвязный слой
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))

    # Выходной слой
    model.add(Dense(units=y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

