import numpy as np
import tensorflow as tf
import os
import logging
from datetime import datetime
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_path='models/cleaned_train_data.npz'):
    logger.info(f"Загрузка данных из {data_path}")

    if not os.path.exists(data_path):
        logger.error(f"Файл {data_path} не найден!")
        logger.info("Создаю демо-данные для тестового обучения...")
        return create_demo_data()

    data = np.load(data_path, allow_pickle=True)
    x = data['x']
    y = data['y']

    logger.info(f"Загружено: X shape {x.shape}, y shape {y.shape}")

    if x.max() > 1.0:
        x = x.astype('float32') / 255.0
        logger.info("Данные нормализованы")

    split = int(0.8 * len(x))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    if len(y_train.shape) == 1:
        num_classes = len(np.unique(y))
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes)
        logger.info(f"Применено one-hot encoding: {num_classes} классов")

    logger.info(f"Train: {len(x_train)} samples, Val: {len(x_val)} samples")

    return x_train, y_train, x_val, y_val


def create_demo_data():
    logger.info("Создание демо-данных...")

    num_samples = 1000
    img_size = 32
    num_classes = 10

    x = np.random.rand(num_samples, img_size, img_size, 3).astype('float32')
    y = np.random.randint(0, num_classes, num_samples)
    y = tf.keras.utils.to_categorical(y, num_classes)

    split = int(0.8 * num_samples)
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    logger.info(f"Создано {num_samples} синтетических образцов")

    return x_train, y_train, x_val, y_val


def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    logger.info(f"Модель создана. Параметров: {model.count_params():,}")
    model.summary(print_fn=logger.info)

    return model


def train_model(epochs=50, batch_size=32):

    logger.info("=" * 60)
    logger.info("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
    logger.info(f"Эпох: {epochs}, Batch size: {batch_size}")
    logger.info("=" * 60)

    x_train, y_train, x_val, y_val = load_and_preprocess_data()

    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[1]
    model = create_model(input_shape, num_classes)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv', separator=';', append=False)
    ]

    logger.info("Старт обучения...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    model.save('models/trained_model.keras')
    logger.info("✅ Финальная модель сохранена в models/trained_model.keras")

    np.savez('models/training_history.npz',
             accuracy=history.history['accuracy'],
             val_accuracy=history.history['val_accuracy'],
             loss=history.history['loss'],
             val_loss=history.history['val_loss'])
    logger.info("✅ История обучения сохранена")

    if os.path.exists('models/cleaned_train_data.npz'):
        data = np.load('models/cleaned_train_data.npz', allow_pickle=True)
        y_all = data['y']
        if len(y_all.shape) > 1:
            y_all = np.argmax(y_all, axis=1)
        classes, counts = np.unique(y_all, return_counts=True)
        np.savez('models/class_distribution.npz', classes=classes, counts=counts)
        logger.info("✅ Распределение классов сохранено")

    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    logger.info("=" * 60)
    logger.info("ИТОГИ ОБУЧЕНИЯ:")
    logger.info(f"Точность на обучении: {final_train_acc * 100:.2f}%")
    logger.info(f"Точность на валидации: {final_val_acc * 100:.2f}%")
    logger.info(f"Потери на обучении: {final_train_loss:.4f}")
    logger.info(f"Потери на валидации: {final_val_loss:.4f}")
    logger.info("=" * 60)

    return model, history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Обучение модели классификации сигналов')
    parser.add_argument('--epochs', type=int, default=50, help='Количество эпох')
    parser.add_argument('--batch_size', type=int, default=32, help='Размер батча')

    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)

    train_model(epochs=args.epochs, batch_size=args.batch_size)
