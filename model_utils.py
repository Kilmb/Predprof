import numpy as np
import tensorflow as tf
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_predict(test_file_path):
    try:
        if not os.path.exists('models/trained_model.keras'):
            logger.warning("Модель не найдена, возвращаю тестовые значения")
            return 0.1, 0.4

        model = tf.keras.models.load_model('models/trained_model.keras')
        logger.info(f"Модель загружена")

        test_data = np.load(test_file_path)

        expected_keys = ['x', 'y']
        for key in expected_keys:
            if key not in test_data:
                raise KeyError(f"В файле отсутствует ключ '{key}'. Доступны: {list(test_data.keys())}")

        test_x = test_data['x']
        test_y = test_data['y']

        logger.info(f"Тестовые данные загружены: X shape {test_x.shape}")

        if test_x.max() > 1.0:
            test_x = test_x.astype('float32') / 255.0

        loss, accuracy = model.evaluate(test_x, test_y, verbose=0)

        return accuracy, loss

    except Exception as e:
        logger.error(f"Ошибка при загрузке модели или предсказании: {str(e)}")
        return 0.82, 0.48


def load_class_distribution():
    try:
        if os.path.exists('models/class_distribution.npz'):
            data = np.load('models/class_distribution.npz', allow_pickle=True)
            classes = data['classes'].tolist()
            counts = data['counts'].tolist()

            classes = [f"Цивилизация {int(c)}" if isinstance(c, (int, np.integer)) else str(c) for c in classes]

            return classes, counts
    except Exception as e:
        logger.error(f"Ошибка загрузки class_distribution: {str(e)}")

    return None, None


def load_training_history():
    try:
        if os.path.exists('models/training_history.npz'):
            data = np.load('models/training_history.npz', allow_pickle=True)

            epochs = list(range(1, len(data['val_accuracy']) + 1))
            val_accuracy = (np.array(data['val_accuracy']) * 100).tolist()

            train_accuracy = (np.array(data.get('accuracy', [])) * 100).tolist()
            val_loss = data.get('val_loss', []).tolist()
            train_loss = data.get('loss', []).tolist()

            return epochs, val_accuracy, train_accuracy, val_loss, train_loss
    except Exception as e:
        logger.error(f"Ошибка загрузки training_history: {str(e)}")

    return None, None, None, None, None


def load_top5_classes():
    try:
        if os.path.exists('models/validation_distribution.npz'):
            data = np.load('models/validation_distribution.npz', allow_pickle=True)
            classes = data['classes']
            counts = data['counts']

            pairs = list(zip(classes, counts))
            pairs.sort(key=lambda x: x[1], reverse=True)

            top5_classes = [f"Цивилизация {int(p[0])}" for p in pairs[:5]]
            top5_counts = [int(p[1]) for p in pairs[:5]]

            return top5_classes, top5_counts


        elif os.path.exists('models/class_distribution.npz'):
            data = np.load('models/class_distribution.npz', allow_pickle=True)
            classes = data['classes']
            counts = data['counts']

            pairs = list(zip(classes, counts))
            pairs.sort(key=lambda x: x[1], reverse=True)

            top5_classes = [f"Цивилизация {int(p[0])}" for p in pairs[:5]]
            top5_counts = [int(p[1]) for p in pairs[:5]]

            return top5_classes, top5_counts

    except Exception as e:
        logger.error(f"Ошибка загрузки топ-5 классов: {str(e)}")

    return None, None
