import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_demo_data():
    """Создает все необходимые демо-данные для работы приложения"""

    print("\n" + "=" * 50)
    print("🛠  Генерация демо-данных")
    print("=" * 50)

    # Создаем папку models если её нет
    os.makedirs('models', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)

    # 1. Демо-данные для распределения классов (10 классов)
    classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    counts = np.array([150, 120, 180, 95, 200, 110, 130, 140, 125, 150])
    np.savez('models/class_distribution.npz', classes=classes, counts=counts)
    logger.info("✓ Создан файл: models/class_distribution.npz")
    print(f"  - Классы: {len(classes)}")
    print(f"  - Всего записей: {sum(counts)}")

    # 2. Демо-данные для истории обучения (50 эпох)
    num_epochs = 50

    # Генерируем реалистичные данные для обучения
    np.random.seed(42)  # Для воспроизводимости

    accuracy = []
    val_accuracy = []
    loss = []
    val_loss = []

    for i in range(num_epochs):
        # Точность на обучении (растет от 40% до 95%)
        base_acc = 0.4 + (0.55 * i / num_epochs)
        noise = np.random.normal(0, 0.02)
        acc = min(0.98, max(0.35, base_acc + noise))
        accuracy.append(acc)

        # Точность на валидации (чуть ниже, более волатильна)
        base_val_acc = 0.38 + (0.54 * i / num_epochs)
        val_noise = np.random.normal(0, 0.03)
        v_acc = min(0.96, max(0.33, base_val_acc + val_noise))
        val_accuracy.append(v_acc)

        # Потери на обучении (убывают)
        base_loss = 1.8 - (1.4 * i / num_epochs)
        loss_noise = np.random.normal(0, 0.05)
        l = max(0.2, base_loss + loss_noise)
        loss.append(l)

        # Потери на валидации
        base_val_loss = 1.9 - (1.4 * i / num_epochs)
        val_loss_noise = np.random.normal(0, 0.07)
        v_l = max(0.25, base_val_loss + val_loss_noise)
        val_loss.append(v_l)

    np.savez('models/training_history.npz',
             accuracy=accuracy,
             val_accuracy=val_accuracy,
             loss=loss,
             val_loss=val_loss)

    logger.info(f"✓ Создан файл: models/training_history.npz")
    print(f"  - Эпох: {num_epochs}")
    print(f"  - Финальная точность (обучение): {accuracy[-1] * 100:.1f}%")
    print(f"  - Финальная точность (валидация): {val_accuracy[-1] * 100:.1f}%")

    # 3. Создаем простую демо-модель Keras
    try:
        import tensorflow as tf

        # Создаем простую модель
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Сохраняем модель
        model.save('models/trained_model.keras')
        logger.info("✓ Создан файл: models/trained_model.keras")
        print("  - Архитектура: CNN для классификации изображений")
        print("  - Вход: 32x32x3, Выход: 10 классов")

    except Exception as e:
        logger.warning(f"Не удалось создать Keras модель: {e}")
        # Создаем заглушку
        with open('models/trained_model.keras', 'w') as f:
            f.write('# Демо-модель. Запустите обучение для создания реальной модели')
        logger.info("✓ Создан файл-заглушка: models/trained_model.keras")

    # 4. Создаем данные для валидационного набора (15 классов)
    valid_classes = np.array([3, 7, 1, 5, 2, 8, 0, 4, 6, 9, 10, 11, 12, 13, 14])
    valid_counts = np.array([45, 42, 38, 35, 32, 28, 25, 22, 20, 18, 15, 12, 10, 8, 5])
    np.savez('models/validation_distribution.npz', classes=valid_classes, counts=valid_counts)
    logger.info("✓ Создан файл: models/validation_distribution.npz")
    print(f"  - Классов в валидации: {len(valid_classes)}")
    print(f"  - Топ-1 класс: Цивилизация {valid_classes[0]} ({valid_counts[0]} записей)")

    # 5. Создаем тестовый .npz файл для демо
    try:
        # Создаем случайные тестовые данные
        np.random.seed(123)
        test_x = np.random.rand(100, 32, 32, 3).astype(np.float32)
        test_y = np.eye(10)[np.random.randint(0, 10, 100)]

        os.makedirs('uploads', exist_ok=True)
        np.savez('uploads/demo_test_data.npz', x=test_x, y=test_y)
        logger.info("✓ Создан демо-тестовый файл: uploads/demo_test_data.npz")
        print("  - Тестовых образцов: 100")
        print("  - Размер изображений: 32x32x3")

    except Exception as e:
        logger.warning(f"Не удалось создать тестовый файл: {e}")

    print("\n" + "=" * 50)
    print("✅ Все демо-данные успешно созданы!")
    print("=" * 50)
    print("\n📁 Структура папок:")
    print("  📂 models/")
    print("     ├── class_distribution.npz     - распределение классов")
    print("     ├── training_history.npz       - история обучения (50 эпох)")
    print("     ├── validation_distribution.npz - данные валидации")
    print("     └── trained_model.keras        - демо-модель")
    print("  📂 uploads/")
    print("     └── demo_test_data.npz         - тестовые данные для загрузки")
    print("\n🚀 Теперь запустите приложение:")
    print("   python app.py")
    print("=" * 50)


if __name__ == '__main__':
    create_demo_data()