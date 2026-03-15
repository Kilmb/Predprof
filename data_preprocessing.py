import numpy as np
import re

# Загружаем исходные данные
data = np.load('путь/к/скачанному/файлу/train_data.npz') # Укажите свой путь
train_x = data['train_x']
train_y_raw = data['train_y']

# Функция для извлечения числа из строки
def extract_number_from_string(s):
    # Если это не строка, а байты, декодируем
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    # Ищем все последовательности цифр
    numbers = re.findall(r'\d+', str(s))
    if numbers:
        # Берем первое найденное число и преобразуем в int
        return int(numbers[0])
    else:
        # Если чисел нет, возвращаем -1 как признак ошибки
        return -1

# Очищаем метки
train_y_clean = np.array([extract_number_from_string(label) for label in train_y_raw])

# Проверяем, нет ли пропусков (элементов -1)
if np.any(train_y_clean == -1):
    print("Предупреждение: не удалось извлечь число из некоторых строк. Проверьте данные.")
    # Можно удалить такие записи или обработать иначе
    # valid_indices = train_y_clean != -1
    # train_x = train_x[valid_indices]
    # train_y_clean = train_y_clean[valid_indices]
else:
    print("Все метки успешно восстановлены!")

# Сохраняем очищенные данные для обучения
np.savez_compressed('models/cleaned_train_data.npz', x=train_x, y=train_y_clean)

# Анализируем распределение классов для будущих диаграмм
unique, counts = np.unique(train_y_clean, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Распределение классов:", class_distribution)

# Сохраняем распределение для фронтенда
np.savez('models/class_distribution.npz', classes=unique, counts=counts)
