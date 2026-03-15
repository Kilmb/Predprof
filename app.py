from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import numpy as np
import os
import logging
from datetime import datetime
from model_utils import load_model_and_predict, load_class_distribution, load_training_history, load_top5_classes

app = Flask(__name__)
app.secret_key = 'supersecretkey123'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  
app.config['MAX_FORM_MEMORY_SIZE'] = 500 * 1024 * 1024 

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_EPOCHS = 10
MAX_EPOCHS = 100


def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT NOT NULL,
                  first_name TEXT,
                  last_name TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='upload_logs'")
    table_exists = c.fetchone()

    if table_exists:
        c.execute("PRAGMA table_info(upload_logs)")
        columns = [column[1] for column in c.fetchall()]

        if 'file_size' not in columns:
            c.execute('''CREATE TABLE upload_logs_new
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          user_id INTEGER,
                          filename TEXT,
                          file_size INTEGER DEFAULT 0,
                          accuracy REAL,
                          loss REAL,
                          uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                          FOREIGN KEY (user_id) REFERENCES users (id))''')

            c.execute('''INSERT INTO upload_logs_new (id, user_id, filename, accuracy, loss, uploaded_at)
                         SELECT id, user_id, filename, accuracy, loss, uploaded_at FROM upload_logs''')

            c.execute("DROP TABLE upload_logs")
            c.execute("ALTER TABLE upload_logs_new RENAME TO upload_logs")
            logger.info("Таблица upload_logs обновлена: добавлена колонка file_size")
    else:
        c.execute('''CREATE TABLE upload_logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      filename TEXT,
                      file_size INTEGER DEFAULT 0,
                      accuracy REAL,
                      loss REAL,
                      uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS uploads_tracking
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  filename TEXT,
                  file_path TEXT,
                  status TEXT DEFAULT 'uploaded',
                  uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')

    test_users = [
        ('admin', 'admin123', 'admin', 'Главный', 'Администратор'),
        ('user', 'user123', 'user', 'Обычный', 'Пользователь'),
        ('ivan', '123456', 'user', 'Иван', 'Петров'),
        ('maria', '123456', 'user', 'Мария', 'Иванова')
    ]

    for user in test_users:
        try:
            c.execute(
                "INSERT OR IGNORE INTO users (username, password, role, first_name, last_name) VALUES (?, ?, ?, ?, ?)",
                user)
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    logger.info("База данных инициализирована")


init_db()


@app.errorhandler(413)
def request_entity_too_large(error):
    flash('Файл слишком большой. Максимальный размер: 500MB')
    logger.warning(f"Попытка загрузки слишком большого файла")
    return redirect(request.url or url_for('user_dashboard'))


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[3]
            session['first_name'] = user[4]
            session['last_name'] = user[5]

            logger.info(f"Пользователь {username} ({user[3]}) вошел в систему")

            if user[3] == 'admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            flash('Неверное имя пользователя или пароль')
            logger.warning(f"Неудачная попытка входа: {username}")

    return render_template('login.html')


@app.route('/logout')
def logout():
    username = session.get('username', 'Неизвестно')
    logger.info(f"Пользователь {username} вышел из системы")
    session.clear()
    return redirect(url_for('login'))


@app.route('/admin', methods=['GET', 'POST'])
def admin_dashboard():
    if 'role' not in session or session['role'] != 'admin':
        logger.warning(f"Попытка доступа к админ-панели без прав: {session.get('username')}")
        return redirect(url_for('login'))

    if request.method == 'POST':
        fname = request.form['first_name']
        lname = request.form['last_name']
        uname = request.form['username']
        pwd = request.form['password']
        role = request.form['role']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, role, first_name, last_name) VALUES (?, ?, ?, ?, ?)",
                      (uname, pwd, role, fname, lname))
            conn.commit()
            flash('Пользователь успешно создан!')
            logger.info(f"Администратор {session['username']} создал пользователя {uname} с ролью {role}")
        except sqlite3.IntegrityError:
            flash('Ошибка: Имя пользователя уже существует.')
            logger.warning(f"Ошибка создания пользователя {uname}: имя уже существует")
        conn.close()
        return redirect(url_for('admin_dashboard'))

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT id, username, role, first_name, last_name, created_at FROM users")
    users = c.fetchall()

    try:
        c.execute('''SELECT COUNT(*) as total, AVG(accuracy) as avg_acc, SUM(file_size) as total_size
                     FROM upload_logs''')
        stats = c.fetchone()
    except sqlite3.OperationalError:
        stats = (0, 0, 0)

    try:
        c.execute('''SELECT u.username, COUNT(ul.id) as upload_count, AVG(ul.accuracy) as avg_acc
                     FROM users u
                     LEFT JOIN upload_logs ul ON u.id = ul.user_id
                     GROUP BY u.id''')
        user_stats = c.fetchall()
    except sqlite3.OperationalError:
        user_stats = []

    conn.close()

    return render_template('admin_dashboard.html',
                           admin_name=session['first_name'],
                           users=users,
                           total_uploads=stats[0] or 0,
                           avg_accuracy=round(stats[1] * 100, 2) if stats[1] else 0,
                           total_size=round(stats[2] / (1024 * 1024), 2) if stats[2] else 0,
                           user_stats=user_stats)


@app.route('/user')
def user_dashboard():
    if 'role' not in session or session['role'] != 'user':
        return redirect(url_for('login'))

    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    try:
        c.execute('''SELECT filename, file_size, accuracy, loss, uploaded_at 
                     FROM upload_logs 
                     WHERE user_id = ? 
                     ORDER BY uploaded_at DESC 
                     LIMIT 20''', (session['user_id'],))
        upload_history = c.fetchall()
    except sqlite3.OperationalError:
        upload_history = []

    try:
        c.execute('''SELECT COUNT(*) as total, AVG(accuracy) as avg_acc, SUM(file_size) as total_size
                     FROM upload_logs 
                     WHERE user_id = ?''', (session['user_id'],))
        user_stats = c.fetchone()
    except sqlite3.OperationalError:
        user_stats = (0, 0, 0)

    conn.close()

    return render_template('user_dashboard.html',
                           user_name=session['first_name'],
                           upload_history=upload_history,
                           total_uploads=user_stats[0] or 0,
                           avg_accuracy=round(user_stats[1] * 100, 2) if user_stats[1] else 0,
                           total_size=round(user_stats[2] / (1024 * 1024), 2) if user_stats[2] else 0)


@app.route('/analytics')
def analytics():
    if 'role' not in session:
        return redirect(url_for('login'))

    logger.info(f"Загрузка страницы аналитики для пользователя {session.get('username')}")

    classes, counts = load_class_distribution()

    epochs, val_accuracy, train_accuracy, val_loss, train_loss = load_training_history()

    top5_classes, top5_counts = load_top5_classes()

    if classes is not None:
        logger.info(f"Загружено распределение классов: {len(classes)} классов")
    if epochs is not None:
        logger.info(f"Загружена история обучения: {len(epochs)} эпох")

    return render_template('analytics.html',
                           class_labels=classes or [0, 1, 2, 3, 4],
                           class_counts=counts or [100, 100, 100, 100, 100],
                           epochs=epochs or list(range(1, 11)),
                           val_acc=val_accuracy or [45, 52, 58, 63, 67, 71, 74, 76, 78, 79],
                           train_acc=train_accuracy or [],
                           val_loss=val_loss or [],
                           train_loss=train_loss or [],
                           top5_classes=top5_classes or ['Цивилизация 3', 'Цивилизация 1', 'Цивилизация 5',
                                                         'Цивилизация 2', 'Цивилизация 4'],
                           top5_counts=top5_counts or [45, 38, 30, 28, 22],
                           min_epochs=MIN_EPOCHS,
                           max_epochs=MAX_EPOCHS)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'role' not in session or session['role'] != 'user':
        logger.warning(f"Попытка загрузки файла без прав: {session.get('username')}")
        return redirect(url_for('login'))

    if 'testfile' not in request.files:
        flash('Файл не выбран')
        return redirect(url_for('user_dashboard'))

    file = request.files['testfile']
    if file.filename == '':
        flash('Файл не выбран')
        return redirect(url_for('user_dashboard'))

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0) 

    max_size = 500 * 1024 * 1024
    if file_size > max_size:
        flash(f'Файл слишком большой. Максимальный размер: 500MB. Ваш файл: {round(file_size / (1024 * 1024), 2)}MB')
        logger.warning(f"Попытка загрузки слишком большого файла: {file_size} bytes")
        return redirect(url_for('user_dashboard'))

    if file and file.filename.endswith('.npz'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{session['user_id']}_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

        try:
            chunk_size = 8192 
            with open(filepath, 'wb') as f:
                while True:
                    chunk = file.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)

            logger.info(f"Файл сохранен: {safe_filename} (размер: {round(file_size / (1024 * 1024), 2)}MB)")

            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('''INSERT INTO uploads_tracking (user_id, filename, file_path, status) 
                       VALUES (?, ?, ?, ?)''',
                      (session['user_id'], file.filename, filepath, 'uploaded'))
            conn.commit()

            if not os.path.exists('models/trained_model.keras'):
                flash('Модель не найдена. Сначала запустите create_demo_data.py')
                logger.error("Модель не найдена")
                conn.close()
                return redirect(url_for('user_dashboard'))

            try:
                accuracy, loss = load_model_and_predict(filepath)

                c.execute('''INSERT INTO upload_logs (user_id, filename, file_size, accuracy, loss) 
                           VALUES (?, ?, ?, ?, ?)''',
                          (session['user_id'], file.filename, file_size, accuracy, loss))

                c.execute('''UPDATE uploads_tracking SET status = 'completed' 
                           WHERE file_path = ?''', (filepath,))

                conn.commit()

                flash(f'✅ Тестирование завершено! Точность: {accuracy * 100:.2f}%, Потери: {loss:.4f}')
                logger.info(f"Тестирование успешно: точность {accuracy * 100:.2f}%, потери {loss:.4f}")

            except Exception as e:
                flash(f'❌ Ошибка при обработке файла: {str(e)}')
                logger.error(f"Ошибка обработки файла: {str(e)}")
                c.execute('''UPDATE uploads_tracking SET status = 'error' 
                           WHERE file_path = ?''', (filepath,))
                conn.commit()

            conn.close()

        except Exception as e:
            flash(f'❌ Ошибка при сохранении файла: {str(e)}')
            logger.error(f"Ошибка сохранения файла: {str(e)}")
    else:
        flash('Пожалуйста, загрузите файл с расширением .npz')
        logger.warning(f"Попытка загрузки файла неверного формата: {file.filename}")

    return redirect(url_for('user_dashboard'))


@app.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    if user_id == session['user_id']:
        flash('Нельзя удалить свою учетную запись')
        return redirect(url_for('admin_dashboard'))

    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    try:
        c.execute("DELETE FROM upload_logs WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM uploads_tracking WHERE user_id = ?", (user_id,))
    except sqlite3.OperationalError:
        pass

    c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

    flash('Пользователь удален')
    logger.info(f"Администратор {session['username']} удалил пользователя ID {user_id}")
    return redirect(url_for('admin_dashboard'))


@app.route('/training_log')
def training_log():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    log_content = ""
    if os.path.exists('training.log'):
        with open('training.log', 'r') as f:
            log_content = f.read()

    return render_template('training_log.html', log_content=log_content)


@app.route('/upload_status')
def upload_status():
    if 'role' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect('users.db')
    c = conn.cursor()

    try:
        c.execute('''SELECT ut.filename, ut.status, u.username, ut.uploaded_at
                     FROM uploads_tracking ut
                     JOIN users u ON ut.user_id = u.id
                     ORDER BY ut.uploaded_at DESC
                     LIMIT 50''')
        uploads = c.fetchall()
    except sqlite3.OperationalError:
        uploads = []

    conn.close()

    return render_template('upload_status.html', uploads=uploads)


@app.route('/reset_db')
def reset_db():
    if 'role' not in session or session['role'] != 'admin':
        return redirect(url_for('login'))

    if os.path.exists('users.db'):
        os.remove('users.db')

    init_db()

    flash('База данных сброшена')
    return redirect(url_for('admin_dashboard'))


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)


    app.run(debug=True, host='0.0.0.0', port=5000)
