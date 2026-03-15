import unittest
import os
import tempfile
import sqlite3
import numpy as np
from app import app, init_db
import models_utils


class TestApp(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
        self.app = app.test_client()

        self.db_fd, app.config['DATABASE'] = tempfile.mkstemp()
        with app.app_context():
            init_db()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(app.config['DATABASE'])

    def test_login_page_loads(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Вход в систему', response.data)

    def test_successful_login_admin(self):
        response = self.app.post('/', data={
            'username': 'admin',
            'password': 'admin123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Панель администратора', response.data)

    def test_successful_login_user(self):
        response = self.app.post('/', data={
            'username': 'user',
            'password': 'user123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Панель пользователя', response.data)

    def test_failed_login(self):
        response = self.app.post('/', data={
            'username': 'wrong',
            'password': 'wrong'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Неверное имя пользователя или пароль', response.data)

    def test_logout(self):
        self.app.post('/', data={
            'username': 'user',
            'password': 'user123'
        })
        response = self.app.get('/logout', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Вход в систему', response.data)

    def test_admin_create_user(self):
        self.app.post('/', data={
            'username': 'admin',
            'password': 'admin123'
        })

        response = self.app.post('/admin', data={
            'first_name': 'Тест',
            'last_name': 'Тестов',
            'username': 'testuser',
            'password': 'test123',
            'role': 'user'
        }, follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Пользователь успешно создан', response.data)

        response = self.app.post('/', data={
            'username': 'testuser',
            'password': 'test123'
        }, follow_redirects=True)
        self.assertIn(b'Панель пользователя', response.data)

    def test_admin_delete_user(self):
        self.app.post('/', data={
            'username': 'admin',
            'password': 'admin123'
        })

        response = self.app.post('/delete_user/3', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Пользователь удален', response.data)

        response = self.app.post('/', data={
            'username': 'ivan',
            'password': '123456'
        }, follow_redirects=True)
        self.assertIn(b'Неверное имя пользователя или пароль', response.data)

    def test_unauthorized_admin_access(self):
        self.app.post('/', data={
            'username': 'user',
            'password': 'user123'
        })

        response = self.app.get('/admin', follow_redirects=True)
        self.assertIn(b'Вход в систему', response.data)

    def test_file_upload_no_file(self):
        self.app.post('/', data={
            'username': 'user',
            'password': 'user123'
        })

        response = self.app.post('/upload', data={}, follow_redirects=True)
        self.assertIn(b'Файл не выбран', response.data)

    def test_file_upload_wrong_format(self):
        self.app.post('/', data={
            'username': 'user',
            'password': 'user123'
        })

        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            f.write(b'test content')
            f.seek(0)

            response = self.app.post('/upload', data={
                'testfile': (f, 'test.txt')
            }, follow_redirects=True)

            self.assertIn(b'загрузите файл с расширением .npz', response.data)

    def test_analytics_page_access(self):
        self.app.post('/', data={
            'username': 'user',
            'password': 'user123'
        })

        response = self.app.get('/analytics')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Аналитика данных', response.data)


class TestModelsUtils(unittest.TestCase):

    def setUp(self):
        os.makedirs('models', exist_ok=True)

        self.test_classes = np.array([0, 1, 2, 3])
        self.test_counts = np.array([100, 80, 120, 90])
        np.savez('models/test_class_distribution.npz',
                 classes=self.test_classes,
                 counts=self.test_counts)

        self.test_epochs = 10
        np.savez('models/test_training_history.npz',
                 accuracy=np.random.rand(10),
                 val_accuracy=np.random.rand(10),
                 loss=np.random.rand(10),
                 val_loss=np.random.rand(10))

    def tearDown(self):
        test_files = ['models/test_class_distribution.npz',
                      'models/test_training_history.npz']
        for f in test_files:
            if os.path.exists(f):
                os.remove(f)

    def test_load_class_distribution(self):
        if os.path.exists('models/class_distribution.npz'):
            os.rename('models/class_distribution.npz',
                      'models/class_distribution.npz.bak')

        os.rename('models/test_class_distribution.npz',
                  'models/class_distribution.npz')

        try:
            classes, counts = models_utils.load_class_distribution()
            self.assertIsNotNone(classes)
            self.assertIsNotNone(counts)
            self.assertEqual(len(classes), 4)
            self.assertEqual(len(counts), 4)
        finally:
            os.rename('models/class_distribution.npz',
                      'models/test_class_distribution.npz')
            if os.path.exists('models/class_distribution.npz.bak'):
                os.rename('models/class_distribution.npz.bak',
                          'models/class_distribution.npz')

    def test_load_training_history(self):
        if os.path.exists('models/training_history.npz'):
            os.rename('models/training_history.npz',
                      'models/training_history.npz.bak')

        os.rename('models/test_training_history.npz',
                  'models/training_history.npz')

        try:
            epochs, val_acc, train_acc, val_loss, train_loss = models_utils.load_training_history()
            self.assertIsNotNone(epochs)
            self.assertIsNotNone(val_acc)
            self.assertEqual(len(epochs), 10)
            self.assertEqual(len(val_acc), 10)
        finally:
            os.rename('models/training_history.npz',
                      'models/test_training_history.npz')
            if os.path.exists('models/training_history.npz.bak'):
                os.rename('models/training_history.npz.bak',
                          'models/training_history.npz')


if __name__ == '__main__':
    unittest.main()
