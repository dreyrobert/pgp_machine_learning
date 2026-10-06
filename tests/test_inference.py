import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.inference import ROOT, load_model, load_history, predict
from src.prediction_features import available_months, build_features
from src.train_export_model import load_snapshot, add_temporal_features


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_model()
        cls.metadata = cls.payload['metadata']
        cls.history = load_history(cls.metadata)
        cls.city = cls.history.municipio.iloc[0]

    def test_features_match_training(self):
        baseline = add_temporal_features(load_snapshot(ROOT, pd.Timestamp(self.metadata['cutoff_month'])))
        rows = baseline.loc[baseline.data_referencia >= available_months(self.history)[0]]
        actual = pd.concat([build_features(self.history, row.municipio, row.data_referencia, self.metadata) for row in rows.itertuples()], ignore_index=True)
        expected = rows[self.payload['feature_columns']].reset_index(drop=True)
        np.testing.assert_allclose(actual[self.payload['numeric_features']], expected[self.payload['numeric_features']], atol=1e-12)
        self.assertEqual(actual.municipio.tolist(), expected.municipio.tolist())

    def test_future_and_direct_prediction(self):
        month = available_months(self.history)[-1]
        self.assertEqual(month, pd.Timestamp(self.metadata['cutoff_month']) + pd.DateOffset(months=1))
        features = build_features(self.history, self.city, month, self.metadata)
        self.assertAlmostEqual(predict(self.payload, features), self.payload['model'].predict(features)[0])

    def test_no_current_or_future_leakage(self):
        month = pd.Timestamp('2025-01-01')
        original = build_features(self.history, self.city, month, self.metadata)
        changed = self.history.copy()
        changed.loc[changed.data_referencia >= month, 'qtd_acidentes'] = 999999
        pd.testing.assert_frame_equal(original, build_features(changed, self.city, month, self.metadata))

    def test_invalid_inputs(self):
        for city, month in [('INEXISTENTE', '2025-01'), (self.city, '2030-01'), (self.city, '2022-02')]:
            with self.assertRaises(ValueError):
                build_features(self.history, city, month, self.metadata)
        incomplete = self.history.drop(self.history.loc[self.history.municipio == self.city].index[0])
        with self.assertRaises(ValueError):
            build_features(incomplete, self.city, '2025-01', self.metadata)
        with patch('src.inference.MODEL_PATH', ROOT / 'models/ausente.joblib'):
            with self.assertRaises(FileNotFoundError):
                load_model()

    def test_streamlit_submission_and_change(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox), 2)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn('result', app.session_state)
        self.assertTrue(app.metric)
        city = app.selectbox[0].options[-1]
        app.selectbox[0].select(city)
        app.selectbox[1].select(pd.Timestamp('2025-01-01'))
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['result'][0], city)
        self.assertEqual(app.session_state['result'][1], pd.Timestamp('2025-01-01'))


if __name__ == '__main__':
    unittest.main()
