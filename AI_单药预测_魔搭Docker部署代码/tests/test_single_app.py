from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app


class SingleUiTests(unittest.TestCase):
    def test_rows_exactly_seven(self):
        self.assertEqual(len(app._blank_rows()),7)

    def test_blank_smiles(self):
        rows,status,_=app.predict_single('')
        self.assertEqual(len(rows),7)
        self.assertIn('请输入',status)

    def test_reject_multiline(self):
        rows,status,_=app.predict_single('CCO\nCCC')
        self.assertIn('只输入一个',status)
        self.assertTrue(all(x[1]=='—' for x in rows))

    def test_reject_long_smiles(self):
        rows,status,_=app.predict_single('C'*501)
        self.assertIn('500',status)
        self.assertTrue(all(x[1]=='—' for x in rows))

    def test_missing_environments_never_fabricate(self):
        state={name:{'ok':False,'status':'missing_interpreter'} for name in app.EXPECTED}
        with patch.object(app,'verify_models',return_value=state):
            rows,status,_=app.predict_single('CCO')
        self.assertIn('0/7',status)
        self.assertTrue(all(row[1]=='—' for row in rows))
        self.assertIn('缺少 IDR',rows[-1][-1])

    def test_real_worker_result_flow_mock_only(self):
        states={name:{'ok':False,'status':'missing_interpreter'} for name in app.EXPECTED}
        states['logs']={'ok':True,'executable':'/bin/python'}
        answer={'value':-2.1,'metadata':{'model_version':'test-checkpoint','model_source':'test-weight'}}
        with patch.object(app,'verify_models',return_value=states), patch.object(app,'_run_one',return_value=answer):
            rows,status,details=app.predict_single('CCO')
        self.assertIn('1/7',status)
        self.assertEqual(rows[0][3],'预测成功')
        self.assertEqual(rows[1][1],'—')
        self.assertEqual(rows[-1][1],'—')
        self.assertIn('test-checkpoint',details)

    def test_bad_worker_cannot_be_marked_success(self):
        states={name:{'ok':False,'status':'missing_interpreter'} for name in app.EXPECTED}
        states['logs']={'ok':True,'executable':'/bin/python'}
        with patch.object(app,'verify_models',return_value=states), patch.object(app,'_run_one',side_effect=RuntimeError('bad')):
            rows,status,_=app.predict_single('CCO')
        self.assertEqual(rows[0][3],'实际预测失败')
        self.assertIn('0/7',status)

    def test_build_gradio_interface(self):
        demo=app.build_demo()
        self.assertIsNotNone(demo)


if __name__=='__main__':
    unittest.main()
