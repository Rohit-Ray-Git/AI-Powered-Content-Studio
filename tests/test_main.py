import unittest
from main import run_pipeline

class TestRunPipelineInputValidation(unittest.TestCase):
    def test_empty_topic(self):
        result = run_pipeline('', content_type='Blog Post')
        self.assertEqual(result, (None, None))

    def test_too_long_topic(self):
        long_topic = 'a' * 201
        result = run_pipeline(long_topic, content_type='Blog Post')
        self.assertEqual(result, (None, None))

    def test_invalid_content_type(self):
        result = run_pipeline('Test topic', content_type='InvalidType')
        self.assertEqual(result, (None, None))

    def test_invalid_language(self):
        result = run_pipeline('Test topic', language='Klingon')
        self.assertEqual(result, (None, None))

    def test_invalid_tone(self):
        result = run_pipeline('Test topic', tone='Sarcastic')
        self.assertEqual(result, (None, None))

    def test_invalid_script_length(self):
        result = run_pipeline('Test topic', content_type='Video/Podcast Script', script_length=0)
        self.assertEqual(result, (None, None))
        result = run_pipeline('Test topic', content_type='Video/Podcast Script', script_length=61)
        self.assertEqual(result, (None, None))
        result = run_pipeline('Test topic', content_type='Video/Podcast Script', script_length='five')
        self.assertEqual(result, (None, None))

    # def test_valid_inputs(self):
    #     # This test only checks that valid input does not return (None, None) due to validation
    #     # It may still return (None, None) if the pipeline fails for other reasons
    #     result = run_pipeline('Test topic', content_type='Blog Post', language='English', tone='Informational')
    #     self.assertNotEqual(result, (None, None))

if __name__ == '__main__':
    unittest.main() 