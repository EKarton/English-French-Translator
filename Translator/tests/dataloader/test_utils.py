import unittest
from parameterized import parameterized

import sys
sys.path.append("../src/dataloader")

from utils import get_tokens_from_line, get_spacy_instance

class Get_Tokens_From_Line_Test(unittest.TestCase):

    def test_get_tokens_from_line_should_make_lowercase(self):
        input_str = "I AM HERE nOWww! :)"
        expected_tokens = ['i', 'am', 'here', 'nowww']

        spacy_instance = get_spacy_instance('en')
        self.assertEquals(get_tokens_from_line(input_str, spacy_instance), expected_tokens)

    def test_get_tokens_from_line_should_omit_punctuations_1(self):
        input_str = '(In the meantime), I should like to observe!!'
        expected_tokens = ['in', 'the', 'meantime', 'i', 'should', 'like', 'to', 'observe']

        spacy_instance = get_spacy_instance('en')
        self.assertEquals(get_tokens_from_line(input_str, spacy_instance), expected_tokens)

    def test_get_tokens_from_line_should_omit_punctuations_2(self):
        input_str = 'Ms. Marlene Jennings (Notre-Dame-de-Grâce-Lachine, Lib.):'
        expected_tokens = ['ms', 'marlene', 'jennings', 'notre', 'dame', 'de', 'grâce', 'lachine', 'lib']

        spacy_instance = get_spacy_instance('en')
        self.assertEquals(get_tokens_from_line(input_str, spacy_instance), expected_tokens)

    def test_get_tokens_from_line_should_not_omit_appostope(self):
        input_str = 'I\'m enjoying my brother\'s dinner'
        expected_tokens = ['i', '\'m', 'enjoying', 'my', 'brother', '\'s', 'dinner']

        spacy_instance = get_spacy_instance('en')
        self.assertEquals(get_tokens_from_line(input_str, spacy_instance), expected_tokens)

    def test_get_tokens_from_line_should_omit_numbers_1(self):
        input_str = 'I like my $300 200-PO jacket'
        expected_tokens = ['i', 'like', 'my', 'jacket']

        spacy_instance = get_spacy_instance('en')
        self.assertEqual(get_tokens_from_line(input_str, spacy_instance), expected_tokens)

    def test_get_tokens_from_line_should_omit_numbers_2(self):
        input_str = 'Bob owes me $200 dollars full of 100items'
        expected_tokens = ['bob', 'owes', 'me', 'dollars', 'full', 'of']

        spacy_instance = get_spacy_instance('en')
        self.assertEqual(get_tokens_from_line(input_str, spacy_instance), expected_tokens)

    def test_get_tokens_from_line_should_break_contractions(self):
        input_str = 'He\'ll explain that I won\'t break the shell?'
        expected_tokens = ['he', '\'ll', 'explain', 'that', 'i', 'wo', 'n\'t', 'break', 'the', 'shell']

        spacy_instance = get_spacy_instance('en')
        self.assertEqual(get_tokens_from_line(input_str, spacy_instance), expected_tokens)

    def test_get_tokens_from_line_should_break_contractions_correctly(self):
        input_str = 'I should like to observe a minute\' s silence'
        expected_tokens = ['i', 'should', 'like', 'to', 'observe', 'a', 'minute', "'s", 'silence']

        spacy_instance = get_spacy_instance('en')
        self.assertEqual(get_tokens_from_line(input_str, spacy_instance), expected_tokens)

    def test_get_tokens_from_line_should_break_contractions_correctly_2(self):
        input_str = 'didn\'t tomorrow\' s lesson cancel?'
        expected_tokens = ['did', 'n\'t', 'tomorrow', '\'s', 'lesson', 'cancel']

        spacy_instance = get_spacy_instance('en')
        self.assertEqual(get_tokens_from_line(input_str, spacy_instance), expected_tokens)

    def test_get_tokens_from_line_should_break_contractions_correctly_3(self):
        input_str = "de l' avis de l' AFET, dont j' ai été le rapporteur"
        expected_tokens = ['de', 'l\'', 'avis', 'de', 'l\'', 'afet', 'dont', 'j\'', 'ai', 'été', 'le', 'rapporteur']

        spacy_instance = get_spacy_instance('fr')
        self.assertEqual(get_tokens_from_line(input_str, spacy_instance), expected_tokens)

    def test_get_tokens_from_line_should_not_break_contractions_correctly_4(self):
        input_str = 'citizens\' representatives'
        expected_tokens = ['citizens', 'representatives']

        spacy_instance = get_spacy_instance('en')
        self.assertEqual(get_tokens_from_line(input_str, spacy_instance), expected_tokens)

    def test_get_tokens_from_line_should_omit_nums(self):
        input_str = 'Cohesion Fund: 2000-2006 [COM(1999)344 - C5-0122/1999 - 1999/2127(COS)]'
        expected_tokens = ['cohesion', 'fund']

        spacy_instance = get_spacy_instance('en')
        self.assertEqual(get_tokens_from_line(input_str, spacy_instance), expected_tokens)
        
if __name__ == '__main__':
    unittest.main()