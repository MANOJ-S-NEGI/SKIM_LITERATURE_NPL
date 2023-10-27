import json
import pandas as pd
from spacy.lang.en import English
import tensorflow as tf
import os


class Text_Process:
    try:
        def __init__(self, text):
            self.text = text

        def sentence(self):
            json_file = self.text
            nlp = English()  # setup English sentence parser
            sentencizer = nlp.add_pipe("sentencizer")  # create sentence splitting pipeline object
            doc = nlp(
                json_file[0]["abstract"])  # Create "doc" of parsed sequences, change index for a different abstract
            return doc

        def abstract_lines(self):
            doc = self.sentence()
            abstract = []
            for i in doc.sents:
                i = (str(i))
                abstract.append(i)
            return abstract

        def data_dict(self):
            abstract_line = self.abstract_lines()
            # Get total number of lines
            total_lines_in_sample = len(abstract_line)

            # Go through each line in abstract and create a list of dictionaries containing features for each line
            sample_lines = []
            for i, line in enumerate(abstract_line):
                sample_dict = {"text": str(line), "line_number": i, "total_lines": total_lines_in_sample - 1}
                sample_lines.append(sample_dict)
            return sample_lines

        def num_line_total_line_ohe(self):
            sample_lines = self.data_dict()
            # Get all line_number values from sample abstract
            test_abstract_line_numbers = []
            test_abstract_total_lines = []
            for line in sample_lines:
                test_abstract_line_numbers.append(line["line_number"])
                test_abstract_total_lines.append(line["total_lines"])

            # One-hot encode to same depth as training data, so model accepts right input shape
            test_abstract_line_numbers_one_hot = tf.one_hot(test_abstract_line_numbers, depth=15)
            test_abstract_total_lines_one_hot = tf.one_hot(test_abstract_total_lines, depth=20)

            abstract_line = self.abstract_lines()
            return test_abstract_line_numbers_one_hot, test_abstract_total_lines_one_hot, abstract_line

    except Exception as e:
        raise e


class Split_Text:
    def __init__(self, data):
        self.data = data

    def split_char(self):
        s_chars = []
        for i in self.data:
            i = " ".join(list(i))
            s_chars.append(i)
        return s_chars


def model_path():
    relative_path = "model_dir/skimlit_tribrid_model"
    absolute_path = "D:/msn/pycharm_projects/skimlit/model_dir/skimlit_tribrid_model"

    if os.path.exists(relative_path):
        return relative_path
    elif os.path.exists(absolute_path):
        return absolute_path
    else:
        raise FileNotFoundError("Neither relative nor absolute path exists.")
