from components.function import *
import tensorflow as tf


# called function num_line_total_line_ohe
def training_prediction(text):
    try:
        file = text
        calling_processed_data = Text_Process(file).num_line_total_line_ohe()
        test_abstract_line_numbers_one_hot, test_abstract_total_lines_one_hot, abstract_lines = calling_processed_data

        # splitting text into words and joining them:
        abstract_chars = Split_Text(abstract_lines).split_char()

        # Load the model
        loaded_model = tf.keras.models.load_model(model_path())

        # Make predictions on sample abstract features
        test_abstract_pred_probs = loaded_model.predict(x=(test_abstract_line_numbers_one_hot,
                                                           test_abstract_total_lines_one_hot,
                                                           tf.constant(abstract_lines),
                                                           tf.constant(abstract_chars)))

        # Turn prediction probabilities into prediction classes
        preds = tf.argmax(test_abstract_pred_probs, axis=1)
        return preds, abstract_lines

    except Exception as e:
        raise e


"""
def read_file():
    with open("D:\msn\pycharm_projects\skimlit\skimlit_example_abstracts.json", "r") as f:
        example_abstracts = json.load(f)
    return example_abstracts


text = read_file()
print(training_prediction(text))
"""
