# Prerequisites

We recommend creating a new virtual environment for this project. You can do this by using your preferred method. Like virtualenv: `virtualenv venv` and then `source venv/bin/activate`.

Install the requirements: `pip install -r requirements.txt`. The listed torch version is for cpu only, which is best for just running the test.

# Running the code

To run the final agent and see its performance, run `python main.py --test-file PATH_TO_FILE`. The file should be an excel file with the same format as the train data.