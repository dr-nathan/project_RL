# Prerequisites

We recommend creating a new virtual environment for this project with Python version > 3.9. This project has some type annotations that are not supported < 3.9.

If you use conda, you can create a new environment with `conda create -n env_name python=3.10`. Then activate the environment with `conda activate env_name`.

If you have the right version of Python installed you can create an environment by using your preferred method. Like virtualenv with `virtualenv venv` and activate it using `source venv/bin/activate`.

Install the requirements: `pip install -r requirements.txt`. The listed torch version is for cpu only, which is best for just running the test.

# Running the code

To run the final agent and see its performance, run `python main.py --test_file PATH_TO_FILE`. The file should be an excel file with the same format as the train data.