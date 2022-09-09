# Intent chatbot training example for library application
Create the chatbot model using pytorch to use it in the [api-server](https://github.com/mal2/python-chatbot-api)
## Setup
Creating a virtual environment and activate it:
```
virtualenv -p python3.9 venv
```
Activate for linux
```
source venv/bin/activate
```
or for Windows:
```
.\venv\Scripts\activate
```
Then install the required python librarys using:
```
pip install -r requirements.txt
```
## Usage
Just run
```
python train.py
```
to train the model with the default parameters.