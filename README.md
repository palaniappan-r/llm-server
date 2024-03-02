A basic API serving LangChain chains using the Python `http-lib`

## Initial Setup

Install everything in the `requirements.txt` file into a `conda` environment and activate the environment

## Downloading LLMs

Download GPTQ models and place the in the `/models` directory. Appropriately reference these folders when loading the LLM in `main.py`.

## Using the API

The API can keep seperately keep track of multiple conversations and their corresponding history, in parallel. Each conversation is identified by the user, and their respective passwords in the `userlist.txt` file.

Will add API docs shortly.