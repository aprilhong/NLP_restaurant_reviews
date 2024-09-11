import os
from pathlib import Path
import logging

# For this configuration, whenever you use a logging function (like logging.info()) with a severity level of INFO or higher, it will be recorded with the timestamp and your message within square brackets. 
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "rest_reviews"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/logger.py",
    # f"src/{project_name}/components/__init__.py",
    # f"src/{project_name}/utils/__init__.py",
    # # f"src/{project_name}/utils/common.py",
    # f"src/{project_name}/config/__init__.py",
    # f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    # f"src/{project_name}/entity/__init__.py",
    # f"src/{project_name}/entity/config_entity.py",
    # f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "requirements.txt",
    "setup.py",

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)     #splitting folder and files into different variables

    #if folder not empty, create folder structure
    if filedir !="":    
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    #check if file exists or not before creating the file
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):  
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")