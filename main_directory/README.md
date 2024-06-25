# Group_15
2612 Advanced Programming for Data Science

Emails:
* Justin Sams (59279): 59279@novasbe.pt
* Tim Gunkel (60161): 60161@novasbe.pt
* Benedikt Tremmel (60253): 60253@novasbe.pt
* Gilian Wagner (58029): 58029@novasbe.pt


# Project Icaras - Group 15

## Table of Contents
1. Project Description
2. How to Install and Run the Project
3. Remarks
4. Additional information

## 1. Project Description
The goal of the project is to analyze Commercial Airflight data for a sustainability study.

## 2. How to Install and Run the Project
This section provides a clear and comprehensive guide on how to obtain a copy of the project for local development and testing.

### Prerequisites

Before you begin, ensure you have the following prerequisites installed on your system:
- Python 3
- conda or Miniconda

These are essential for creating and managing the project environment using the provided `adpro.yml` file.

### Installation Steps

Copy and paste the commands after $ into your terminal (Linux/ Mac OS) or the Anaconda Prompt (Windows)

1. Clone the Repository

        $ git clone https://github.com/JustinSms/Group_15.git
        $ cd main_directory

2. Create and activate the Conda environment


CUse the provided adpro.yml file to create an environment with all necessary dependencies:

        $ conda env create -f adpro.yml

Wait for Conda to set up the environment. Once it's done, you can activate it with:

        $ conda activate adpro

With the environment active, all project dependencies are available for use.

### Run the Project

After installation, launch the Jupyter notebook interface to view and run the project notebooks:

        jupyter notebook

Navigate to the showcase.ipynb notebook within the Jupyter interface to view the project's main presentation and analysis.

## 3. Remarks

Setting the PEP 8 compliance threshold of pylint to 8 is a practical compromise that balances code quality with developer efficiency. It enforces adherence to Python's style guidelines while allowing flexibility for scenarios where perfect compliance is not feasible or necessary. This approach ensures code remains readable and maintainable without making linting a bottleneck in development processes. It promotes a culture of quality, while also accommodating the practical challenges of software development.

## 4. Additional information

Built With:
- Python - The programming language used.
- Conda - Dependency Management.

Authors: Justin Sams, Tim Gunkel, Gilian Wagner, Benedikt Tremmel

License
This project is licensed under the MIT License - see the LICENSE.md file for details