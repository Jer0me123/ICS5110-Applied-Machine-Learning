# ICS5110 Applied Machine Learning

This repository contains the implementation of the Support Vector Machine (SVM), Logistic Regression & Random Forest algorithms trained on the UCI-Adult Dataset to predict the household Salary and the Healthcare-Insurance dataset to predict the Charges.

These models can be tested via this link: https://jer0me123.github.io/ICS5110-Applied-Machine-Learning/

Repository Contents:

* **Gradio/** - This directory contains the **app.py** file encapsulating the gradio implementation alongside the **models** sub-directory containing the SVM, LR & RF models used in the gradio UI.
* **LR/** - This directory contains three files **LogisticRegression - Salary / Medical / Education** each with an LR implementation predicting the Salary, Charges and Education level labels.
* **SVM/** - This directory contains three files **SVM - Salary / Medical / Education** each with an SVM implementation predicting the Salary, Charges and Education level labels.
* **RF/** - This directory contains three files **RF - Salary / Medical / Education** each with an RF implementation predicting the Salary, Charges and Education level labels.
* **Testing/** - This directory contains extra files used for testing, those of note being the **UCI_adult.ipynb** & **Healthcare_Insurance.ipynb** as these contain the data exploration carried out prior to the model implementation.
