import gradio as gr
import pickle
import pandas as pd
import ast
import numpy as np
import os
import matplotlib.pyplot as plt

# Set the option to opt into future behavior
pd.set_option('future.no_silent_downcasting', True)

# List of options for the dropdown

[("SVM - Jerome Agius", 0), ("Logistic Regression - Isaac Muscat", 1), ("Random Forest - Kyle Demicoli", 2)]

workclass_options = [('State Government', 'State-gov'), 
                    ('Self Employed Not Incorporated', 'Self-emp-not-inc'), 
                    'Private', ('Federal Government', 'Federal-gov'), ('Local Government', 'Local-gov'), ('Self Employed Incorporated', 'Self-emp-inc'), ('Without Pay', 'Without-pay')]

education_option = [('Pre-School', 'Preschool'), '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', ('High School Graduate', 'HS-grad'), ('Collage', 'Some-college'), ('Associate Degree - Vocational', 'Assoc-voc'), ('Associate Degree - Academic', 'Assoc-acdm'), 'Bachelors', 'Masters', ('Professional School', 'Prof-school'), 'Doctorate']

marital_status_option = [('Never Married','Never-married'), ('Married Civilian Spouse', 'Married-civ-spouse'), 'Divorced', 'Separated', ('Married Armed Forces Spouse', 'Married-AF-spouse'), 'Widowed', ('Married Spouse Absent', 'Married-spouse-absent')]
occupation_option = [('Administrative Clerical', 'Adm-clerical'), ('Executive Managerial', 'Exec-managerial'), ('Handlers and Cleaners', 'Handlers-cleaners'), ('Professional Specialty', 'Prof-specialty'), 'Sales', ('Farming and Fishing', 'Farming-fishing'), ('Machine Operator and Inspector', 'Machine-op-inspct'), ('Other Service', 'Other-service'), ('Transport and Moving', 'Transport-moving'), ('Technical Support', 'Tech-support'), ('Craft and Repair', 'Craft-repair'), ('Protective Services', 'Protective-serv'), ('Armed Forces', 'Armed-Forces'), ('Private Household Services' ,'Priv-house-serv')]
relationship_option = [('Not In Family', 'Not-in-family'), 'Husband', 'Wife', ('Biological Child', 'Own-child'), 'Unmarried', ('Other Relative', 'Other-relative')]
race_option = ['White', 'Black', 'Other', ('Asian', 'Asian-Pac-Islander'), ('Indian', 'Amer-Indian-Eskimo')]
sex_option = sorted(['Male', 'Female'])
age = [0, 100]
capital_gain = [0, 99999]
capital_loss = [0, 4356]
hours_per_week = [20, 60]

children_count = [0, 15]
bmi = [10, 100]
region_option = ['southwest', 'southeast', 'northwest', 'northeast']
smoker_option = ['yes', 'no']

# Mapping for education
education_mapping = "{'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16}"
education_dict = ast.literal_eval(education_mapping)

# List of the columns present in dataframe used to train the model
salary_columns = ['age', 'education-num', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'workclass_Local-gov', 'workclass_Private',
        'workclass_Self-emp-inc', 'workclass_Self-emp-not-inc',
        'workclass_State-gov', 'workclass_Without-pay',
        'marital-status_Married-AF-spouse', 'marital-status_Married-civ-spouse',
        'marital-status_Married-spouse-absent', 'marital-status_Never-married',
        'marital-status_Separated', 'marital-status_Widowed',
        'occupation_Armed-Forces', 'occupation_Craft-repair',
        'occupation_Exec-managerial', 'occupation_Farming-fishing',
        'occupation_Handlers-cleaners', 'occupation_Machine-op-inspct',
        'occupation_Other-service', 'occupation_Priv-house-serv',
        'occupation_Prof-specialty', 'occupation_Protective-serv',
        'occupation_Sales', 'occupation_Tech-support',
        'occupation_Transport-moving', 'relationship_Not-in-family',
        'relationship_Other-relative', 'relationship_Own-child',
        'relationship_Unmarried', 'relationship_Wife', 'race_Asian-Pac-Islander',
        'race_Black', 'race_Other', 'race_White']

health_columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']

# Code for SVM
def Salary(model, workclass, education, marital_status, occupation, relationship, race, sex, age, capital_gain, capital_loss, hours_per_week):

    # Set the working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if model == 0:
        model_used = "SVM"
        with open('models/best_svm_OvM_Salary_Classification.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Loading the scaler and transform the data
        with open('models/z-score_scaler_svm_salary_classification.pkl', 'rb') as f:
            scaler = pickle.load(f)
    elif model == 1:
        model_used = "Logistic Regression"
        with open('models/best_lr_Salary_Classification.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Loading the scaler and transform the data
        with open('models/z-score_scaler_lr_salary_classification.pkl', 'rb') as f:
            scaler = pickle.load(f)
    elif model == 2:
        model_used = "Random Forest"
        # Add Random Forest model

    new_data = {
        'age': age,
        'workclass': workclass,
        'education': education,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
    }
    new_data = pd.DataFrame([new_data])
    new_data['education'] = new_data['education'].map(education_dict)
    new_data = new_data.rename(columns={'education': 'education-num'})

    # Create an empty DataFrame with these columns
    formattedDF = pd.DataFrame(columns=salary_columns)

    # Copying over the continuous columns
    formattedDF['age'] = new_data['age']
    formattedDF['education-num'] = new_data['education-num']
    formattedDF['capital-gain'] = new_data['capital-gain']
    formattedDF['capital-loss'] = new_data['capital-loss']
    formattedDF['hours-per-week'] = new_data['hours-per-week']
    formattedDF['workclass_'+new_data['workclass']] = 1 
    formattedDF['marital-status_'+new_data['marital-status']] = 1
    formattedDF['occupation_'+new_data['occupation']] = 1
    formattedDF['relationship_'+new_data['relationship']] = 1
    formattedDF['race_'+new_data['race']] = 1
    formattedDF['sex'] = formattedDF['sex'].apply(lambda x: 1 if x == 'Male' else 0)

    # Fill remaining columns with 0
    formattedDF.fillna(0, inplace=True)
    formattedDF = formattedDF.astype(int)
    formattedDF = formattedDF[formattedDF.columns.intersection(salary_columns)]

    # Assuming 'high_skew_columns' from training is a list of columns with high skewness
    for column in  ['capital-gain', 'capital-loss']:
        formattedDF[column] = np.log1p(formattedDF[column])

    # Apply the scaler to the unseen data
    continuous_columns = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    formattedDF[continuous_columns] = scaler.transform(formattedDF[continuous_columns])

    # Make predictions with the loaded model
    prediction = loaded_model.predict(formattedDF)

    probability = loaded_model.predict_proba(formattedDF)

    # Get the number of classes
    num_classes = probability.shape[1]

    class_dict = {
        0: '<=50K',
        1: '>50K'
    }

    # Select the probabilities for a single sample (e.g., the first sample)
    probabilities = probability[0] 

    class_labels = [class_dict[i] for i in range(num_classes)]
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))  # Use a colormap for consistent colors

    fig, ax = plt.subplots(figsize=(10, 10))
    _, _, autotexts = ax.pie(probabilities, colors=colors, autopct='%1.1f%%', startangle=140, pctdistance=1.1)

    # Create a legend with colored boxes
    legend_elements = []
    for i, (color, label) in enumerate(zip(colors, class_labels)):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=label))

    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_title("Predicted Class Probabilities")

    for i, p in enumerate(probabilities):
      prob = float(round(p*100, 2))
      if prob > 0:
          autotexts[i].set_text(f"{prob}%")
      else:
          autotexts[i].set_text('')

    salary_result = '<=50K' if prediction[0] == 0 else '>50K'

    return f"Predicted using {model_used} Salary Class: {salary_result}", fig

def Health(model, age, sex, bmi, children, smoker, region):

    # Set the working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if model == 0:
        model_used = "SVM"
        with open('models/best_health_svm_OvM_Charges_Classification.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Loading the scaler and transform the data
        with open('models/z-score_scaler_svm_charges_classification.pkl', 'rb') as f:
            scaler = pickle.load(f)
    elif model == 1:
        model_used = "Logistic Regression"
        with open('models/best_health_lr_Charges_Classification.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Loading the scaler and transform the data
        with open('models/z-score_scaler_lr_charges_classification.pkl', 'rb') as f:
            scaler = pickle.load(f)
    elif model == 2:
        model_used = "Random Forest"
        # Add Random Forest model

    #Inverting the dict to map the 'charges' values back to 'charges' labels
    inverse_mapping_charges = {
        0: 'Very Low (<= 5000)',
        1: 'Low (5001 - 10000)',
        2: 'Moderate (10001 - 15000)',
        3: 'High (15001 - 20000)',
        4: 'Very High (> 20001)',
    }

    new_data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region,
    }
    
    new_data = pd.DataFrame([new_data])

    # Create an empty DataFrame with these columns
    formattedDF = pd.DataFrame(columns=health_columns)

    # Copying over the continuous columns
    formattedDF['age'] = new_data['age']
    formattedDF['sex'] = new_data['sex'].apply(lambda x: 1 if x == 'Male' else 0)
    formattedDF['bmi'] = new_data['bmi']
    formattedDF['children'] = new_data['children']
    formattedDF['smoker'] = new_data['smoker'].apply(lambda x: 1 if x == 'Yes' else 0)
    formattedDF['region_'+new_data['region']] = 1

    # Fill remaining columns with 0
    formattedDF.fillna(0, inplace=True)
    formattedDF = formattedDF.astype(int)
    formattedDF = formattedDF[formattedDF.columns.intersection(health_columns)]

    # Apply the scaler to the unseen data
    continuous_columns = ['age', 'bmi']
    formattedDF[continuous_columns] = scaler.transform(formattedDF[continuous_columns])

    # Make predictions with the loaded model
    prediction = loaded_model.predict(formattedDF)[0]
    prediction = inverse_mapping_charges[prediction]

    probability = loaded_model.predict_proba(formattedDF)   
    
    # Get the number of classes
    num_classes = probability.shape[1]

    class_dict = {
        0: 'Very Low (<= 5000)',
        1: 'Low (5001 - 10000)',
        2: 'Moderate (10001 - 15000)',
        3: 'High (15001 - 20000)',
        4: 'Very High (> 20001)',
    }

    # Select the probabilities for a single sample (e.g., the first sample)
    probabilities = probability[0] 

    class_labels = [class_dict[i] for i in range(num_classes)]
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))  # Use a colormap for consistent colors

    fig, ax = plt.subplots(figsize=(10, 10))
    _, _, autotexts = ax.pie(probabilities, colors=colors, autopct='%1.1f%%', startangle=140, pctdistance=1.1)

    # Create a legend with colored boxes
    legend_elements = []
    for i, (color, label) in enumerate(zip(colors, class_labels)):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=label))

    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_title("Predicted Class Probabilities")

    for i, p in enumerate(probabilities):
      prob = float(round(p*100, 2))
      if prob > 0:
          autotexts[i].set_text(f"{prob}%")
      else:
          autotexts[i].set_text('')

    return f"Predicted using {model_used} Charges Class: {prediction}", fig

# interface one
iface1 = gr.Interface(
    fn=Salary,
    inputs=[
        gr.Dropdown(choices=[("SVM - Jerome Agius", 0), ("Logistic Regression - Isaac Muscat", 1), ("Random Forest - Kyle Demicoli", 2)], label="Model", value=0),
        gr.Dropdown(choices=workclass_options, label="Workclass"),
        gr.Dropdown(choices=education_option, label="Education"),
        gr.Dropdown(choices=marital_status_option, label="Marital Status"),
        gr.Dropdown(choices=occupation_option, label="Occupation"),
        gr.Dropdown(choices=relationship_option, label="Relationship"),
        gr.Dropdown(choices=race_option, label="Race"),
        gr.Dropdown(choices=sex_option, label="Sex"),
        gr.Slider(minimum=age[0], maximum=age[1], step=1, label="Age"),
        gr.Slider(minimum=capital_gain[0], maximum=capital_gain[1], step=1, label="Capital Gain"),
        gr.Slider(minimum=capital_loss[0], maximum=capital_loss[1], step=1, label="Capital Loss"),
        gr.Slider(minimum=hours_per_week[0], maximum=hours_per_week[1], step=1, label="Hours per Week"),
    ],
    outputs=[gr.Text(label="Predicted Label"), gr.Plot(label="Predicted Class Probabilities")],
    title="SVM - Salary",
    flagging_mode="never"
)

# interface two
iface2 = gr.Interface(
    fn=Health,
    inputs=[
        gr.Dropdown(choices=[("SVM - Jerome Agius", 0), ("Logistic Regression - Isaac Muscat", 1), ("Random Forest - Kyle Demicoli", 2)], label="Model", value=0),
        gr.Slider(minimum=age[0], maximum=age[1], step=1, label="Age"),
        gr.Dropdown(choices=sex_option, label="Sex"),
        gr.Slider(minimum=bmi[0], maximum=bmi[1], step=0.1, label="BMI"),
        gr.Slider(minimum=children_count[0], maximum=children_count[1], step=1, label="Children"),
        gr.Dropdown(choices=smoker_option, label="Smoker"),
        gr.Dropdown(choices=region_option, label="Region"),
    ],
    outputs=[gr.Text(label="Predicted Label"), gr.Plot(label="Predicted Class Probabilities")],
    title="SVM - Health",
    flagging_mode="never"
)

demo = gr.TabbedInterface([iface1, iface2], ["Salary Prediction", "Health Charges Prediction"])

# Run the interface
demo.launch(share=True)