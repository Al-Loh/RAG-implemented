import pandas as pd


# ABOVE ARE THE MAIN IMPORTS SPECIFIC TO THIS SCRIPT.


# THE PREPROCESS FUNCTION TAKES IN JSON AND TRANSFORMS IT.
# IT, ALSO DOES ALL THE NECESSARY CLEANING STEPS IF NEEDED.
def wrangle(df_temp=pd.DataFrame()):
    temp = []
    questions = []
    answers = []
    for column in df_temp.columns:
        temp.append(df_temp[column].dropna())
    for series in temp:
        for question in series.index:
            questions.append(question)
    for series in temp:
        for answer in series:
            answers.append(answer)
    data_dict = {"questions": questions, "answers": answers}
    df_2 = pd.DataFrame(data_dict)
    return df_2


# WE READ THE JSON FILE AND SEND TO THE PREPROCESS FUNCTION.
df = pd.read_json("faqs.json")
data = wrangle(df)
data.to_csv('data-sources/processed.csv', index=False)
