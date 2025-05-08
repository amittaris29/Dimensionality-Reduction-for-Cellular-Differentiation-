import pandas as pd

def data_selector(num_of_data):
    # Load CSV into a DataFrame
    data = pd.read_csv('df_main.csv')
    #Remove the unwanted columns
    numerical_data = data.drop(data.columns[[0,6,8]], axis=1) 


    #Ensure only numerical data is used
    numerical_data = numerical_data.select_dtypes(include=['float64', 'int64'])

    #Shuffle the rows randomly
    selected_data = numerical_data.sample(frac=1, random_state=42).reset_index(drop=True)


    #Separate timepoints 

    D2_data = selected_data[selected_data.iloc[:, 5].isin([2.0])]
    D2_5_data = selected_data[selected_data.iloc[:, 5].isin([2.5])]
    D3_data = selected_data[selected_data.iloc[:, 5].isin([3.0])]
    D3_5_data = selected_data[selected_data.iloc[:, 5].isin([3.5])]
    D4_data = selected_data[selected_data.iloc[:, 5].isin([4])]
    D4_5_data = selected_data[selected_data.iloc[:, 5].isin([4.5])]
    D5_data = selected_data[selected_data.iloc[:, 5].isin([5.0])]

    #Choose equal amount from each 
    desirable_amount_of_data= num_of_data

    reduced_D2= D2_data.iloc[:desirable_amount_of_data, :] 
    reduced_D2_5= D2_5_data.iloc[:desirable_amount_of_data, :] 
    reduced_D3= D3_data.iloc[:desirable_amount_of_data, :] 
    reduced_D3_5= D3_5_data.iloc[:desirable_amount_of_data, :] 
    reduced_D4= D4_data.iloc[:desirable_amount_of_data, :] 
    reduced_D4_5= D4_5_data.iloc[:desirable_amount_of_data, :] 
    reduced_D5= D5_data.iloc[:desirable_amount_of_data, :] 


    # Combine all reduced DataFrames
    combined_equal_data = pd.concat([
        reduced_D2, 
        reduced_D2_5, 
        reduced_D3, 
        reduced_D3_5, 
        reduced_D4, 
        reduced_D4_5, 
        reduced_D5
    ], ignore_index=True)



    #Selected data with not equal proportions in respect to timepoints
    final_selected_data = combined_equal_data.drop(data.columns[7], axis=1) 

    #DO phate to whole data set and then just color 
    return final_selected_data, combined_equal_data