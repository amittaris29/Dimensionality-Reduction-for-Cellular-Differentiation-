import pandas as pd

def data_selector_for_bd(num_of_data, df):
    # Ensure only gene expression columns are kept
    gene_columns = ['TBX6', 'BRA', 'CDX2', 'SOX2', 'SOX1', 'timepoint']
    selected = df[gene_columns].copy()

    # Shuffle the full dataset to ensure randomness
    selected = selected.sample(frac=1, random_state=42).reset_index(drop=True)

    # Filter data by each timepoint
    timepoints = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    balanced_data = []

    for tp in timepoints:
        group = selected[selected['timepoint'] == tp]
        reduced_group = group.iloc[:num_of_data, :]
        balanced_data.append(reduced_group)

    # Combine all equally sampled timepoint groups
    combined_equal_data = pd.concat(balanced_data, ignore_index=True)

    # Final output with only the gene expression columns (drop timepoint)
    final_selected_data = combined_equal_data.drop(columns='timepoint')

    return final_selected_data, combined_equal_data
