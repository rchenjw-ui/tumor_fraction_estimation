import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.pandas2ri as pandas2ri
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt 
from tqdm.contrib import tzip
import seaborn as sns
 
def rdata_to_dataframes_with_names(rdata_file):
    """
    Load RData file and convert R matrices/data frames to pandas DataFrames with proper index and columns.
    Returns a dict of variable name -> pandas DataFrame or original R object.
    """
    robjects.r['load'](rdata_file)
    var_names = list(robjects.r('ls()'))
    result = {}
    with localconverter(robjects.default_converter + pandas2ri.converter):
        for varname in var_names:
            r_obj = robjects.r[varname]
            try:
                # Try to get row and column names
                row_names = list(robjects.r(f'rownames({varname})'))
                col_names = list(robjects.r(f'colnames({varname})'))
                # Convert to DataFrame with assigned labels
                df = pd.DataFrame(robjects.conversion.rpy2py(r_obj), index=row_names, columns=col_names)
                result[varname] = df
            except Exception:
                # Fallback - return raw object if conversion fails
                result[varname] = r_obj
    return result


def extract_beta_columns(df, ref_probes):
    """
    Extract columns ending with '_AVG_Beta' from df, filter rows to ref_probes, 
    and organize as probes x samples DataFrame.
    
    Args:
        df (pd.DataFrame): Full dataframe loaded from RData.
        ref_probes (list or set): Probe IDs to keep.
        
    Returns:
        pd.DataFrame: probes x samples beta values.
    """
    # Select beta columns
    beta_columns = [c for c in df.columns if c.endswith('_AVG_Beta')]
    beta_df = df[beta_columns].copy()
    
    # Rename columns: remove suffix '_AVG_Beta'
    beta_df.columns = [c.replace('_AVG_Beta', '') for c in beta_df.columns]
    
    # Filter rows by probe IDs (assuming probe IDs are in index)
    filtered_beta_df = beta_df.loc[beta_df.index.intersection(ref_probes)]
    
    return filtered_beta_df


def beta_df_to_sample_probe_dict(beta_df):
    """
    Convert a DataFrame of beta values (probes x samples) into nested dictionary format.
    Args:
        beta_df (pd.DataFrame): DataFrame indexed by probes, columns are sample IDs.
    Returns:
        dict: {sample: {probe: beta_value}}
    """
    sample_dict = {}
    for sample in beta_df.columns:
        sample_dict[sample] = beta_df[sample].dropna().to_dict()
    return sample_dict




#gets beta values for embryo

def beta_values_embryonic(ref_probes, paths = ('embryo_data/GSE57362_Embryo_retinas_meth_signal_2.txt', 'embryo_data/GSE57362_Embryo_retinas_unmeth_signal_2.txt')):
    with open(paths[0], 'r') as f:
        meth_data = f.read().strip().split('\n')
    with open(paths[1], 'r') as f:
        unmeth_data = f.read().strip().split('\n')
    sample_ids = meth_data[0] 
    beta_values_embryo = {sample_id: {} for sample_id in sample_ids.split('\t')}  
    for line, line1 in tzip(meth_data[1:], unmeth_data[1:]):
        probe = line.split('\t')[0]
        if probe in ref_probes:
            data1 = line.split('\t')[1:]
            data2 = line1.split('\t')[1:]
            for sample_id, value1, value2 in zip(beta_values_embryo.keys(), data1, data2):
                beta = float(value1) / (float(value1) + float(value2)) if (float(value1) + float(value2)) != 0 else float('nan')
                beta_values_embryo[sample_id][probe] = beta
    return beta_values_embryo

#gets data for tfx

def beta_values_rb(ref_probes, path = 'tumor_data/GSE58783_Methylation_processed_data_2.RData'):
    rdata_dict = rdata_to_dataframes_with_names(path)
    beta_df = rdata_dict.get('processed_data')
    filtered_beta_df = extract_beta_columns(beta_df, ref_probes)
    beta_values_rb = beta_df_to_sample_probe_dict(filtered_beta_df)
    return beta_values_rb

#convert both into np 

def generate_ref_probes(name):

    if name == "probes":
        return ['cg10507988', 'cg08447324', 'cg23748923', 'cg15384383', 'cg02408775', 'cg15320905', 'cg12621203', 'cg15258447', 'cg06038470', 'cg26678920', 'cg27298164', 'cg13118906', 'cg21042627', 'cg02809746', 'cg26188698', 'cg26966245', 'cg02132702', 'cg14769786', 'cg00766482', 'cg19509829', 'cg13276570', 'cg16520288', 'cg27493151', 'cg27041381', 'cg08502239', 'cg05542681', 'cg16493531', 'cg11802666', 'cg26228266', 'cg08384314', 'cg17029019', 'cg27201775', 'cg20814095', 'cg16094026', 'cg21615583']
    path = name+"_probe_beta_pairs.csv"
    df = pd.read_csv(path, usecols=['probe'])
    column_list = df["probe"].tolist()
    df = pd.read_csv("pilot_8samples_merged.csv", usecols = ["ProbeID"])
    probes = df["ProbeID"].tolist()
    column_list = [x for x in column_list if x in probes]
    return column_list

def plot_probe_betas_heatmap(probe_list, rb_beta_df, retina_beta_df, filename="top_bottom_heatmap.jpg"):
   """
   Plot a heatmap for the provided probes showing beta values for RB (left) and WT (right) samples.
   Args:
       probe_list: list of probe IDs (order will be row order)
       rb_beta_df: DataFrame, rows=probe, cols=RB samples
       retina_beta_df: DataFrame, rows=probe, cols=WT/retina samples
   """
   mat_rb = rb_beta_df.loc[probe_list].copy()
   mat_retina = retina_beta_df.loc[probe_list].copy()
   mat_all = pd.concat([mat_rb, mat_retina], axis=1)
   col_colors = ["#44a8f4"] * mat_rb.shape[1] + ["#f9da41"] * mat_retina.shape[1]
   plt.figure(figsize=(2 + mat_all.shape[1]//3, 1 + 0.3*len(probe_list)))
   sns.heatmap(mat_all, cmap="YlGnBu", linewidths=0.2,
               linecolor='gray', cbar=True, robust=True,
               xticklabels=False, yticklabels=probe_list)
   plt.title("Beta values for RB (left) and Retina (right) samples")
   plt.xlabel("Samples (RB left, WT right)")
   plt.ylabel("Probes")
   plt.tight_layout()
   plt.savefig(filename, dpi=300)
   plt.close()
   print(f"Heatmap saved as: {filename}")



c = 1
for c in range(1,11,2):
    data = []
    rows = []
    i = f"top_{c*50}_probes"
    print("\n")
    probes = generate_ref_probes(i)
    print(len(probes))
    beta_embryo = beta_values_embryonic(probes)
    beta_rb = beta_values_rb(probes)
    
    for probe in tqdm(probes):
        print(probe)
        if probe not in rows:
            rows.append(probe)
        embryo_items = [beta_embryo[x][probe] for x in beta_embryo]
        rb_items = [beta_rb[x][probe] for x in beta_rb]
        data.append(rb_items+embryo_items)
        print(np.mean(embryo_items))
        print(np.mean(rb_items))

    data = np.array(data)


    fig, ax = plt.subplots(figsize=(12, 10))

    # 3. Create the heatmap plot using imshow
    # 'cmap' sets the color scheme; 'viridis', 'plasma', 'coolwarm', or 'YlGnBu' are common
    im = ax.imshow(data, cmap='viridis')

    # 4. Add a colorbar legend to scale colors to values
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Metric Scale", rotation=-90, va="bottom")

    ax.set_aspect('equal')

    # 5. Set tick positions and apply labels
    ax.set_yticks(np.arange(len(rows)), labels=rows)
    plt.show()
