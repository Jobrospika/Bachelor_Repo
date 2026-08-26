import pickle
import pandas as pd
import yaml
import time

def evaluate_safety_violations(df):
    """
    Evaluate the number of runs which resulted in safety violations.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the experiment results.
    
    Returns:
        int: Number of runs which resulted in safety violations.
    """
    df["violation"] = df["safety_violation"] > 0
    return df['violation'].sum()

def evaluate_not_started(df):
    """Evaluate the number of runs which could not be started due to safety violations.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the experiment results.
        
    Returns:
        int: Number of runs which could not be started."""
    return df['no_opt_possible'].sum()


def evaluate_all(experiment_path, experiment_name):
    """Evaluate all runs of an experiment.
    
    Parameters:
        experiment_path (str): Path to the experiment results.
        experiment_name (str): Name of the experiment.
        
    Returns:
        pd.DataFrame: DataFrame containing the evaluation results.
    """
    
    res_df = pd.DataFrame(columns=["onb_function", "safety_violations", "not_started"])
    for i in range(0,100):
            res_dict={}
            df = pd.read_csv(experiment_path + f"_{i}/experiment_{experiment_name}_0_10000.csv")
            res_dict["onb_function"] = i 
            res_dict["safety_violations"] = evaluate_safety_violations(df)
            res_dict["not_started"] = evaluate_not_started(df)
            res_df = pd.concat([res_df, pd.DataFrame([res_dict])], ignore_index=True)
    return res_df

def evaluate_experiment_1(path):
    """Evaluate the performance of the experiment.
    
    Parameters:
        path (str): Path to the experiment results.
    
    Returns:
        float: Performance of the experiment. 
    """
    # read csv 
    df = pd.read_csv(path)
    return (10000 - df["contains_all"].sum())/10000


def evaluate_performance(path, function_info, ax, label):
    """
    Evaluate the performance of the experiment. Normalized pred_opt values are used.
    
    Parameters:
        path (str): Path to the experiment results.
        function_info (dict): Information about the function being optimized.
        ax (matplotlib.axes): Axes to plot the results.
        label (str): Label for the plot.
    
    Returns:
        matplotlib.axes: Axes with the plot of the results.
        
    """
    all_data = pd.DataFrame()
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)
    
    for seed in loaded_dict.keys():
        df = pd.DataFrame(loaded_dict[seed])
        df["pred_opt"] = (df["pred_opt"]-function_info["safety_threshhold"])/(function_info["optimum_value_noiseless"]-function_info["safety_threshhold"])
        all_data = pd.concat([all_data, df], ignore_index=True)

    # Group by 'iteration' and calculate mean and standard deviation
    grouped = all_data.groupby('iteration')['pred_opt']
    mean = grouped.mean()
    std = grouped.std()
    q1 = grouped.quantile(0.1)
    q9 = grouped.quantile(0.9)
    all_data["safety_vioation"] = all_data["y"]<function_info["safety_threshhold"]
    
    # Plotting
    ax.plot(mean.index, mean, label=f'{label} Mean')
    ax.fill_between(mean.index, q1, q9, alpha=0.2, label=f'{label} Std Dev')
    return ax

def evaluate_performance_ref(path, function_info, ax, label):
    all_data = pd.DataFrame()
    with open(path, 'rb') as f:
        loaded_dict = pickle.load(f)

    dfs = [pd.concat([pd.DataFrame(loaded_dict[seed])] * 21, ignore_index=True) if len(pd.DataFrame(loaded_dict[seed])) == 1 else pd.DataFrame(loaded_dict[seed]) for seed in loaded_dict.keys()]

    # Updating the 'iteration' column for each dataframe that was expanded
    for df in dfs:
        if len(df) == 21:
            df['iteration'] = range(0, 21)
    df = pd.concat(dfs, ignore_index=True)
        
    # Calculate pred_opt and add it to the DataFrame
    df["pred_opt"] = (df["pred_opt"] - function_info["safety_threshhold"]) / (function_info["optimum_value_noiseless"] - function_info["safety_threshhold"])
    all_data = pd.concat([all_data, df], ignore_index=True)
    end = time.time()
    grouped = all_data.groupby('iteration')['pred_opt']
    mean = grouped.mean()
    std = grouped.std()
    q1 = grouped.quantile(0.1)
    q9 = grouped.quantile(0.9)
    all_data["safety_vioation"] = all_data["y"]<function_info["safety_threshhold"]
    
    # Plotting
    ax.plot(mean.index, mean, label=f'{label} Mean')
    ax.fill_between(mean.index, q1, q9, alpha=0.2, label=f'{label} Std Dev')
    return ax

def evaluate_multiple_performance(experiment_path, experiment_name, function_type,  ax, label, color="b"):
    """
    Evaluate the performance of the experiment. Csv files are evaluated and stored in a pickle file for plotting.  
    Normalized pred_opt values are used.
    
    Parameters:
        experiment_path (str): Path to the experiment results.
        experiment_name (str): Name of the experiment.
        function_type (str): Type of the function.
        ax (matplotlib.axes): Axes to plot the results.
        label (str): Label for the plot.
        color (str): Color for the plot.
    
    Returns:
        pd.DataFrame: DataFrame containing the evaluation results.
        matplotlib.axes: Axes with the plot of the results.
    """
    res_df = pd.DataFrame(columns=["function", "safety_violations", "not_started", "final_performance"])
    all_data = pd.DataFrame()
    mean_funs = pd.DataFrame()
    for i in range(0,100):
            print(i)
            start = time.time()
            
            try:
                with open(f"src/function_configs/{function_type}_functions" + f"/{function_type}_function_{i}.yaml", 'r') as yaml_file:
                    function_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
                #function_info = yaml.load(open(experiment_path+f"/pre_rkhs_function_{i}.yaml"), Loader=yaml.FullLoader)
                path = experiment_path + f"/{function_type}_function_{i}/experiment_results_{experiment_name}_0_10000.pkl"
                with open(path, 'rb') as f:
                    loaded_dict = pickle.load(f)
            except:
                with open("src/function_configs/onb_functions" + f"/onb_function_{i}.yaml", 'r') as yaml_file:
                    function_info = yaml.load(yaml_file, Loader=yaml.FullLoader)
                #function_info = yaml.load(open(experiment_path+f"/pre_rkhs_function_{i}.yaml"), Loader=yaml.FullLoader)
                path = experiment_path + f"/onb_function_{i}/experiment_results_{experiment_name}_0_10000.pkl"
                with open(path, 'rb') as f:
                    loaded_dict = pickle.load(f)
            not_started = 0
            #for seed in loaded_dict.keys(): 
            for i in range(1000):
                seed = f"seed_{i}"      
                if len(pd.DataFrame(loaded_dict[seed])) == 1:
                    not_started=+1 
            # Use list comprehension for DataFrame creation and concatenation
            dfs = [pd.concat([pd.DataFrame(loaded_dict[seed])] * 21, ignore_index=True) if len(pd.DataFrame(loaded_dict[seed])) == 1 else pd.DataFrame(loaded_dict[seed]) for seed in loaded_dict.keys()]

            # # Updating the 'iteration' column for each dataframe that was expanded
            for df in dfs:
                if len(df) == 21:
                    df['iteration'] = range(0, 21)
            df = pd.concat(dfs, ignore_index=True)
            
            # Calculate pred_opt and add it to the DataFrame
            df["pred_opt"] = (df["pred_opt"] - function_info["safety_threshhold"]) / (function_info["optimum_value_noiseless"] - function_info["safety_threshhold"])
            grouped1 = df.groupby('iteration')['pred_opt']
            mean_funs[f'fun{i}'] = grouped1.mean()
            all_data = pd.concat([all_data, df], ignore_index=True)

            end = time.time()
            safety_violation = (df["y"]<function_info["safety_threshhold"]).sum()
            res_df = pd.concat([res_df, pd.DataFrame([{"function": i, "safety_violations": safety_violation, "not_started": not_started, "final_performance": df["pred_opt"].iloc[-1]}])], ignore_index=True)

            print(end-start)
    grouped = all_data.groupby('iteration')['pred_opt']
    mean = grouped.mean()
    q1 = grouped.quantile(0.1)
    q9 = grouped.quantile(0.9)
    all_data["safety_vioation"] = all_data["y"]<function_info["safety_threshhold"]
    
    # Plotting
    ax.plot(mean.index, mean, label=f'{label} Mean', color=color)
    ax.fill_between(mean.index, q1, q9, alpha=0.2, label=f'{label} Std Dev', color=color)
    ax.plot(mean_funs.index, mean_funs.iloc[:,1:100], alpha=0.1, linestyle="solid", color=color, linewidth=0.1)
    
    save_res_dict = {"mean": mean, "q1": q1, "q9": q9, "mean_funs": mean_funs}
    with open(experiment_path + f"/summary_results_{experiment_name}_1000.pkl", 'wb') as f:
        pickle.dump(save_res_dict, f)
        
    return res_df, ax