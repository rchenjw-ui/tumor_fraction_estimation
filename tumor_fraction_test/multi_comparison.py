import numpy as np
import sklearn
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#future plans: feature importance, apply standard scaler, EM-SEQ

def og_tfx():
    with open('probe_beta_pairs.csv', 'r') as file:
        file = file.read().splitlines()
        probes_to_sample = {}
        for i in file[1:]:
            line = i.split(',')
            probe = line[0]
            beta_rb = line[1]
            beta_wt = line[2]       
            if probe not in probes_to_sample:
                probes_to_sample[probe] = {}
            probes_to_sample[probe]['beta_rb'] = beta_rb
            probes_to_sample[probe]['beta_wt'] = beta_wt


    with open('pilot_8samples_merged.csv', 'r') as file:
        #create dictionary, sample_wise probe addition
        file = file.read().splitlines()
        print(file[0])
        out = {}
        for i in file[1:]:
            line = i.split(',')
            probe = line[0]
            if probe in probes_to_sample:
                for c in range(len(line)-1):
                    id = file[0].split(",")[c+1]
                    data = line[c+1]
                    #handling
                    if id not in out:
                        out[id] = {}
                    out[id][probe] = data
    for c in out:
        print(c)
        print(len(out[c]))
        tfx_list =[]
        for i in out[c]:  
            probe = i
            rb_beta = float(probes_to_sample[i]['beta_rb'])
            wt_beta = float(probes_to_sample[i]['beta_wt'])
            sample_beta = float(out[c][i])
            estimated_tfx = (sample_beta - wt_beta) / (rb_beta - wt_beta)
            #print(probe, estimated_tfx)
            tfx_list.append(estimated_tfx)
        print("median tfx: ", np.median(tfx_list))
    

def graph_metrics(results,name):

    data = []
    for i in results:
        array = results[i]
        print(array)
        data = data + [array/np.max(array)]

    data = np.array(data)
    items = list(range(1,11,2))
    metrics = list(results.keys())

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap='viridis')  # 'viridis', 'plasma', 'coolwarm', etc.

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Scale Label", rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(items)), labels=items)
    ax.set_yticks(np.arange(len(metrics)), labels=metrics)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title(f"{name} Metrics Heatmap")
    fig.tight_layout()
    plt.show()


def display_preds(preds, name):
    plt.figure(figsize=(12, 7))
    plt.bar(samples.keys(), preds, color=["#00bfff", "#00bfff", "#ffe600", "#ffe600","#00bfff", "#00bfff", "#ffe600", "#ffe600"])
    plt.ylabel(f"Predicted Tumor Fraction - {name}")
    plt.title(f"Predicted Tumor Fraction for Each Sample - {name}")  
    plt.show()
    plt.savefig(f"predicted_tumor_fraction_barplot_{name}.png")
    plt.close()

def display_residual(residuals, name):
    plt.figure(figsize=(12, 7))
    plt.bar(samples.keys(), residuals, color=["#00bfff", "#00bfff", "#ffe600", "#ffe600","#00bfff", "#00bfff", "#ffe600", "#ffe600"])
    plt.ylabel(f"Residuals - {name}")
    plt.title(f"Residuals - {name}")  
    plt.show()
    plt.savefig(f"residuals_barplot_{name}.png")
    plt.close()





#______________________________________________________________________________________________________________________________________________________________________________

lr_metrics = {"um_to_blood":[], "cancer_to_blood":[], "um_to_cancer":[],"cv_score":[], "train_vs_test": [], "f1": []}
nnls_metrics = {"um_to_blood":[], "cancer_to_blood":[], "um_to_cancer":[], "residual_to_signal": []}
tfx_metrics = {"um_to_blood":[], "cancer_to_blood":[], "um_to_cancer":[]}


for name in range(1,11,2):

    #start_logistic_regression
    print(name*50)
    y = np.load(f'output_matrix_top_{name*50}_probes.npy')
    X = np.load(f'feature_matrix_top_{name*50}_probes.npy')

    #k-fold cross validation: 

    model = sklearn.linear_model.LogisticRegression()



    #run cross validation
    print("\nRunning cross-validation...")
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores):.4f}")



    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    model = sklearn.linear_model.LogisticRegression()

    model.fit(X_train, y_train)
    #need accuracy and f1 score
    #evaluate F1 score and accuracy
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    print("\nEvaluating model performance on test set...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    train_vs_test = roc_auc_score(y_train, model.predict(X_train)) / roc_auc_score(y_test, model.predict(X_test))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC_AUC train vs test: {train_vs_test}")

    print("Training model on full data...")
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_x = scaler.transform(X)
    print("scaler scale")
    print(scaler.scale_)
    print(len(scaler.scale_))
    print("scaler mean")
    print(scaler.mean_)
    model.fit(scaled_x, y)


    #stop_logistic_regression
    
    path = f"top_{name*50}_probes_probe_beta_pairs.csv"
    df = pd.read_csv(path)
    ref_probes = df["probe"].tolist()
    ref_wt = df["beta_wt"].tolist()
    ref_rb = df["beta_rb"].tolist()
    A = np.array([ref_wt, ref_rb]).T
    
    #get the values 

    samples = {}
    with open("pilot_8samples_merged.csv", 'r') as file:
        #create dictionary, sample_wise probe addition
        file = file.read().splitlines()
        labels = file[0].split(",")
        for i in file[1:]:
            line = i.split(',')
            if line[0] in ref_probes:
                for j in range(1,len(line)):
                    if labels[j] in samples:
                        samples[labels[j]].append(float(line[j]))
                    else:
                        samples[labels[j]] = [float(line[j])]

    nnls_tfx = []

    preds_nnls = []
    residual_to_signal = []
    for i in samples:
        B = np.array(samples[i])
        tfx, rnorm = scipy.optimize.nnls(A, B)
        l2 = np.linalg.norm(B)
        preds_nnls.append(tfx[1])
        nnls_tfx.append((i, tfx[1], rnorm, rnorm/l2))
        residual_to_signal.append(rnorm/l2)
        #find l2 norm of B for signal to noise given residual
        print(f"Sample: {i}, Estimated Tumor Fraction: {tfx[1]}, Residual Norm: {rnorm}, Residual To Signal: {rnorm/l2}")

    
    preds = [x[1] for x in model.predict_proba(scaler.transform([samples[i] for i in samples]))]
    coefficients = model.coef_
    print(coefficients[0].shape)
    probe_vs_coeff = {ref_probes[i]:coefficients[0][i] for i in range(len(ref_probes))}
    print("\n")
    max_probe = max(probe_vs_coeff, key = probe_vs_coeff.get)
    print(f"Probe with largest coefficient: {max_probe}")
    print(f"Odds ratio: {np.exp(max(probe_vs_coeff.values()))}")
    min_probe = min(probe_vs_coeff, key = probe_vs_coeff.get)
    print(f"Probe with smallest coefficient: {min_probe}")
    print(f"Odds ratio: {np.exp(min(probe_vs_coeff.values()))}")
    print("\n")
    
    #graph the mean CV score, F1, and accuracy

    fig, ax = plt.subplots(figsize=(12, 7))
    plt.bar(["CV", "F1", "Accuracy"], [np.mean(cv_scores), f1, accuracy])
    plt.title(f"Cross Validation and Train/Test metrics - logistic regression - {name*50}")   
    ax.bar_label(ax.bar(["CV", "F1", "Accuracy"], [np.mean(cv_scores), f1, accuracy]), padding=3) 
    plt.show()
    plt.savefig(f"Logistic_Regression_Metrics_{name*50}.png")
    plt.close()
    #data visualization code
    results_tfx = og_tfx()
    display_preds(results_tfx, "tfx")
    display_preds(preds, "logistic regression")
    display_preds(preds_nnls, "nnls")
    display_residual(residual_to_signal, "nnls")


    um_to_blood = ((preds[6]+preds[7])/2) /((preds[2]+preds[3])/2)
    cancer_to_blood = ((preds[0]+preds[1])/2) /((preds[2]+preds[3])/2)
    um_to_cancer = um_to_blood / cancer_to_blood
    working_results = {"um_to_blood":um_to_blood, "cancer_to_blood":cancer_to_blood, "um_to_cancer":um_to_cancer, "cv_score":np.mean(cv_scores), "train_vs_test": train_vs_test, "f1": f1}
    for i in lr_metrics:
        lr_metrics[i] = lr_metrics[i] + [working_results[i]]

    um_to_blood = ((preds_nnls[6]+preds_nnls[7])/2) /((preds_nnls[2]+preds_nnls[3])/2)
    cancer_to_blood = ((preds_nnls[0]+preds_nnls[1])/2) /((preds_nnls[2]+preds_nnls[3])/2)
    um_to_cancer = um_to_blood / cancer_to_blood
    working_results = {"um_to_blood":um_to_blood, "cancer_to_blood":cancer_to_blood, "um_to_cancer":um_to_cancer, "residual_to_signal":np.nanmean(residual_to_signal)}
    for i in nnls_metrics:
        nnls_metrics[i] = nnls_metrics[i] + [working_results[i]]

    um_to_blood = ((results_tfx[6]+results_tfx[7])/2) /((results_tfx[2]+results_tfx[3])/2)
    cancer_to_blood = ((results_tfx[0]+results_tfx[1])/2) /((results_tfx[2]+results_tfx[3])/2)
    um_to_cancer = um_to_blood / cancer_to_blood    
    for i in tfx_metrics:
        tfx_metrics[i] = tfx_metrics[i] + [working_results[i]]

    #print results
    for i, j in zip(labels[1:], preds):
        print(f"sample: {i} tumor fraction: {j}")





#metrics farming

graph_metrics(lr_metrics, "Logistic Regression")
graph_metrics(nnls_metrics, "NNLS")
graph_metrics(tfx_metrics, "tfx")







