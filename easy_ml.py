# General Packages
import os
from pathlib import Path
import inspect as insp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Models
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Evaluation Packages
from scipy.stats import uniform, randint
import seaborn as sns
from sklearn import metrics
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import classification_report, cohen_kappa_score, brier_score_loss
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
import shap
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor 

# function parameters:
# filepath: string, is the absolute path of the csv file that contains the paths for all feature tables.
    # There should be one path per cell, with as many cells as desired in the first column of the file. 
    # Each table pointed to by the paths must contain id_column in it for proper joining. 
    # All data in the tables pointed to by paths should be numeric. Categorical variables should already be one-hot encoded. 
# merge_base_path : string, to the absolute path of the csv file that contains the cohort of samples you want to 
    # build on. Must contain the id_column in it for merging. Only pulls the id_column from the file. 
# id_column: string, the name of the column that uniquely identifies each sample. 
    # Should be present in all tables being combined from filepath and merge_base_bath.
    # Defaults to subjectId, which is the column used in CENTER-TBI for this purpose. 
# drop_columns: a list that contains the column names that we want to drop
# rename_columns: the format should be a dictionary like {'A':'a'}
# drop_id/keep_id: a list that contains the subjectIds that we want to drop/keep
# export_name : feature csv file name, like 'test.csv'
# export_path : export path, path format, eg: '/home/idies/workspace/Storage/kgong1'

# 04/03/2022 update: add the numeric check function
# 06/21/2022 update: minor commenting changes, added in id_column as a specified unique identifer column. 
    # Also made the detector for non-numerical columns spit out column names and file names instead. 
    # Also applied column dropping before checking for numeric columns.
    # Also automated removal of columns that are all nan, and reported them back.
    # Also allowed resulting feature space table to output directly instead of just saved off. 
    # And changed function name to Feature_Assemble to be a more accurate name.
def feature_assemble(filepath,\
                    merge_base_path = '/home/idies/workspace/Storage/kgong1/CENTER-TBI Code and Results/ICU_subjectIds.csv',\
                    id_column = 'subjectId',\
                    drop_columns = None, rename_columns = None, drop_id = None, keep_id = None,\
                    export_name = 'temp.csv', export_path = os.getcwd()):
    # ignore warnings
    warnings.filterwarnings('ignore')
    
    # take in the testing csv and return feature csv
    path = pd.read_csv(filepath)
    path_len = path.shape[0]
    merge_base_path = Path(merge_base_path)
    df = pd.read_csv(merge_base_path)
    df = df.loc[:, id_column]

    # merge all listed feature on subjectId, and make sure all data is numerical, 
    # Drop columns to be dropped from this table. 
    # also print out the ones are not
    for i in range(0,path_len):
        curr_path = path.iloc[i,0]
        temp = pd.read_csv(curr_path)
        
        # filter the feature table by dropping certain columns
        if drop_columns:
            # Find intersection between list of dropped columns and columns in this feature table. 
            col_intersect = set(temp.columns) & set(drop_columns)
            # Drop them.
            temp = temp.drop(col_intersect, axis = 1)
        
        # Remove columns that were entirely missing, and report which ones.
        temp_orig = temp.copy()
        temp = temp.dropna(axis = 1, how = 'all')
        # Find columns that were removed for being all nans.
        col_all_nans = list(set(temp_orig.columns) - set(temp.columns) )
        # Report them.
        if len(col_all_nans) != 0:
            warnings.warn('The following columns from ' + curr_path + ' were all NANs and must be removed.')
            warnings.warn(str(col_all_nans))
        
        temp_without_subject = temp.drop([id_column],axis = 1)
        numeric_check = temp_without_subject.applymap(np.isreal)
        # Check if all columns are numeric. If not, spit out which ones aren't and which file they're from. 
        for col in temp_without_subject.columns:
            if numeric_check.loc[:,col].all() == False:
                warnings.warn('The column ' + col + ' of File ' + curr_path + ' is not numeric.')
        df = pd.merge(df, temp, on = id_column)

    # rename columns
    if rename_columns:
        df.rename(columns = rename_columns,inplace = True)
    
    # drop subjectIds as specified in the relevant parameter.
    if drop_id:
        drop_id_size = len(drop_id)
        for j in range(drop_id_size):
            drop_index = df[df[id_column] == drop_id[j]].index
            df = df.drop(drop_index, axis = 0)
            
    # keep subjectIds as specified in the relevant parameter.
    if keep_id:
        df = df[df[id_column].isin(keep_id)]

    # export part
    wd = Path(export_path)
    df.to_csv(wd.joinpath(export_name), index = False)
    return df

# df -> dataframe name, feature space data frame
# filepath -> path variable, feature space file path
    # must have at least one of the two above satisfied
# sample_id -> str, the column which only serves to identify different rows, will be dropped, eg: subjectId
# y_col -> str, label column
# test_size -> float between [0,1], train test split parameters
# random_state -> Integer, random seed for train test split
# shuffle -> Boolean variable for the option to shuffle
# scaling -> str, option for scaling strategy, including min_max scaling and standard scaling
# imputer_strategy -> str, parameter for Simple Imputer
# return X_train, X_test, y_train, y_test in memory as dataframe, without sample_id(subjectId)
# x_cols -> return feature names list in memory

# 04/08/2022 update: drop rows who have invalid label values(nan); replace infinite values with nan
def data_prep(df = None, filepath = None, sample_id = 'subjectId', y_col = 'GOSE', test_size = 0.2, random_state = 1, shuffle = True, \
              scaling = 'std', imputer_strategy = 'mean'):
    # ignore warnings
    warnings.filterwarnings('ignore')
    
    if filepath:
        dfs = pd.read_csv(filepath)
    if df.empty == False:
        dfs = df
    else:
        raise Exception('You have to input data!')
    
    # drop rows based on labels: those who have invalid labels(nan) will be dropped
    dfs = dfs[dfs[y_col].notna()]
    
    # replace infinite value with np.nan
    dfs = dfs.replace([np.inf,-np.inf], np.nan)
    
    # create the X and y dataframe based on the imput of y_col, meaning the user need to imput the
    # y column name
    X = dfs.drop([y_col, sample_id], axis = 1)
    y = dfs[[y_col]]
    
    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = \
                                                       random_state, shuffle = shuffle)
    
    # Report columns that were entirely missing.
    X_train_orig = X_train.copy()
    temp = X_train.dropna(axis = 1, how = 'all')
    # Find columns that were removed for being all nans.
    col_all_nans = list(set(X_train_orig.columns) - set(temp.columns) )
    # Report them.
    if len(col_all_nans) != 0:
        warnings.warn('The following columns were all NANs and must be removed.')
        warnings.warn(str(col_all_nans))
    
    # scaling options
    # standard scaler
    if scaling == 'std':
        scaler = preprocessing.StandardScaler()
    # min max scaler
    if scaling == 'MinMax':
        scaler = preprocessing.MinMaxScaler() 

    # get the x_col name as a list for scaling
    x_cols = X.columns.tolist()
    # scaling
    X_train = scaler.fit_transform(X_train[x_cols])
    X_test = scaler.fit_transform(X_test[x_cols])
    
    # imputation 
    imp = SimpleImputer(missing_values = np.nan, strategy = imputer_strategy)
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)
    
    # convert to dataframe
    X_train = pd.DataFrame(X_train,columns = x_cols)
    X_test = pd.DataFrame(X_test,columns = x_cols)
    y_train = pd.DataFrame(y_train,columns = [y_col])
    y_test = pd.DataFrame(y_test,columns = [y_col])
    
    return X_train, X_test, y_train, y_test, x_cols

# version 2: seperate the model training and evaluation
# models implemented: glm, logistic regression(lr), multiclass logistic regression(mlr),
##                    SVM, XGBoost, Lasso
# evaluation metric: roc&auc value, pr&F1 score, confusion matrix 
# evaluation: clf -> classification evaluation metric, reg -> regression evaluation metric
# add the cross validation parameter
# cross validation roc curve
# some thoughts: add the multiclass classifier to be a seperate parameters?
# drop features if less than feature importance threshold 

# parameter clarification
# X_train, X_test, y_train, y_test -> must be dataframe and altered for different fitting problems(regression, binary classification, multi-class classification)
# models: str, 'glm', 'lr', 'mlr'(multi-class classification), 'Lasso', 'SVM', 'XGBoost'
# evaluation: str, clf -> classification evaluation metric, reg -> regression evaluation metric
    # clf evaluation metric: roc&auc value, pr&F1 score, confusion matrix 
    # reg evaluation metric: Rooted mean square error
# cv -> Boolean, cross validation option, default 5 folds, return roc plot, auc value plot
# n_cpus -> the number of CPUs available to your code that you want to use. Passed to sklearn functions as n_jobs. 
# shapley -> Boolean, return shapley plot
# calibration_curve_bins -> # bins for the calibration curve, default 5
# feature_importance_plot -> Boolean, return feature importance plot
# feature_importance_select -> Boolean, decide whether to apply feature selection based on feature importance
    # if feature_importance_select == True, below will work
    # select based on feature importance
        # feature_importance_threshold -> Integer, value less than this threshold will be dropped
    # return list of column names to be dropped
# vif_select -> Boolean, decide whether to apply feature selection based on VIF
    # vif_threshold -> iterate the process of dropping max vif column until max vif is less than the threshold
    # return list of column names to be dropped
# model_export -> export model using pickle, named as model_pkl

# Updated 07/25/22: Add the brier score for binary classification evaluation and calibration curve
# Updated 6/23/22: Removed feature_importance_abs as it's unnecessary, and just had code always take absolute value as there will never be harm from it.
# Also defined vif_drop_cols and fi_drop_cols outside of if statements, so that return at end will return empty lists if feature importance/VIF not done.
def model_training_evaluation(X_train = None, X_test = None, y_train = None, y_test = None, 
                              models = 'glm', evaluation = 'clf', cv = True, n_cpus = 1, 
                              shapley = True, calibration_curve_bins = 5,
                              feature_importance_plot = True, feature_importance_select = True, 
                              feature_importance_abs = True, feature_importance_threshold = 0,
                              vif_select = True, vif_threshold = 10, model_export = True):
    
    # ignore warnings
    warnings.filterwarnings('ignore')
    
    if models:
        pass
    else:
        raise Exception('You have to choose a model!')
        
    # model selection
    if models == 'glm':
        clf = linear_model.LinearRegression()
            
    if models == 'lr':
        # logistic regression
        clf = linear_model.LogisticRegression()
            
    if models == 'mlr':
        # define model
        model = linear_model.LogisticRegression(max_iter = 200)
        # define the ovr strategy
        clf = OneVsRestClassifier(mlr)
    
    if models == 'Lasso':
        # Lasso -- Logistic regression with l1 penalty
        model = linear_model.LogisticRegression(solver = 'saga')
        para = {'C':np.arange(0.1,1.1,0.1),
               'penalty':['l1']}
        cv_inner = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)
        # define search
        search = RandomizedSearchCV(model, para, n_iter = 30, scoring = 'roc_auc', 
                                    cv = cv_inner, verbose = 1, refit = True, random_state = 1, n_jobs = n_cpus)
    
    if models == 'SVM':
        # SVM with rbf kernel
        model = SVC(kernel = 'rbf')
        para = {'C':uniform(loc=0, scale = 100),
               'gamma':uniform(loc=0, scale = 1)}
        cv_inner = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)
        # define search
        search = RandomizedSearchCV(model, para, n_iter = 30, scoring = 'roc_auc', cv = cv_inner, 
                                    verbose = 1, refit = True, random_state = 1, n_jobs = n_cpus)
        
    if models == 'xgb':
        # xgboost
        model = XGBClassifier(use_label_encoder = False)
        para = {'eta' : uniform(loc=0, scale = 1),
                'gamma' : randint(0,100),
                'max_depth' : randint(1,X_train.shape[1]),
                'n_estimators' : randint(10,1000),
                'colsample_bytree' : uniform(loc=0, scale = 1),
                'min_child_weight' : [0, 1, 2, 3, 4]}
        cv_inner = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)
        # define search
        search = RandomizedSearchCV(model, para, n_iter = 30, scoring = 'roc_auc', cv = cv_inner, 
                                    verbose = 1, refit = True, random_state = 1, n_jobs = n_cpus)
        
    # check the stability of the model
    ## use the best estimator for cross validation make sense?
    if cv:
        if (models == 'lr') | (models == 'mlr'):
            classifier = clf
        if (models == 'glm'):
            raise Exception('GLM is not applicable at this moment and will be added in later version!')
        if (models == 'SVM') | (models == 'xgb') | (models == 'Lasso'):
            classifier = search
        cv_outer = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)
        k = [1,2,3,4,5]
        scores = cross_val_score(classifier, X_train, y_train, scoring = 'roc_auc', 
                                 cv = cv_outer, verbose = 1, n_jobs = n_cpus)
        plt.figure(1)
        plt.plot(k, scores, 's-')
        plt.ylim(0,1)
        plt.xticks([1,2,3,4,5])
        plt.xlabel('kth cross validation')
        plt.ylabel('AUC value')
        plt.title(models + ' cross validation result')
        # roc curve for each fold
        X_train_arr = X_train.to_numpy()
        y_train_arr = y_train.to_numpy()

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(figsize = (8,8))
        for i, (train, test) in enumerate(cv_outer.split(X_train_arr, y_train_arr)):
            classifier.fit(X_train_arr[train], y_train_arr[train])
            viz = metrics.plot_roc_curve(classifier,X_train_arr[test],y_train_arr[test],
                                         name="ROC fold {}".format(i),alpha=0.3,lw=1,ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            
        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )
        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title=models + ' cross validation results',
        )
        ax.legend(loc="lower right") # prop={'size': 6}
        plt.show()
    
    # lr, mlr, glm:fit the training data 
    # Lasso, SVM, XGBoost: get the best estimator
    if (models == 'lr') | (models == 'mlr') | (models == 'glm'):
        clf.fit(X_train,y_train)
    else:
        # search for the best model
        result = search.fit(X_train, y_train)
        clf = result.best_estimator_
    
    # classification model evaluation
    if evaluation == 'clf':
        y_test_pred = clf.predict(X_test)
        print(classification_report(y_test, y_test_pred))
        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 6))
        # Create the matrix
        cm = metrics.confusion_matrix(y_test, y_test_pred)
        cmp = metrics.ConfusionMatrixDisplay(cm)
        cmp.plot(ax=ax)
        plt.show();
        # brier score
        y_test_pred_prob_temp = clf.predict_proba(X_test)
        y_test_pred_prob = np.max(y_test_pred_prob_temp, axis=1)
        brier_score = brier_score_loss(y_test, y_test_pred_prob)
        print('Brier score is',brier_score)
        # calibration curve
        if calibration_curve_bins:
            y_test_pos_prob = y_test_pred_prob_temp[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_test_pos_prob, n_bins=calibration_curve_bins)
            disp = CalibrationDisplay(prob_true, prob_pred, y_test_pos_prob)
            disp.plot()
        # roc curve
        metrics.plot_roc_curve(clf, X_test, y_test)
        plt.plot([0, 1], [0, 1],'r--')
        plt.title(models + ' ROC curve')
        # pr curve
        metrics.plot_precision_recall_curve(clf, X_test, y_test)
        # dummy classifiar
        no_skill = len(y_test[y_test.iloc[:,0]==1]) / len(y_test)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.ylim((0,1))
        plt.title(models + ' Precision Recall Curve')
    
    # regression model evaluation
    if evaluation == 'reg':
        y_test_pred = clf.predict(X_test)
        print('Accuracy is',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
    
    # shapley value
    if shapley:
        ex = shap.Explainer(clf, X_test)
        shap_values = ex.shap_values(X_test)
        fig = plt.figure()
        shap.summary_plot(shap_values, X_test)
    
    # calculate feature importance anyway
    if (models == 'glm') | (models == 'lr') | (models == 'mlr') | (models == 'Lasso'):
        feat = clf.coef_[0]
    if models == 'xgb':
        feat = clf.feature_importances_
    top_feats = pd.DataFrame(data = {'column_name': X_train.columns.values.tolist(),
                             'coefficient': feat})
    top_feats['abs_coef'] = top_feats['coefficient'].abs()
    # plot top 20 feature importance
    if feature_importance_plot:
        if models == 'SVM':
            raise Exception('Feature importance does not apply to SVC with non-linear kernel!')
        top_feats.sort_values('abs_coef', ascending = False, inplace = True)
        # For testing. 
        top_feats.to_csv('test_feat_imp.csv', index = False)
        top_feats.iloc[0:20,:].sort_values('abs_coef', ascending = True).plot.barh(x = 'column_name', 
                                                                           y = 'abs_coef')
        plt.title(models + ' feature importance')
    
    # select feature based on feature importance
    fi_drop_cols = []
    if feature_importance_select:
        if models == 'SVM':
            raise Exception('Feature importance does not apply to SVC with non-linear kernel!')
        if (feature_importance_threshold < 0):
            raise Exception("Your parameters don't make sense with a negative threshold!")
        # fi refers to feature importance
        fi_drop_cols = top_feats[top_feats['abs_coef'] <= feature_importance_threshold]
        print(len(fi_drop_cols), ' features to drop based on feature importance: ')
        print(fi_drop_cols['column_name'].values.tolist())
    
    #variance inflation factor
    vif_drop_cols = []
    if vif_select:
        if models == 'SVM':
            raise Exception('Variance inflation factor does not apply to SVC with non-linear kernel!')
        # VIF dataframe
        max_vif = np.inf
        
        temp = top_feats[top_feats['abs_coef'] != 0]['column_name'].values.tolist()
        X_test_temp = X_test.loc[:,temp]
        X_test_temp['intercept'] = 1
        
        while max_vif > vif_threshold:
            vif_data = pd.DataFrame()
            vif_data["feature"] = X_test_temp.columns
            vif_data["VIF"] = [variance_inflation_factor(X_test_temp.values, i)
                              for i in range(len(X_test_temp.columns))]
            vif_data.sort_values('VIF', inplace = True, ascending = False)
            # get the max value and corresponding feature
            max_vif = vif_data['VIF'].max()
            max_vif_feature = vif_data.iloc[0,0]
            print('max_vif:' + str(max_vif))
            if max_vif < vif_threshold:
                break
            # drop max_vif feature
            X_test_temp.drop(columns = [max_vif_feature], inplace = True)
            vif_drop_cols.append(max_vif_feature)
            
        print(len(vif_drop_cols), ' features to drop based on VIF below:')
        print(vif_drop_cols)
        
    if model_export == True:
        with open('model_pkl', 'wb') as files:
            pickle.dump(clf, files)
    return clf, fi_drop_cols, vif_drop_cols