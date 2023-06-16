%% H.W. 2 - Features selection
%% Load data, event triggered segmentation, and feature extraction
% X — Feature matrix
% Y — labels
% tp - true positive. segment/event that contains 1-6 labels. 
% fp - false positive. segment/event that doesn't contains 1-6 labels.
% fn - false negetive. 1-5 events that were missed. must be zero on train dataset.
% tp, fp, fn in values and not precentages.


tmp = split(pwd,'\');
tmp = join(tmp(1:end-1,1),'\');
mainpath = tmp{1,1};
train_folder_path = strcat(mainpath,'\train');
test_folder_path = strcat(mainpath,'\test');

[X_train, Y_train ,tp ,fp, fn] = event_triggered_feature_extraction(train_folder_path); 

precision = tp/(tp+fp);
sensitivity = tp/(tp+fn);
f1_score = 2/(1/sensitivity+1/precision);
disp(['train feature matrix dim: ', num2str(size(X_train))])
disp(['train labels dim: ', num2str(size(Y_train))])
disp(['train false negetive: ', num2str(fn)]) % must be zero on train dataset
disp(['train precision: ', num2str(precision)])
disp(['train sensitivity: ', num2str(sensitivity)])
disp(['train f1_score: ', num2str(f1_score)])

[X_test, Y_test ,tp ,fp, fn] = event_triggered_feature_extraction(test_folder_path); 

precision = tp/(tp+fp);
sensitivity = tp/(tp+fn);
f1_score = 2/(1/sensitivity+1/precision);
disp(['test feature matrix dim: ', num2str(size(X_test))])
disp(['test labels dim: ', num2str(size(Y_test))])
disp(['test false negetive: ', num2str(fn)]) 
disp(['test precision: ', num2str(precision)])
disp(['test sensitivity: ', num2str(sensitivity)])
disp(['test f1_score: ', num2str(f1_score)])

%% Features vetting
% vff_max - feature-feature maximum correlation value
% vff_mean - feature-feature average correlation value
% vft_max - feature-target maximum Relieff value
% vft_mean - feature-target average Relieff value

[vff_max, vff_mean, vft_max, vft_mean] = before_features_vetting_fit(X_train, Y_train); 
% return stats of scores (for monitoring) before you apply features vetting
% procedure on more then 40 features.
disp(['train prior to features vetting feature-feature max: ', num2str(vff_max)])
disp(['train prior to features vetting feature-feature average: ', num2str(vff_mean)])
disp(['train prior to features vetting feature-target max: ', num2str(vft_max)])
disp(['train prior to features vetting feature-target average: ', num2str(vft_mean)])

[Xv_train, vff_max, vff_mean, vft_max, vft_mean] = features_vetting_fit(X_train, Y_train); 
% perform features vetting
disp(['train features vetting feature-feature max: ', num2str(vff_max)])
disp(['train features vetting feature-feature average: ', num2str(vff_mean)])
disp(['train features vetting feature-target max: ', num2str(vft_max)])
disp(['train features vetting feature-target average: ', num2str(vft_mean)])

[Xv_test, vff_max, vff_mean, vft_max, vft_mean] = features_vetting_transform(X_test, Y_test); 
% apply features vetting manualy on test dataset
disp(['test features vetting feature-feature max: ', num2str(vff_max)])
disp(['test features vetting feature-feature average: ', num2str(vff_mean)])
disp(['test features vetting feature-target max: ', num2str(vft_max)])
disp(['test features vetting feature-target average: ', num2str(vft_mean)])

%, worst_idx_spearman, best_20_relieff
%% Features selection
% Return the following:
% n_combinations - number of combinations
% nft_max - combinations maximum value
% nft_mean - combinations average value
% nft_std - combinations std value
% best_comb - best combination in any format 

[Xs_train, n_combinations, nft_max, nft_mean, nft_std, best_comb] = features_selection_fit(Xv_train, Y_train); 

disp(['train features selection combinations: ', num2str(n_combinations)])
disp(['train features selection feature-target max: ', num2str(nft_max)])
disp(['train features selection feature-target average: ', num2str(nft_mean)])
disp(['train features selection feature-target std: ', num2str(nft_std)])
disp(['best combinat ion: ',num2str(best_comb)])

[Xs_test, nft] = features_selection_transform(Xv_test, Y_test); 
% nft - test dataset mutual information features-target value for selected combination
disp(['test features selection feature-target value: ', num2str(nft)])

%% Random Forest Classification 
% accuracy_vs_n_trees - accuracy vector of accuracy vs. number of trees 
% sensitivity_arr - sensitivity per class 
% example - sensitivity_arr = [0.66 for class 1 , ... , 0.59 for class 5]
% precision_arr - precision per class 
% f1_score_arr - f1_score per class 
% auc_arr_arr - auc_arr per class

[accuracy_vs_n_trees, sensitivity_arr, precision_arr, f1_score_arr, train_auc_arr, test_auc_arr] = ...
    RF_classification(Xs_test, Y_test, Xs_train, Y_train); 

disp(['accuracy vs. number of trees - accuracy vector: ', num2str(accuracy_vs_n_trees)])
disp(['sensitivity per class: ', num2str(sensitivity_arr)])
disp(['average sensitivity : ', num2str(mean(sensitivity_arr))])
disp('----------')
disp(['precision per class: ', num2str(precision_arr)])
disp(['average precision : ', num2str(mean(precision_arr))])
disp('----------')
disp(['f1 per class: ', num2str(f1_score_arr)])
disp(['average f1 : ', num2str(mean(f1_score_arr))])
disp('----------')
disp(['train auc per class: ', num2str(train_auc_arr)])
disp(['average train auc : ', num2str(mean(train_auc_arr))])
disp('----------')
disp(['test auc per class: ', num2str(test_auc_arr)])
disp(['average test auc : ', num2str(mean(test_auc_arr))])
disp('----------')




