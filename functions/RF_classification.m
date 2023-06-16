function [accuracy_vs_n_trees, sensitivity_arr, precision_arr, f1_score_arr, train_auc_arr, test_auc_arr] = RF_classification(Xs_test, Y_test, Xs_train, Y_train)
    %% a - Create the Random Forest ensemble and finding the number of trees
 
    [train_idx, ~, test_idx] = dividerand(size(Xs_train,1), 0.7, 0, 0.3);
    %Train Set 
    istrain = Xs_train(train_idx, :);
    isY_train = Y_train(1,train_idx);
    %Test Set
    istest = Xs_train(test_idx, :);
    isY_test = Y_train(1,test_idx);
        
    % loop over the possible number of trees an calculate accuracy per num
    % of trees
    N = size(istrain, 1);
    t = templateTree('MaxNumSplits', N, 'NumVariablesToSample', 3);
    num_trees=10:10:100;
    accuracy_vs_n_trees = zeros(1, length(num_trees));

    for i = 1:length(num_trees) 
        RUS = fitcensemble(istrain,isY_train,'Method','RUSBoost', ...
        'NumLearningCycles',num_trees(i),'Learners',t,'LearnRate',0.1,'nprint',10);
        
        [Y_predict,~] = predict(RUS,istest);
        CM = confusionmatStats(isY_test,Y_predict);
        accuracy_vs_n_trees(i) = CM.accuracy;
    end
     
    % plot of graph to chose th elbow point for the best num of trees
    figure(1)
    plot(num_trees,accuracy_vs_n_trees.*100,num_trees,accuracy_vs_n_trees.*100,'o');
    xlabel('number of tress')
    ylabel('accuracy [%]')
    title('accuracy vs. number of trees')
    
    
    %% b - sensitivity, precision and f1 score per class
    % training the random forest model with the best number of trees on
    % 100% of the train data set 
    
    final_num_of_trees =30;
    disp(['chosen number of trees is ',num2str(final_num_of_trees)])
    RUS_test = fitcensemble(Xs_train,Y_train,'Method','RUSBoost', ...
        'NumLearningCycles',final_num_of_trees,'Learners',t,'LearnRate',0.1,'nprint',10);
   
    % calculate predictions on test data set with the trained model    
    [Y_predict,score_test] = predict(RUS_test,Xs_test);
    CM = confusionmatStats(Y_test,Y_predict);   % confusiuon matrix

    %calculatuin sensitivity, specificity, precision and f1 score
    sensitivity_arr = CM.sensitivity';
    specificity_arr = CM.specificity';
    precision_arr = CM.precision';
    f1_score_arr = CM.Fscore';
    
    % leaving only 1-5 lables scores
    sensitivity_arr = sensitivity_arr(2:6);
    specificity_arr = specificity_arr(2:6);
    precision_arr = precision_arr(2:6);
    f1_score_arr = f1_score_arr(2:6);

    %% c - Calculate ROC AUC per class (1-5 classes, one vs. rest)
    labels_occur_test=tabulate(Y_test);
    labels_occur_train=tabulate(isY_test);
    train_auc_arr = zeros(1,size(labels_occur_train,1));
    test_auc_arr = zeros(1,size(labels_occur_test,1));

    % training the random forest model with the best number of trees on
    % 70% of the train data set to make predictions on the other 30%
     RUS_train = fitcensemble(istrain,isY_train,'Method','RUSBoost', ...
        'NumLearningCycles',final_num_of_trees,'Learners',t,'LearnRate',0.1,'nprint',10);
        
    [~,score_train] = predict(RUS_train,istest); %predictions scores on the other 30% of the training data
   
    % calculation of AUC for train and test data sets 
    for label = 1:size(labels_occur_test,1)
        [~,~,~,train_auc_arr(label)] = perfcurve(isY_test,score_train(:,label),labels_occur_train(label));
    end
    for label = 1:size(labels_occur_test,1)
        [~,~,~,test_auc_arr(label)] = perfcurve(Y_test,score_test(:,label),labels_occur_test(label));
    end
    train_auc_arr = train_auc_arr(2:6);
    test_auc_arr = test_auc_arr(2:6);


end