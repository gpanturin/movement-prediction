function [Xv, vff_max, vff_mean, vft_max, vft_mean] = features_vetting_fit(X, Y)
    K_neighbors = 6;
    Xv=X;  
    % Calculate Spearman correlation for each feature
    corr_spearman = corr(Xv,'Type', 'Spearman');
    corr_spearman = tril(corr_spearman,-1);
    % Calculate feature importance using ReliefF algorithm
    [ranking, scores] = relieff(Xv, Y', K_neighbors);
    
   
    % Select top 20 features based on scores
    selected_features = ranking(1:20);
    
    % Loop until all selected features have Spearman correlation below 0.8
    while true
      high_corr_features = find(abs(corr_spearman(selected_features)) > 0.8);
      if isempty(high_corr_features)
        break;
      end
      
      % Select feature with smallest ReliefF score
      [~, min_index] = min(scores(selected_features(high_corr_features)));
      selected_features(high_corr_features(min_index)) = [];
    end
    Xv = Xv(:,selected_features); 
    

    rff_Spearman = abs(corr(Xv,'type','Spearman'));    
    rff_Spearman_Low_diag = tril(rff_Spearman,-1);

    vff_max = max(max(rff_Spearman_Low_diag));      
    vff_mean = mean(mean(rff_Spearman_Low_diag));  

    %releiff
    [idx, weights] = relieff(Xv,Y',6);   

    vft_max = weights(idx(1));
    weight_no_nan = weights;
    weight_no_nan(isnan(weights)) = [];
    vft_mean = mean(weight_no_nan);

    disp('----------')
    disp(['features after vetting: ',num2str(sort(selected_features))]);
    disp('----------')

end