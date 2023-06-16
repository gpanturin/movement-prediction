function [Xv, vff_max, vff_mean, vft_max, vft_mean] = features_vetting_transform(X, Y)
    %% removing features according to spearman on train set
    Xv = X; 
    K_neighbors = 6;
%     Xv(:,worst_idx_spearman) = [];
% 
%     %% leaving the best 20 features according to relieff on train set  
%     Xv = Xv(:,best_20_relieff); 
    %Xv = Xv(:,[16 28 29 30 49 64 65 66 67 68 70 87 91 92 93 94 95 96 140 144]);
    Xv = Xv(:,sort([ 3    9   13   16   49   55   85   87   91   92   93   94   95   96  146  155  157  169  170  171]));
    %% returning max and mean values after feature vetting
    %spearman
    rff_Spearman = abs(corr(Xv,'type','Spearman'));    
    rff_Spearman_Low_diag = tril(rff_Spearman,-1);

    vff_max = max(max(rff_Spearman_Low_diag));      
    vff_mean = mean(mean(rff_Spearman_Low_diag));  

    %releiff
    [idx, weights] = relieff(Xv,Y',K_neighbors);   

    vft_max = weights(idx(1));
    weight_no_nan = weights;
    weight_no_nan(isnan(weights)) = [];
    vft_mean = mean(weight_no_nan);

end
% ,worst_idx_spearman,best_20_relieff