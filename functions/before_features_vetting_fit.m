function [vff_max, vff_mean, vft_max, vft_mean] = before_features_vetting_fit(X, Y)
% vff_max - feature-feature maximum correlation value
% vff_mean - feature-feature average correlation value
% vft_max - feature-target maximum Relieff value
% vft_mean - feature-target average Relieff value

    rff_Spearman = abs(corr(X,'type','Spearman'));
%     for i = 1:size(rff_Spearman,1)
%         rff_Spearman(i,i)=0;
%     end
  
    rff_Spearman_Low_diag = tril(rff_Spearman,-1);
    vff_max = max(max(rff_Spearman_Low_diag));
    vff_mean = mean(mean(rff_Spearman_Low_diag));
    [idx, weights] = relieff(X,Y,10,'method','classification');
    vft_max = weights(idx(1));
    weight_no_nan = weights;
    weight_no_nan(isnan(weights)) = [];
    vft_mean = mean(weight_no_nan);

end