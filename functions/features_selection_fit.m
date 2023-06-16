function [Xs_train, n_combinations, nft_max, nft_mean, nft_std, best_comb] = features_selection_fit(Xv, Y)
    
    [Xv_discret,~] = discretize(Xv,3); 
    % creating all possible combinations
    C = nchoosek(1:size(Xv_discret,2),10);
    MI = zeros(size(C,1),1);
    n_combinations = size(C,1);

    for comb = 1:size(C,1)
        curr_comb = zeros(size(Xv_discret,1),size(C,2));
        for feat = 1:size(C,2)                                                    
            curr_comb(:,feat) = Xv_discret(:,C(comb,feat));       % creating back the feature matrix of the current 10 feature cimbination
        end
        MI(comb) = mutual_information2(curr_comb,Y);    %calculating MI for the current combination 
    end
    
    [max_MI,best_comb_idx] = max(MI);
    %returnung parameters
    best_comb = C(best_comb_idx,:);
    nft_max = max_MI;
    nft_mean = mean(MI);
    nft_std = std(MI);
    
    % creating features matrix from the best 10 features combinatin
    Xs_train = zeros(size(Xv_discret,1),size(C,2)); 
    for feature = 1:size(C,2)
        Xs_train(:,feature) = Xv(:,C(best_comb_idx,feature));
    end
end