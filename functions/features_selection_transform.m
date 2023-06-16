function [Xs_test, nft] = features_selection_transform(Xv, Y)
    features_before_selection = [3    9   13   16   49   55   85   87   91   92   93   94   95   96  146  155  157  169  170  171];
    Xs_test = Xv(:,[1,2,3,4,5,6,7,8,13,19]);
    [Xv_discret,~] = discretize(Xs_test,3); 
    nft = mutual_information2(Xv_discret,Y);
    final_features = features_before_selection([1,2,3,4,5,6,7,8,13,19]);
    
    disp('-----------------')
    disp(['final features after selection are: ',num2str(sort(final_features))])
    disp('-----------------')
end
