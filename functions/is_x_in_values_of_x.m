function j=is_x_in_values_of_x(x,values_of_x)
% j=True i.e. x is a part of the set values_of_x
% j=False i.e. x is not a part of the set values_of_x

if isempty(values_of_x)
    j=false;
else
    j=false;
    for r=1:size(values_of_x,1)
        if x==values_of_x(r,:)
            j=true;
        end
    end
end
end