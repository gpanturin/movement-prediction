function MI=mutual_information2(x,y)

n=size(x,1);
Y=unique(y);
values_of_x=[];
P=[];
x_ind=0;
for r=1:size(x,1)
    current_x=x(r,:);
    if ~is_x_in_values_of_x(current_x,values_of_x)
        x_ind=x_ind+1;
        values_of_x=[values_of_x;current_x];
        P=[P;zeros(1,length(Y))];
        for rr=r:size(x,1)
            if x(rr,:)==current_x
                y_ind=find(Y==y(rr));
                P(x_ind,y_ind)=P(x_ind,y_ind)+1;
            end
        end
    end
end
% P=P/n;
sum1=sum(sum(P(P~=0).*log2(P(P~=0))));
sum2=sum(sum(P,2).*log2(sum(P,2)));
sum3=sum(sum(P,1).*log2(sum(P,1)));

MI = log2(n) + (sum1 - sum2 - sum3)/n;
end
