clc;clear;
load('T-61_5130_data_3.mat');
[n_r, n_c]  = size(cancerInputs);
train_len = ceil(n_r*3/4);
test_len =  n_r - train_len; 
t_set = randperm(n_r,train_len);
training_set = cancerInputs(t_set,:); 
te_set = setdiff(1:n_r,t_set);
test_set = cancerInputs(te_set,:);
for no_centers=2:20
    fprintf('No of Centers %d\n',no_centers);
    phi = zeros(train_len,no_centers);
    phi_test = zeros(test_len,no_centers);
    centers = zeros(no_centers,n_c);
    for i = 1:no_centers
        random_center = ceil(rand()*train_len);
        centers(i,:) = training_set(random_center,:);
    end
    max_dist = 0;
    for i = 1:no_centers
        for j = i:no_centers
            dist = (centers(i,:) - centers(j,:)) * (centers(i,:) - centers(j,:))';
            if(dist > max_dist)
                max_dist = dist;
            end
        end    
    end
    sigma = max_dist/sqrt(no_centers);
    for i = 1:no_centers
        for j =1:train_len
            dist = (centers(i,:) - training_set(j,:)) * (centers(i,:) - training_set(j,:))';
            phi(j,i) = exp(-(dist/sigma^2));
        end
    end
    for i = 1:no_centers
        for j =1:test_len
            dist = (centers(i,:) - test_set(j,:)) * (centers(i,:) - test_set(j,:))';
            phi_test(j,i) = exp(-(dist/sigma^2));
        end
    end
    weights = pinv(phi)*cancerTargets(t_set,:);
    y_train = phi*weights;
    for j =1:train_len
       m = find(y_train(j,:)==max(y_train(j,:)));
       y_train(j,:) = 0;
       y_train(j,m) = 1;
    end
    accuracy_train = sum(y_train(:,1) == cancerTargets(t_set,1));
    fprintf('Prediction accuracy on Train set %f\n',accuracy_train*100/train_len);
    y_test = phi_test*weights;
    for j =1:test_len
       m = find(y_test(j,:)==max(y_test(j,:)));
       y_test(j,:) = 0;
       y_test(j,m) = 1;
    end
    accuracy_test = sum(y_test(:,1) == cancerTargets(te_set,1));
    fprintf('Prediction accuracy on Test set %f\n',accuracy_test*100/test_len);
end