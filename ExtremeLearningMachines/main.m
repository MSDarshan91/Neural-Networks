clc;clear;
load('T-61_5130_data_2.mat');
[n_r, n_c]  = size(cancerInputs);
train_len = ceil(n_r*3/4);
test_len =  n_r - train_len; 
t_set = randperm(n_r,train_len);
training_set = [ones(train_len,1) cancerInputs(t_set,:)]; 
te_set = setdiff(1:n_r,t_set);
test_set = [ones(test_len,1) cancerInputs(te_set,:)];
for no_hidden_nodes=2:50
    fprintf('No of Hidden Neurons %d\n',no_hidden_nodes);
    random_weights = rand(no_hidden_nodes,(n_c+1));
    H = tanh(training_set*random_weights');
    H_test = tanh(test_set*random_weights');
    B = pinv(H)*cancerTargets(t_set,:);
    y_train = H*B;
    for j =1:train_len
       m = find(y_train(j,:)==max(y_train(j,:)));
       y_train(j,:) = 0;
       y_train(j,m) = 1;
    end
    accuracy_train = sum(all(y_train==cancerTargets(t_set,:),2));
    fprintf('Prediction accuracy on Train set %f\n',accuracy_train*100/train_len);
    y_test = H_test*B;
    for j =1:test_len
       m = find(y_test(j,:)==max(y_test(j,:)));
       y_test(j,:) = 0;
       y_test(j,m) = 1;
    end
    accuracy_test = sum(all(y_test==cancerTargets(te_set,:),2));
    fprintf('Prediction accuracy on Test set %f\n',accuracy_test*100/test_len);
end