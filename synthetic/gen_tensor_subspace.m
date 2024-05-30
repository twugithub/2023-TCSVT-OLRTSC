function [ U, V, E ] = gen_tensor_subspace( n1, n3, all_d, all_n, rho )

p = n1*n3;
k = length(all_d);

U = cell(1,k);
V = cell(1,k);

U_all = orth_tensor(randn(n1,n1,n3));

start_idx = 1;

for kidx = 1:k
    d = all_d(kidx);
    n = all_n(kidx);
    end_idx = start_idx + d - 1;
    
    U{kidx} = U_all(:, start_idx:end_idx, :);
    V{kidx} = randn(n,d,n3);
    start_idx = end_idx + 1;
end

num_samples = sum(all_n);
num_elements = p * num_samples;
temp = randperm(num_elements);
numCorruptedEntries = round(rho * num_elements);
corruptedPositions = temp(1:numCorruptedEntries);
E = zeros(n1, num_samples, n3);
E(corruptedPositions) = 20 *(rand(numCorruptedEntries, 1) - 0.5);

end