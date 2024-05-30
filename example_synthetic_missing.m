clearvars;
close all;
clc

rng('shuffle');
addpath('synthetic');
addpath('OLRTSC');

n1 = 50;
n3 = 20;
clustern = 5;
rank_ratio = 0.1;
p = n1*n3;

all_n = 300 * ones(clustern, 1);
all_d = round(n1 * ones(clustern, 1) * rank_ratio);
rho = 0.3;

[U_gt, V_gt, E] = gen_tensor_subspace(n1, n3, all_d, all_n, rho);

d = sum(all_d);
U_all = zeros(n1,d,n3);
for kidx = 1:clustern
    U_all(:, all_d(1)*(kidx-1)+1:all_d(1)*kidx, :) = U_gt{kidx};
end

end_idx = cumsum(all_n);
start_idx = end_idx - all_n + 1;
n = sum(all_n);
gt = zeros(n,1);

for kidx = 1:clustern
    gt(start_idx(kidx):end_idx(kidx)) = kidx;
end

Z_clean = zeros(n1,n,n3);
for kidx = 1:clustern
    Z_clean(:, start_idx(kidx):end_idx(kidx), :) = tprod(U_gt{kidx}, tran(V_gt{kidx}));
end
Z_noise = Z_clean + E;

eta = 0.8;
missingtensordata = NaN(size(Z_noise));

for i = 1:size(Z_noise,2)
    tempdata = squeeze(Z_noise(:,i,:));
    chosen = randperm(size(tempdata,1)*size(tempdata,2), round(eta*size(tempdata,1)*size(tempdata,2)));
    temp2 = NaN(size(tempdata,1),size(tempdata,2));
    temp2(chosen) = tempdata(chosen);
    missingtensordata(:,i,:) = reshape(temp2, [size(tempdata,1),1,size(tempdata,2)]);
end

perm = randperm(n);
missingtensordata = missingtensordata(:,perm,:);

lambda1 = 1;
lambda2 = 1/sqrt(n1*n3);
lambda3_base = 1/sqrt(n1*n3);

M = zeros(n1,d,n3);
A = zeros(d,d,n3);
B = zeros(n1,d,n3);
U = zeros(n,d,n3);
V = zeros(n,d,n3);
E = zeros(n1,n,n3);

D = randn(n1,d,n3);

tic

for t = 1:n

    if mod(t, 500) == 0
        fprintf('rOLRTSC: access sample %d\n', t);
    end

    z = missingtensordata(:,t,:);
    lambda3 = sqrt(t) * lambda3_base;

    [w, v, e] = OTLRR_solve_missing_ve(z, D, lambda1, lambda2);

    wfft = fft(w,[],3);
    Dfft = fft(D,[],3);
    Mfft = fft(M,[],3);
    ufft = zeros(d,1,n3);
    for i = 1:n3
        normw = norm(wfft(:,:,i));
        ufft(:,:,i) = (Dfft(:,:,i) - Mfft(:,:,i))' * wfft(:,:,i) / (normw * normw + n3/lambda3);
    end
    u = ifft(ufft,[],3);

    M = M + tprod(w, tran(u));
    A = A + tprod(v, tran(v));
    B = B + tprod(w-e, tran(v));

    D = OTLRR_solve_D(D, M, A, B, lambda1, lambda3);

    U(t,:,:) = tran(u);
    V(t,:,:) = tran(v);
    E(:,t,:) = e;

end

toc

D_p = orth_tensor(D);
ev = compute_EV(D_p, U_all);