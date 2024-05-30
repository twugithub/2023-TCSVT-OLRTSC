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

perm = randperm(n);
Z = Z_noise(:,perm,:);

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
        fprintf('OLRTSC: access sample %d\n', t);
    end

    z = Z(:,t,:);
    lambda3 = sqrt(t) * lambda3_base;

    [v, e] = OTLRR_solve_ve(z, D, lambda1, lambda2);

    zfft = fft(z,[],3);
    Dfft = fft(D,[],3);
    Mfft = fft(M,[],3);
    ufft = zeros(d,1,n3);
    for i = 1:n3
        normz = norm(zfft(:,:,i));
        ufft(:,:,i) = (Dfft(:,:,i) - Mfft(:,:,i))' * zfft(:,:,i) / (normz * normz + n3/lambda3);
    end
    u = ifft(ufft,[],3);

    M = M + tprod(z, tran(u));
    A = A + tprod(v, tran(v));
    B = B + tprod(z-e, tran(v));

    D = OTLRR_solve_D(D, M, A, B, lambda1, lambda3);

    U(t,:,:) = tran(u);
    V(t,:,:) = tran(v);
    E(:,t,:) = e;

end

toc

D_p = orth_tensor(D);
ev = compute_EV(D_p, U_all);