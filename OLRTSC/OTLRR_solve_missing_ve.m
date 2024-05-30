function [ h, v, e ] = OTLRR_solve_missing_ve( z, D, lambda1, lambda2 )

[n1,d,n3] = size(D);
I = eye(d,d);

Dfft = fft(D,[],3);
aux = zeros(d,n1,n3);
for i = 1:n3
    aux(:,:,i) = (Dfft(:,:,i)' * Dfft(:,:,i) + n3/lambda1 * I) \ Dfft(:,:,i)';
end

thd = lambda2 / lambda1;

v = zeros(d,1,n3);
e = zeros(n1,1,n3);
y = zeros(n1,1,n3);
oldv = v;olde = e;

converged = false;
mu = 0.1;
rho = 1.9;
mu_max = 1e10;
maxIter = 100;
iter = 0;

zsq = squeeze(z);
Indicator = ~isnan(zsq(:));
missloc = isnan(zsq(:));

while ~converged
    iter = iter + 1;

    zysq = squeeze(mu*z - y);
    hsq = zeros(size(zsq));
    tmp = squeeze(tprod(D, v) + e);
    hsq(Indicator) = (lambda1 * tmp(Indicator) + zysq(Indicator)) / (lambda1 + mu);
    hsq(missloc) = tmp(missloc);
    h = reshape(hsq, [size(z,1), 1, size(z,3)]);

    a = h - e;
    afft = fft(a,[],3);
    vfft = zeros(d,1,n3);
    for i = 1:n3
        vfft(:,:,i) = aux(:,:,i)*afft(:,:,i);
    end
    v = ifft(vfft,[],3);
    e = sign(h - tprod(D, v)) .* max(abs(h - tprod(D, v)) - thd, 0);
    dd = squeeze(h - z);
    dd(missloc) = 0;

    stopc = max([sqrt(sum(sum(dd.^2))), sqrt(sum(sum((v - oldv).^2))), sqrt(sum(sum((e - olde).^2)))]) / (n1*n3);

    if stopc < 1e-6 || iter > maxIter
        converged = true;
    else
        ysq = squeeze(y);
        hsq = squeeze(hsq);
        ysq(Indicator) = ysq(Indicator) + mu*(hsq(Indicator) - zsq(Indicator));
        y = reshape(ysq, [size(z,1), 1, size(z,3)]);
        mu = min(rho*mu, mu_max);
        oldv = v;olde = e;
    end
end

end