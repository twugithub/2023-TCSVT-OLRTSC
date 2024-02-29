function [ EV ] = compute_EV( L, U )

UUt = tprod(U, tran(U));
LLt = tprod(L, tran(L));
traceLLt = trace(LLt(:,:,1));

coh = tprod(tran(L), UUt);
coh = tprod(coh, L);
coh = trace(coh(:,:,1));

EV = coh / traceLLt;

end