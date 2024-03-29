
A = [3,2,1; 2,4,2; 1,2,5]
try chol(A);
    disp('Matrix is symmetric positive definite.')
catch ME
    disp('Matrix is not symmetric positive definite')
end


[L1,D1] = ldl(A);
M1 = L1 * D1 * L1.'

% GLSL LDL IMPL
L2 = [1, 0, 0; 0.666667, 1, 0; 0.3333333, 0.5, 1];
D2 = diag([3.0, 2.666667, 4.0]);

M2 = L2 * D2 * L2.'

try chol(M2);
    disp('Matrix is symmetric positive definite.')
catch ME
    disp('Matrix is not symmetric positive definite')
end
% GLSL GMW81 IMPL

M3 = [0.9,2,1; 2,4.4889,2; 1,2,5];
try chol(M3);
    disp('Matrix is symmetric positive definite.')
catch ME
    disp('Matrix is not symmetric positive definite')
end

x = [1;1;1];

sol1 = A \ x
sol2 = M2 \ x

