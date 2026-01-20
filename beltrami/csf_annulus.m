clear; clc; close all;

fprintf('1. Initializing Geometry (Type III: Annulus)...\n');

N_bdy_base = 1000; 
t = linspace(0, 2*pi, N_bdy_base+1)'; t(end) = []; 

r_out = 1.0 + 0.05*sin(3*t) + 0.04*cos(5*t);
[x_out, y_out] = pol2cart(t, r_out);

r_in_local = 0.40 + 0.02*sin(4*t); 
[x_in_local, y_in_local] = pol2cart(t, r_in_local);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
shift_x = 0.40; 
shift_y = -0.10;
x_in = x_in_local + shift_x; 
y_in = y_in_local + shift_y;

fprintf('2. Generating High-Res Quasi-Uniform Mesh...\n');

n_radial = 24;   
n_angular = 180; 

[x_out_uniform, y_out_uniform] = reparam_curve(x_out, y_out, n_angular);
[x_in_uniform, y_in_uniform] = reparam_curve(x_in, y_in, n_angular);

r_vals = linspace(0, 1, n_radial)';
X_mesh = zeros(n_radial, n_angular+1);
Y_mesh = zeros(n_radial, n_angular+1);

for j = 1:n_angular+1
    P_in = [x_in_uniform(j), y_in_uniform(j)];
    P_out = [x_out_uniform(j), y_out_uniform(j)];
    
    X_mesh(:, j) = (1-r_vals) * P_in(1) + r_vals * P_out(1);
    Y_mesh(:, j) = (1-r_vals) * P_in(2) + r_vals * P_out(2);
end

fprintf('   Executing Iterative Smoothing (n=100)...\n');
n_smooth_iter = 100; 

for k = 1:n_smooth_iter
    X_new = X_mesh; Y_new = Y_mesh;
    
    for r = 2:n_radial-1
        for c = 1:n_angular+1
            c_prev = c - 1; if c_prev < 1, c_prev = n_angular; end
            c_next = c + 1; if c_next > n_angular+1, c_next = 2; end
            
            val_x = 0.25 * (X_mesh(r-1,c) + X_mesh(r+1,c) + X_mesh(r,c_prev) + X_mesh(r,c_next));
            val_y = 0.25 * (Y_mesh(r-1,c) + Y_mesh(r+1,c) + Y_mesh(r,c_prev) + Y_mesh(r,c_next));
            
            X_new(r,c) = val_x;
            Y_new(r,c) = val_y;
        end
    end
    X_mesh = X_new; Y_mesh = Y_new;
end

fprintf('3. Visualizing Result...\n');
figure('Color', 'w', 'Position', [100, 100, 800, 700]);
axis equal; axis off; hold on;

step_draw = 3; 
for j = 1:step_draw:n_angular
    plot(X_mesh(:, j), Y_mesh(:, j), 'r-', 'LineWidth', 0.8);
end

for i = 1:n_radial
    plot(X_mesh(i, :), Y_mesh(i, :), 'b-', 'LineWidth', 0.8);
end

plot(X_mesh(end, :), Y_mesh(end, :), 'k-', 'LineWidth', 1.5); 
plot(X_mesh(1, :),   Y_mesh(1, :),   'k-', 'LineWidth', 1.5); 

title('Generated Type III Mesh (CSF-QCM)', 'FontSize', 14);

outputFile = 'Annulus_mapping.mat';
fprintf('4. Saving data to %s...\n', outputFile);

save(outputFile, 'X_mesh', 'Y_mesh');

fprintf('Done! You can now run "beltrami_annulus.m".\n');

function [xn, yn] = reparam_curve(x, y, N)
    x = x(:); y = y(:);
    if (x(1) ~= x(end)) || (y(1) ~= y(end))
        x(end+1) = x(1); y(end+1) = y(1);
    end
    dx = diff(x); dy = diff(y);
    dist = sqrt(dx.^2 + dy.^2);
    s = [0; cumsum(dist)];
    s_targets = linspace(0, s(end), N+1);
    
    xn = interp1(s, x, s_targets, 'spline'); 
    yn = interp1(s, y, s_targets, 'spline');
end
