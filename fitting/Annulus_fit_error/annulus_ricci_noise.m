function annulus_ricci_noise()
clear; clc; close all;

fprintf('[Step 1] Generating Wavefront (Annulus)...\n');
N = 300; 
x = linspace(-1.6, 1.6, N); y = linspace(-1.6, 1.6, N);
[X, Y] = meshgrid(x, y);
[Theta, R] = cart2pol(X, Y);

r_out_b = 1.0 + 0.05*sin(3*Theta) + 0.04*cos(5*Theta);
[Theta_in, R_in_local] = cart2pol(X - 0.4, Y + 0.1); 
r_in_b = 0.40 + 0.02*sin(4*Theta_in);
mask = (R <= r_out_b) & (R_in_local > r_in_b);
mask_eval = mask;

z_modes = 36;
coeffs_true = (rand(z_modes, 1) - 0.5) .* (1./(1:z_modes)');
coeffs_true(1:3) = 0;
W_raw = zeros(size(X));
for j = 1:z_modes
    W_raw = W_raw + coeffs_true(j) * zernike_std(j, R, Theta);
end

vals = W_raw(mask_eval); 
vals = vals - mean(vals);
target_val = 0.09;
scale_factor = target_val / std(vals); 
W_true = nan(size(W_raw)); 
W_true(mask_eval) = vals * scale_factor; 
vals_final = W_true(mask_eval);
fprintf('   Ground Truth RMS: %.4f lambda\n', std(vals_final));
fprintf('   Ground Truth PV : %.4f lambda\n', max(vals_final) - min(vals_final));

fprintf('[Step 2] Meshing & Ricci Flow Flattening...\n');
[nodes, tri, bdy_out_idx, bdy_in_idx] = build_mesh(); 
n_nodes = size(nodes, 1); 

L = compute_edge_lengths(nodes, tri);
u = zeros(n_nodes, 1);
is_bdy = false(n_nodes, 1); is_bdy([bdy_out_idx, bdy_in_idx]) = true;

for iter = 1:50
    [K_curr, L_cot, ~] = compute_curv_metric_proper(tri, L, u, n_nodes, is_bdy);
    delta_K = -K_curr; 
    delta_K(is_bdy) = 0; 
    if norm(delta_K, inf) < 1e-2, break; end 
    fixed_id = 1; 
    H_mat = L_cot; H_mat(fixed_id, :) = 0; H_mat(:, fixed_id) = 0; H_mat(fixed_id, fixed_id) = 1;
    delta_K(fixed_id) = 0;
    du = H_mat \ delta_K;
    u = u + 0.5 * du; 
end
[~, ~, L_flat] = compute_curv_metric_proper(tri, L, u, n_nodes, is_bdy);

fprintf('[Step 3] Cutting and Flattening to UV...\n');
pts_out = nodes(bdy_out_idx, :); pts_in  = nodes(bdy_in_idx, :);
dist_mat = pdist2(pts_out, pts_in);
[~, min_idx] = min(dist_mat(:));
[r, c] = ind2sub(size(dist_mat), min_idx);
start_node = bdy_out_idx(r); end_node = bdy_in_idx(c);

E_all = [tri(:,1), tri(:,2); tri(:,2), tri(:,3); tri(:,3), tri(:,1)];
W_safe = [L.l3; L.l1; L.l2]; 
G_graph = graph(E_all(:,1), E_all(:,2), W_safe);
cut_path = shortestpath(G_graph, start_node, end_node);
cut_edges = sort([cut_path(1:end-1)', cut_path(2:end)'], 2);

uv_tri = flatten_mesh_naive(tri, nodes, L_flat, cut_edges);

fprintf('[Step 4] Mapping & Building Matrix...\n');
uv_points_all = reshape(uv_tri, [], 2); 
coeff_pca = pca(uv_points_all); 
uv_rot_all = uv_points_all * coeff_pca;
u_rot = uv_rot_all(:,1); v_rot = uv_rot_all(:,2);
range_u = range(u_rot); range_v = range(v_rot);

if range_v > range_u 
    temp = u_rot; u_rot = v_rot; v_rot = temp;
    range_u = range(v_rot); range_v = range(u_rot);
end

pix_idx = find(mask);
TR = triangulation(tri, nodes);
ti = pointLocation(TR, [X(pix_idx), Y(pix_idx)]);
valid_mask = ~isnan(ti);
valid_ti = ti(valid_mask);
valid_pix_idx = pix_idx(valid_mask);
bc = cartesianToBarycentric(TR, valid_ti, [X(valid_pix_idx), Y(valid_pix_idx)]);

pts_uv_pixel = zeros(length(valid_pix_idx), 2);
for k = 1:length(valid_pix_idx)
    tri_id = valid_ti(k);
    tri_uvs = squeeze(uv_tri(tri_id, :, :)); 
    pts_uv_pixel(k, :) = bc(k, :) * tri_uvs;
end

pts_uv_rot = pts_uv_pixel * coeff_pca;
uv_check = uv_points_all * coeff_pca;
if range(uv_check(:,2)) > range(uv_check(:,1))
     pts_uv_rot = [pts_uv_rot(:,2), pts_uv_rot(:,1)];
end

theta_fit = (pts_uv_rot(:,1) - min(u_rot)) / range_u * 2 * pi;
phi_fit   = (pts_uv_rot(:,2) - min(v_rot)) / range_v;
epsilon_rec = 0.4; 
rho_fit   = epsilon_rec + (1-epsilon_rec) * phi_fit;
theta_fit = max(0, min(2*pi, theta_fit));
rho_fit = max(0, min(1, rho_fit));

max_noll = 36;
H = zeros(length(rho_fit), max_noll);
for j=1:max_noll
    H(:,j) = zernike_std(j, rho_fit, theta_fit);
end

fprintf('[Step 5] Monte Carlo Noise Scanning (N=50)...\n');
w_clean = W_true(valid_pix_idx);
noise_levels = linspace(0.05, 0.20, 15);
noise_pct = noise_levels * 100;
n_trials = 50;
res_ricci_mean = zeros(length(noise_levels), 1);
res_ricci_std  = zeros(length(noise_levels), 1);

U_ideal = exp(1i * w_clean);

fprintf('   Progress: ');
for k = 1:length(noise_levels)
    sigma = noise_levels(k);
    errs_trial = zeros(n_trials, 1);
    
    if mod(k,2)==0, fprintf('%.0f%% ', k/length(noise_levels)*100); else, fprintf('.'); end
    
    for t = 1:n_trials
        n_real = sigma * randn(size(w_clean));
        n_imag = sigma * randn(size(w_clean));
        U_noise = (n_real + 1i * n_imag) / sqrt(2);
        
        U_total = U_ideal + U_noise;
        phase_noise = angle(U_total ./ U_ideal);
        w_noisy = w_clean + phase_noise;
        
        c_est = H \ w_noisy;
        w_rec = H * c_est;
        diff_vec = w_clean - w_rec;
        errs_trial(t) = sqrt(mean(diff_vec.^2));
    end
    
    res_ricci_mean(k) = mean(errs_trial);
    res_ricci_std(k)  = std(errs_trial);
end
fprintf('\n');

save('annulus_ricci_noise.mat', 'noise_levels', 'res_ricci_mean', 'res_ricci_std');

figure('Color','w', 'Position', [200, 200, 600, 500]);
errorbar(noise_levels, res_ricci_mean, res_ricci_std, '-o', 'LineWidth', 1.5, ...
    'MarkerSize', 6, 'MarkerFaceColor', 'b', 'CapSize', 10);
grid on;
xlabel('Input Noise Level (\sigma_{phasor})', 'FontSize', 12);
ylabel('Reconstruction RMSE (\lambda)', 'FontSize', 12);
title('Impact of Noise on Cut-Domain Fitting (No Stitching)', 'FontSize', 14);
subtitle(['Monte Carlo N=' num2str(n_trials) ', Zernike Modes=' num2str(max_noll)]);
xlim([min(noise_levels)-0.02, max(noise_levels)+0.02]);

text(noise_levels(1), res_ricci_mean(1)*1.1, sprintf('Min Err: %.3f', res_ricci_mean(1)), 'FontSize', 10);
text(noise_levels(end), res_ricci_mean(end)*0.9, sprintf('Max Err: %.3f', res_ricci_mean(end)), 'FontSize', 10);

end

function [nodes, tri, idx_out, idx_in] = build_mesh()
    t=linspace(0,2*pi,120); t(end)=[];
    ro=1+.05*sin(3*t)+.04*cos(5*t); xo=ro.*cos(t); yo=ro.*sin(t);
    ri=.4+.02*sin(4*t); xi=ri.*cos(t)+.4; yi=ri.*sin(t)-.1;
    [px,py]=meshgrid(linspace(-1.5,1.5,35)); px=px(:); py=py(:);
    valid = inpolygon(px,py,xo,yo) & ~inpolygon(px,py,xi,yi);
    nodes=[xo' yo'; xi' yi'; px(valid) py(valid)];
    idx_out=1:length(t); idx_in=(length(t)+1):(2*length(t));
    DT=delaunayTriangulation(nodes); c=incenter(DT);
    tri=DT.ConnectivityList(inpolygon(c(:,1),c(:,2),xo,yo) & ~inpolygon(c(:,1),c(:,2),xi,yi), :);
end

function L0 = compute_edge_lengths(XY, tri)
    p1=XY(tri(:,1),:); p2=XY(tri(:,2),:); p3=XY(tri(:,3),:);
    L0.l1=sqrt(sum((p2-p3).^2,2)); L0.l2=sqrt(sum((p1-p3).^2,2)); L0.l3=sqrt(sum((p1-p2).^2,2));
end

function [K, L_cot, L_new] = compute_curv_metric_proper(tri, L0, u, n, is_bdy)
    i1=tri(:,1); i2=tri(:,2); i3=tri(:,3);
    l1=L0.l1.*exp((u(i2)+u(i3))/2); l2=L0.l2.*exp((u(i1)+u(i3))/2); l3=L0.l3.*exp((u(i1)+u(i2))/2);
    L_new=struct('l1',l1,'l2',l2,'l3',l3);
    f=@(a,b,c) max(-1,min(1,(b.^2+c.^2-a.^2)./(2.*b.*c)));
    ang1=acos(f(l1,l2,l3)); ang2=acos(f(l2,l1,l3)); ang3=acos(f(l3,l1,l2));
    cot1=1./tan(ang1); cot2=1./tan(ang2); cot3=1./tan(ang3);
    I=[i2;i3;i3;i1;i1;i2]; J=[i3;i2;i1;i3;i2;i1]; V=0.5*[cot1;cot1;cot2;cot2;cot3;cot3];
    L_cot=sparse(I,J,-V,n,n); L_cot=L_cot+sparse(1:n,1:n,-sum(L_cot,2),n,n);
    ang_sum = sparse(tri(:), 1, [ang1; ang2; ang3], n, 1);
    K = zeros(n, 1); K(~is_bdy) = 2*pi - ang_sum(~is_bdy); K(is_bdy)  = pi - ang_sum(is_bdy);
end

function uv_tri = flatten_mesh_naive(tri, nodes, L_flat, cut_edges)
    n_tri = size(tri, 1);
    uv_tri = zeros(n_tri, 3, 2); 
    tri_visited = false(n_tri, 1);
    neighbors_list = neighbors(triangulation(tri, nodes));

    seed_id = 1;
    l1=L_flat.l1(seed_id); l2=L_flat.l2(seed_id); l3=L_flat.l3(seed_id);
    p1=[0,0]; p2=[l3,0];
    cos_p1 = max(-1, min(1, (l3^2+l2^2-l1^2)/(2*l3*l2)));
    p3=[l2*cos_p1, l2*sqrt(1-cos_p1^2)];
    uv_tri(seed_id,:,:) = [p1; p2; p3];
    tri_visited(seed_id) = true;
    queue = seed_id; head = 1;
    while head <= length(queue)
        curr_t = queue(head); head = head + 1;
        curr_uvs = squeeze(uv_tri(curr_t,:,:));
        curr_nodes = tri(curr_t,:);
        for k=1:3
            next_t = neighbors_list(curr_t, k);
            if isnan(next_t) || tri_visited(next_t), continue; end
            if k==1, e_nodes=sort([curr_nodes(2),curr_nodes(3)]);
            elseif k==2, e_nodes=sort([curr_nodes(1),curr_nodes(3)]);
            else, e_nodes=sort([curr_nodes(1),curr_nodes(2)]); end
            if ismember(e_nodes, cut_edges, 'rows'), continue; end 
            
            [~, loc_next] = ismember(e_nodes, tri(next_t,:));
            new_v_idx = setdiff(1:3, loc_next);
            [~, loc_curr] = ismember(e_nodes, curr_nodes);
            A_uv = curr_uvs(loc_curr(1),:); B_uv = curr_uvs(loc_curr(2),:);
            l_next = [L_flat.l1(next_t), L_flat.l2(next_t), L_flat.l3(next_t)];
            get_l = @(i,j) l_next(setdiff(1:3,[i,j]));
            r_A = get_l(loc_next(1), new_v_idx); r_B = get_l(loc_next(2), new_v_idx);
            d_AB = norm(A_uv - B_uv);
            ang = acos(max(-1, min(1, (d_AB^2+r_A^2-r_B^2)/(2*d_AB*r_A))));
            vc_A=complex(A_uv(1),A_uv(2)); vc_B=complex(B_uv(1),B_uv(2));
            is_ccw = (loc_next(2) == mod(loc_next(1),3)+1);
            offset = r_A * exp((1i*ang)*(2*is_ccw-1)) * ((vc_B-vc_A)/d_AB);
            new_uv = [real(vc_A+offset), imag(vc_A+offset)];
            uv_next=zeros(3,2); uv_next(loc_next(1),:)=A_uv; uv_next(loc_next(2),:)=B_uv; uv_next(new_v_idx,:)=new_uv;
            uv_tri(next_t,:,:) = uv_next;
            tri_visited(next_t) = true;
            queue(end+1) = next_t;
        end
    end
end

function Z = zernike_std(j, r, t)
    [n, m] = get_noll_nm(j); R_nl = zeros(size(r));
    for k = 0:(n-abs(m))/2
        num = (-1)^k * factorial(n-k);
        den = factorial(k) * factorial((n+m)/2 - k) * factorial((n-m)/2 - k);
        R_nl = R_nl + (num/den) * r.^(n-2*k);
    end
    if m >= 0, Z = R_nl .* cos(m*t); else, Z = R_nl .* sin(abs(m)*t); end
end

function [n, m] = get_noll_nm(j)
    n_list = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7];
    m_list = [0, 1, -1, 0, -2, 2, -1, 1, -3, 3, 0, -2, 2, -4, 4, -1, 1, -3, 3, -5, 5, 0, -2, 2, -4, 4, -6, 6, -1, 1, -3, 3, -5, 5, -7, 7];
    if j <= length(n_list), n=n_list(j); m=m_list(j); else, n=0; m=0; end
end
