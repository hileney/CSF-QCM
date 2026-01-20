function plot_comparison_type3()
    clear; close all; clc;

    file_qcm   = 'Annulus_mapping.mat';
    file_ricci = 'ricci_area_stats.mat';
    
    if exist(file_qcm, 'file')
        data_q = load(file_qcm);
        if isfield(data_q, 'ratio_qcm')
            vals_qcm = data_q.ratio_qcm;
        elseif isfield(data_q, 'area_ratios')
            vals_qcm = data_q.area_ratios;
        else
            error('ratio_qcm not found');
        end
        fprintf('  [Loaded] CSF-QCM (Max Ratio: %.2f)\n', max(vals_qcm));
    else
        error('File not exist: %s', file_qcm);
    end
    
    if exist(file_ricci, 'file')
        data_r = load(file_ricci);
        if isfield(data_r, 'ratio_ricci')
            vals_ricci = data_r.ratio_ricci;
        elseif isfield(data_r, 'ratio_rf')
            vals_ricci = data_r.ratio_rf;
        elseif isfield(data_r, 'area_ratios')
            vals_ricci = data_r.area_ratios;
        else
            fnames = fieldnames(data_r);
            vals_ricci = data_r.(fnames{1});
         end
        fprintf('  [Loaded] Ricci Flow  (Max Ratio: %.2f)\n', max(vals_ricci));
    else
        error('File not exist: %s', file_ricci);
    end

    [x_qcm_smooth, y_qcm]     = process_ecdf(vals_qcm);
    [x_ricci_smooth, y_ricci] = process_ecdf(vals_ricci);

    figure('Position', [150, 150, 800, 600], 'Color', 'w');
    
    semilogx(x_ricci_smooth, y_ricci, 'b-', 'LineWidth', 3); hold on;
    semilogx(x_qcm_smooth, y_qcm, 'r-', 'LineWidth', 3);
    
    grid on;
    set(gca, 'FontSize', 15, 'LineWidth', 1.5, 'TickDir', 'out');
    set(gca, 'XScale', 'log'); 
    set(gca, 'XMinorGrid', 'on', 'YMinorGrid', 'on');
    
    xlabel('Normalized Area Ratio ($A_i / \bar{A}$) [Log Scale]', ...
        'Interpreter', 'latex', 'FontSize', 16);
    ylabel('Cumulative Probability (ECDF)', 'FontSize', 16);
    title('Ricci Flow vs. CSF-QCM','FontSize', 18);
   
    legend({'Ricci Flow Mapping','CSF-QCM (Proposed)'}, ...
        'Location', 'southeast', 'FontSize', 14, 'Box', 'on');
    
    max_r = max(vals_ricci);
    max_q = max(vals_qcm);
    ratio_val = max_r / max_q; 
    
    stats_msg = {'\bf Statistics Comparison \rm'};
    stats_msg{end+1} = sprintf('\\color{blue}Ricci Flow Max: %.1f', max_r);
    stats_msg{end+1} = sprintf('\\color{red}CSF-QCM Max:  %.2f', max_q);
    stats_msg{end+1} = sprintf('\\color{black}Improvement:   %.3f', ratio_val);
    
    text(0.05, 0.5, stats_msg, 'Units', 'normalized', ...
        'BackgroundColor', 'w', 'EdgeColor', 'k', ...
        'Margin', 6, 'FontSize', 16, 'Interpreter', 'tex');
    
    xlim([0.01, 10]); 
    ylim([0, 1.01]); 
end

function [x_smooth, y] = process_ecdf(data)
    x_raw = sort(data(:)); 
    y = (1:length(x_raw))' / length(x_raw);
    window_size = round(length(x_raw) * 0.02);
    if window_size < 3, window_size = 3; end
    x_smooth = smoothdata(x_raw, 'gaussian', window_size);
end
