clear; close all; clc;

file_fornberg = 'butterfly_fornberg_noise_error.fig';
file_csf      = 'butterfly_csf_noise_error.fig';

if ~exist(file_fornberg, 'file') || ~exist(file_csf, 'file')
    error('Error: Specified .fig files not found. Ensure they are in the current directory.');
end

fprintf('Extracting data from figure files...\n');

data_fly = extract_data_from_fig(file_fornberg);
data_csf = extract_data_from_fig(file_csf);

x_fly_pct = data_fly.x * 100;
x_csf_pct = data_csf.x * 100;

fprintf('   Data extraction complete. X-axis range: %.1f%% - %.1f%%\n', min(x_fly_pct), max(x_fly_pct));

figure('Color', 'w', 'Name', 'Robustness Comparison: Butterfly', 'Position', [100, 100, 680, 500]);
hold on; box on;

errorbar(x_fly_pct, data_fly.y, data_fly.err, 'o-', ...
    'LineWidth', 1.5, ...
    'Color', '#0072BD', ...          
    'MarkerFaceColor', '#0072BD', ...
    'CapSize', 8, ...
    'DisplayName', 'Fornberg Conformal');

errorbar(x_csf_pct, data_csf.y, data_csf.err, 's-', ...
    'LineWidth', 1.5, ...
    'Color', '#D95319', ...          
    'MarkerFaceColor', '#D95319', ...
    'CapSize', 8, ...
    'DisplayName', 'CSF-QCM (Proposed)');

grid on;
legend('Location', 'NorthWest', 'FontSize', 11);

xlabel('Input Noise Amplitude (%)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Reconstruction RMS Error (\lambda)', 'FontSize', 12, 'FontWeight', 'bold');

title({'Robustness Comparison (Butterfly)', 'Speckle Noise Model (Monte Carlo N=50)'}, ...
    'FontSize', 13);

xlim([4, 21]); 
ylim([0, 0.06]);

hold off;

function data = extract_data_from_fig(filename)
    fig_handle = openfig(filename, 'invisible');
    h_err = findobj(fig_handle, 'Type', 'ErrorBar');
    
    if isempty(h_err)
        h_line = findobj(fig_handle, 'Type', 'Line');
        if isempty(h_line)
            close(fig_handle); error('No data found in file %s.', filename);
        end
        [~, idx] = max(arrayfun(@(x) length(x.XData), h_line));
        target = h_line(idx);
        data.x = target.XData; data.y = target.YData; data.err = zeros(size(data.y));
    else
        [~, idx] = max(arrayfun(@(x) length(x.XData), h_err));
        target = h_err(idx);
        data.x = target.XData; data.y = target.YData; data.err = target.YPositiveDelta;
    end
    close(fig_handle);
end
