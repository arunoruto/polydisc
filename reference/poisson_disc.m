function [points, pointRadii] = poissondisc(varargin)
% Generates evenly spread points with specified radii as distance or from a
% distribution specified in as a char vector in radii
% - space              [n x 1]: size of the hypercube in each dimension. If spherical is specified as
%                               each element represents the size of the half axis in each dimension
%                               (all equal=hypersphere).
% - radii                     : can be either a struct or an array of numbers.
%                               Numbers: the array will be sampled uniformly for the radius around 
%                               the point.
%                               Struct: provide a distribution to be sampled using the random function.
%                               radii.name         -> random('name', ...) name of the distribution
%                               radii.rmax         -> max radii
%                               radii.rmin         -> min radii
%                               radii.A and so on  -> additional paramater for the distribution
%                               See: https://mathworks.com/help/stats/prob.normaldistribution.random.html
% - spherical           [bool]: put the points in a spherical container instead of cubical.
% - repetitions   [int|double]: number of repetitions to try to fit a new element until the current
%                               active member is removed from the queue.
% - additionaloffset  [double]: additional offset between the hyperspheres around the points. Used
%                               to avoid touching faces.
% - visualise           [bool]: visualise the generation procedure. This will be only affective if
%                               the number of dimensions is n=2 or n=3!
% - debug               [bool]: get further information about the current processing for debuging the 
%                               code in scenarios where it behaves differently

% if (nargin < 2)
%     error('Not enough parameters!');
% end
% 
% % default values for parameters to be provided (see manual and cases below for additinal information)
% nK = 30;
% fO = 10;
% bVis = false;
% bSphere = false;
% bCentered = true;
% bDebug = false;
% 


p = inputParser;
% Required values
addRequired(p, 'space', @isnumeric);
addRequired(p, 'radii');
% Optional values, if not provided, the default value will be used
addParameter(p, 'repetitions',      30,     @isnumeric);
addParameter(p, 'additionaloffset', 10,     @isnumeric);
addParameter(p, 'spherical',        false,  @islogical);
addParameter(p, 'centered',         true,   @islogical);
addParameter(p, 'visualize',        false,  @islogical);
addParameter(p, 'debug',            false,  @islogical);
parse(p, varargin{:});

space       = p.Results.space;
radii       = p.Results.radii;
nK          = p.Results.repetitions;
fO          = p.Results.additionaloffset;
bSphere     = p.Results.spherical;
bCentered   = p.Results.centered;
bVis        = p.Results.visualize;
bDebug      = p.Results.debug;

space = squeeze(space);
nN = length(space);
if ~(ismatrix(space) && size(space, 1) == 1)
    error('Please provide a vector as "space"');
end


% grid size
if isstruct(radii)
    fld = ["name", "rmax", "rmin"];
    for f = fld
        if ~isfield(radii, f)
            error('Please provide the struct field %s to radii', f);
        end
    end
    params = string(fields(radii));
    fld = [fld, "n"];
    params = params(~contains(params, fld));
    switch lower(radii.name)
        case {'normal', 'norm'}
            if ~all(isfield(radii, {'mu', 'sigma'}))
                error('Please provide radii.mu and radii.sigma to the struct');
            end
        case {'poisson', 'poiss'}
            if ~all(isfield(radii, 'lambda'))
                error('Please provide radii.lambda to the struct');
            end
        case {'gamma', 'gam'}
            if ~all(isfield(radii, {'a', 'b'}))
                error('Please provide radii.a and radii.b to the struct');
            end
        case {'uniform', 'unif'}
            if ~all(isfield(radii, {'lower', 'upper'}))
                error('Please provide radii.a and radii.b to the struct');
            end
        otherwise
            error(['Provided distribution %s is not implemented', ...
                'or can not be used with the random function.'], ...
                radii.name);
    end
    if isfield(radii, 'n')
        E = radii.rmin:(radii.rmax-radii.rmin)/radii.n:radii.rmax;
        eval(sprintf("funSample = @() (radii.rmax-radii.rmin)/(2*radii.n) + E(discretize(random(truncate(makedist('%s' %s), radii.rmin, radii.rmax)), E));", radii.name, sprintf(", '%s', radii.%s", [params, params]')));
    else
        eval(sprintf("funSample = @() random(truncate(makedist('%s' %s), radii.rmin, radii.rmax));", radii.name, sprintf(", '%s', radii.%s", [params, params]')));
    end
    w = 2*radii.rmin / sqrt(nN);
    funGridRadius = @(r) ceil((radii.rmax+r+fO)/w);
    % if bVis
    %     cmap = @(r) interp1(linspace(radii.rmin, radii.rmax, 256), jet, r);
    % end
else
    funSample = @() radii(randi(length(radii)));
    w = 2 * min(radii) / sqrt(nN);
    funGridRadius = @(r) ceil((max(radii)+r+fO)/w);
    % if bVis
    %     cmap = @(r) interp1(radii, jet(length(radii)), r);
    % end
end

% if bVis
%     figure;
%     set(gcf, 'WindowState', 'maximized');
% end

%% Step 0
if bSphere
    space = space * 2;
end

bgGrid = cell(ceil(space / w));
bgGridRadii = zeros(ceil(space / w));

%% Step 1
if bSphere
    % if bVis
    %     if nN == 2
    %         plot(...
    %             space(1)*(1+cos(linspace(0,2*pi)))/2, ...
    %             space(2)*(1+sin(linspace(0,2*pi)))/2, ...
    %             'Color', 'k');
    %     elseif nN == 3
    %         [xS, yS, zS] = sphere;
    %         surf(...
    %             space(1)/2 .* (1+xS), ...  % Shift and scale x data
    %             space(2)/2 .* (1+yS), ...  % Shift and scale y data
    %             space(3)/2 .* (1+zS), ...  % Shift and scale z data
    %             'EdgeColor', 'none', ...
    %             'FaceColor', 'k');
    %         hold on;
    %     end
    %     alpha(.05);
    % end
    
    pos = (hypersphere(2*pi*rand(nN-1,1), 1)' .* rand(1,nN) + 1) .* space / 2;
else
    pos = rand(1,nN) .* space;
end
p = num2cell(ceil(pos/w));
p = sub2ind(size(bgGrid), p{:});
bgGrid{p} = pos;
bgGridRadii(p) = funSample();
active = cell(1,1);
active{1} = pos;
activeRadii = bgGridRadii(p);

% if bVis
%     axis equal;
%     if nN == 2
%         rectangle('Position', [bgGrid{p}, 3*bgGridRadii(p), 3*bgGridRadii(p)]-bgGridRadii(p), ...
%             'Curvature', [1 1], ...
%             'FaceColor', cmap(bgGridRadii(p)));
%         axis([0, space(1), 0, space(2)]);
%     elseif nN == 3
%         [xS, yS, zS] = sphere;
%         surf(...
%             bgGrid{p}(1)+xS.*bgGridRadii(p), ...  % Shift and scale x data
%             bgGrid{p}(2)+yS.*bgGridRadii(p), ...  % Shift and scale y data
%             bgGrid{p}(2)+zS.*bgGridRadii(p), ...  % Shift and scale z data
%             'EdgeColor', 'none', ...
%             'FaceColor', cmap(bgGridRadii(p)));
%         axis([0, space(1), 0, space(2), 0, space(3)]);
%         zticks(0:w:space(3));
%         set(gca,'zticklabel',[]);
%     end
%     xticks(0:w:space(1));
%     yticks(0:w:space(2));
%     set(gca,'xticklabel',[]);
%     set(gca,'yticklabel',[]);
%     grid on;
%     hold on;
% end

%% Step 2
while ~isempty(active)
    activeIdx = randi(length(active));
    activePoint = active{activeIdx};
    activeRadius = activeRadii(activeIdx);
    
    % TODO: generate k samples and do the maths parallel
    found = false;
    for j = 1:nK
        % Sample is generated close to the active point with a distance
        % of r*(1+rand) which is between r and 2r
        sampleRadius = funSample();
        sample = activePoint + hypersphere(2*pi*rand(nN-1,1), fO + sampleRadius + activeRadius + min(sampleRadius,activeRadius)*rand)';
        
        if bSphere
            if sum((2*sample./space - 1).^2) > 1
                continue;
            end
        else
            if any([sample < 0, sample > space])
                continue;
            end
        end
        
        d = funGridRadius(sampleRadius);
        p = ceil(sample / w);
        p = num2cell(p - d + (linspace(0,2*d,2*d+1) .* ones(nN, 2*d+1))', 1);
        mgrid = cell(1,nN);
        [mgrid{:}] = ndgrid(p{:});
        mgrid = cellfun(@(a) a(:), mgrid, 'UniformOutput', false);
        mask = cell2mat(mgrid);
        mask = ~any([mask<1, mask>size(bgGrid)], 2);
        mgrid = cellfun(@(a) a(mask), mgrid, 'UniformOutput', false);
        p = sub2ind(size(bgGrid), mgrid{:});
        p = p(bgGridRadii(p) ~= 0);
        
        collision = false;
        if ~isempty(cell2mat(bgGrid(p)'))
            collision = any(sum((cell2mat(bgGrid(p))-sample).^2, 2) < (fO + sampleRadius + bgGridRadii(p)).^2);
        end
        
        if ~collision
            p = num2cell(ceil(sample/w));
            p = sub2ind(size(bgGrid), p{:});
            bgGrid{p} = sample;
            active{end+1} = sample;
            bgGridRadii(p) = sampleRadius;
            activeRadii(end+1) = sampleRadius;
            found = true;
            break;
        end
        
        if bVis && nN == 3
            v = get(gca, 'View');
            v(1) = v(1) + 1/nK;
            view(v);
        end
    end
    
    if ~found
        active(activeIdx) = [];
        activeRadii(activeIdx) = [];
    end
    
    if all([found bVis])
        if nN==2
            rectangle('Position', [sample, 3*sampleRadius, 3*sampleRadius]-sampleRadius, ...
                'Curvature', [1 1], ...
                'FaceColor', cmap(sampleRadius));
        elseif nN==3
            surf(...
                sample(1)+xS.*sampleRadius, ...  % Shift and scale x data
                sample(2)+yS.*sampleRadius, ...  % Shift and scale y data
                sample(3)+zS.*sampleRadius, ...  % Shift and scale z data
                'EdgeColor', 'none', ...
                'FaceColor', cmap(sampleRadius));
            axis equal;
            axis([0, space(1), 0, space(2), 0, space(3)]);
        end
        drawnow;
        pause(0.01);
    end
end

mask = ~cellfun(@isempty, bgGrid(:));
points = cell2mat(bgGrid(mask));
pointRadii = bgGridRadii(mask);

if bCentered
    points = points - space/2;
end

end