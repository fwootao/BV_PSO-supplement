function [bestFitness, bestX, curve] = BVPSO(popSize, maxIter, lb, ub, dim, fobj)
    popNum = popSize / 30; %部落(种群)数
    popSize = 30;

    c1=2.5;  % 忠诚度(个体最优)       
    c2=0.5;  % 忠诚度(部落最优)        
    
    % 活力粒子参数
    rho_max = 0.17;
    rho_min = 0.12;
    
    % 多样性增强参数
    diversity_threshold = 0.15;  % 多样性阈值 - 提高到0.15使其更容易触发
    reinit_percent = 0.2;        % 重初始化比例 - 增加到40%
    
    % 无改进计数器和阈值 - 针对各种群独立
    no_improve_count = zeros(popNum, 1);  % 每个种群独立的无改进计数器
    no_improve_threshold = 15; % 连续15次迭代无改进触发强化探索
    
    % 三策略比例 - 每个种群独立控制
    % ratio1: 策略1(竞争学习)比例, ratio2: 策略2(双驱动力学习)比例, ratio3: 策略3(活力粒子引导PSO)比例
    ratio1 = (1/3) * ones(popNum, 1);  % 初始比例各为1/3
    ratio2 = (1/3) * ones(popNum, 1);
    ratio3 = (1/3) * ones(popNum, 1);
    
    % 各种群的最优适应值记录 - 用于检测改进
    last_pop_best_fit = inf * ones(popNum, 1);

    % 双驱动力学习策略相关参数
    num_neighbors = 2;  % 新颖性计算的邻居数量

    % 速度限值（移动长度限值在定义域的1/10）
    velMax = 0.2*(ub-lb);
    VelMin = -velMax;
    % 种群最优/最差个体
    popBestX = zeros(popNum, dim);
    popBestFit = zeros(popNum, 1);
    popWorstIndex = zeros(popNum, 1);
    popWorstX = zeros(popNum, dim);
    popWorstFit = zeros(popNum, 1);

    % 初始化全局优适应值为最大double
    gBestX = zeros(1, dim);
    gBestFit=inf;    
    
    % 初始化所有个体
    x = Initialization(popSize * popNum, dim, ub, lb);
    
    % 所有个体的速度列表[p, d]
    vel = zeros(popNum*popSize,dim);
   
    % 所有个体适应值列表[p, 1]
    fit = zeros(popNum*popSize,1);

    % 计算适应值, 记录最优个体x, f, 所在部落
    for i=1:popNum*popSize
        fit(i)=fobj(x(i,:));
        if fit(i)<gBestFit
            gBestX=x(i,:);
            gBestFit=fit(i);
        end
    end

    curve = zeros(1, maxIter);
    %收敛曲线第一列记录最优适应值

    % 更新个人最优[x f(x)]
    pBestX = x;
    pBestFit = fit;
   
    % 更新种群中最优[x f(x)]
    for i=1:popNum
        MemberCost = fit((i - 1) * popSize + 1:i * popSize);
        [M, I] = min(MemberCost);
        popBestFit(i) = M;
        popBestX(i,:) = x((i-1) * popSize + I(1),:);
        [M,I] = max(MemberCost);
        popWorstFit(i) = M;
        popWorstX(i,:) = x((i-1) * popSize + I(1),:);
        popWorstIndex(i) = I(1);
    end
    
    % 初始化每个种群的活力粒子
    vitalityX = zeros(popNum, dim);
    vitalityFit = inf * ones(popNum, 1);
       
    for it=1:maxIter
        % 检查各种群最优是否有改进 - 针对各种群独立进行
        for g = 1:popNum
            if abs(popBestFit(g) - last_pop_best_fit(g)) < 1e-10
                no_improve_count(g) = no_improve_count(g) + 1;
            else
                no_improve_count(g) = 0;
                last_pop_best_fit(g) = popBestFit(g);
            end
        end
        
        % 忠诚度和撤退参数
        r1 = rand(popNum*popSize, dim);
        r2 = rand(popNum*popSize, dim);

        % 计算当前迭代的ρ值
        rho = rho_max - (rho_max - rho_min) * (it / maxIter);

        %% 种群排序 - 为混合策略做准备
        for g = 1:popNum
            startIdx = (g-1) * popSize + 1;
            endIdx = g * popSize;
            
            % 获取当前种群的适应值和索引
            [~, sortIdx] = sort(fit(startIdx:endIdx), 'descend');
            
            % 重新排列当前种群的各个数组
            x(startIdx:endIdx, :) = x(startIdx + sortIdx - 1, :);
            vel(startIdx:endIdx, :) = vel(startIdx + sortIdx - 1, :);
            pBestX(startIdx:endIdx, :) = pBestX(startIdx + sortIdx - 1, :);
            pBestFit(startIdx:endIdx) = pBestFit(startIdx + sortIdx - 1);
        end

        %% 构建活力粒子
        for g=1:popNum
            % 为每个种群构建一个新的活力粒子
            newVitalityX = zeros(1, dim);
            
            % 对每个维度独立选择生成方式
            for d=1:dim
                if rand() > rho
                    % 方式1：从pBest中随机选择两个，取适应值更小的
                    idx1 = randi(popNum*popSize);
                    idx2 = randi(popNum*popSize);
                    if pBestFit(idx1) < pBestFit(idx2)
                        newVitalityX(d) = pBestX(idx1, d);
                    else
                        newVitalityX(d) = pBestX(idx2, d);
                    end
                else
                    % 方式2：在[lb, ub]范围内随机生成
                    if size(lb, 2) == 1
                        newVitalityX(d) = rand() * (ub - lb) + lb;
                    else
                        newVitalityX(d) = rand() * (ub(d) - lb(d)) + lb(d);
                    end
                end
            end
            
            % 计算新活力粒子的适应值
            newVitalityFit = fobj(newVitalityX);
            
            % 如果新活力粒子比旧活力粒子更好，则替换
            if newVitalityFit < vitalityFit(g)
                vitalityX(g, :) = newVitalityX;
                vitalityFit(g) = newVitalityFit;
            end
            
            % 如果活力粒子比种群最优还好，更新种群最优
            if vitalityFit(g) < popBestFit(g)
                popBestFit(g) = vitalityFit(g);
                popBestX(g, :) = vitalityX(g, :);
            end
            
            % 如果活力粒子是全局最优，更新全局最优
            if vitalityFit(g) < gBestFit
                gBestFit = vitalityFit(g);
                gBestX = vitalityX(g, :);
            end
        end

        %% 上层最优替换下层最差
        for popIndex = popNum-1:-1:1
            if popBestFit(popIndex) < popBestFit(popIndex + 1)
                % popBestX/popBestFit/pBestX/pBestFit
                popBestFit(popIndex + 1) = popBestFit(popIndex);
                popBestX(popIndex + 1, :) = popBestX(popIndex, :);
                pBestFit(popIndex * popSize + popWorstIndex(popIndex + 1)) = popBestFit(popIndex);
                pBestX(popIndex * popSize + popWorstIndex(popIndex + 1),:) = popBestX(popIndex, :);
            end
        end
        
        %% 双驱动力学习策略预计算优化：避免重复计算
        % 预计算双驱动力学习策略参数（每次迭代只计算一次）
        p_ratio_fit = 0.7*(1/(1+exp(0.001*((it-1)-(maxIter/2))/dim))) + 0.1;
        
        % 为每个种群预计算排序和新颖性（每次迭代每个种群只计算一次）
        popSortedIdx = cell(popNum, 1);  % 存储每个种群的适应度排序索引
        popNoveltyIdx = cell(popNum, 1); % 存储每个种群的新颖性排序索引
        
        for g = 1:popNum
            startIdx = (g-1) * popSize + 1;
            endIdx = g * popSize;
            
            % 适应度排序（升序，越小越好）- 基于pBestFit进行排序
            currentPopPBestFit = pBestFit(startIdx:endIdx);
            [~, popSortedIdx{g}] = sort(currentPopPBestFit, 'ascend');
            
            % 新颖性计算和排序（降序，越大越好）- 基于pBestX进行计算
            currentPopPBestX = pBestX(startIdx:endIdx, :);
            novelty = computeNovelty(currentPopPBestX, popSize, num_neighbors);
            [~, popNoveltyIdx{g}] = sort(novelty, 'descend');
        end

        oldPBestFit = pBestFit;
         
        for i=1:popNum*popSize % [1, p]
            GroupNum=floor((i-1)/popSize)+1;
            %% 三策略混合：竞争学习 + 双驱动力学习 + 活力粒子引导PSO
            alpha = 1 - (1 - 0.5) * ((it - 1) / (maxIter - 1)).^2;
            c = alpha * 0.9;

            % 策略1：竞争学习更新
            % 计算三种不同的中心位置（用于竞争学习）
            startIdx = (GroupNum-1) * popSize + 1;
            endIdx = GroupNum * popSize;
            center1 = mean(x(startIdx:endIdx, :), 1);                    % 当前种群x的均值
            center2 = mean(pBestX(startIdx:endIdx, :), 1);              % 当前种群pBestX的均值
            center3 = mean(vitalityX, 1);                               % 所有种群活力粒子的均值
            
            % 等概率选择三种center中的一种
            rand_val = rand();
            if rand_val < 1/2
                center = center1;  % 选择当前种群x的均值
            elseif rand_val < 5/6
                center = center2;  % 选择当前种群pBestX的均值
            else
                center = center3;  % 选择所有种群活力粒子的均值
            end
            
            % 生成获胜者索引（用于竞争学习）
            winidxmask = repmat((startIdx:endIdx)', 1, dim);
            winidx = winidxmask + ceil(rand(popSize, dim) .* (endIdx - winidxmask));
            pwin = x(startIdx:endIdx, :);
            for d = 1:dim
                pwin(:, d) = x(winidx(:, d), d);
            end

            % 计算个体在当前种群中的相对位置
            localIdx = i - startIdx + 1;
            c3 = dim/popSize * 0.1; % 社会学习因子，随维度和种群大小调整
            vel1 = (1.5 - GroupNum / popNum) * rand(1, dim) .* vel(i,:)...
                    + (0.5 + GroupNum / popNum ) * r1(i, :) .* (pwin(localIdx, :) - x(i,:))...
                    + c3 * r2(i, :) .* (center - x(i,:));
            
            % 策略2：双驱动力学习更新
            % 使用预计算的参数和排序结果
            % 适应度引导：从预计算的排序中选择
            selectIdx1 = max(1, ceil(popSize * p_ratio_fit * rand()));
            pbest_fit_idx = startIdx + popSortedIdx{GroupNum}(selectIdx1) - 1;
            
            % 新颖性引导：从预计算的排序中选择
            selectIdx2 = max(1, ceil(popSize * p_ratio_fit * rand()));
            pbest_novelty_idx = startIdx + popNoveltyIdx{GroupNum}(selectIdx2) - 1;
            
            % 双驱动力学习速度更新
            vel2 =  (0.9 - 0.4 * GroupNum / popNum) * vel(i, :)...
                    + (0.9 - 0.2 * GroupNum / popNum) * rand(1, dim) .* (pBestX(pbest_fit_idx, :) - x(i, :))...
                    + (0.5 + 0.2 * GroupNum / popNum) * rand(1, dim) .* (pBestX(pbest_novelty_idx, :) - x(i, :));
            
            % 策略3：活力粒子引导PSO更新
            if pBestFit(i) >= vitalityFit(GroupNum)
                % 如果个体不如活力粒子，使用活力粒子替代pBestX
                vel3 = (c - 0.45 * GroupNum / popNum) * vel(i,:)...
                        + (c1 - 2 * GroupNum / popNum) * r1(i, :) .* (vitalityX(GroupNum,:) - x(i,:))...
                        + (c2 + 2 * GroupNum / popNum) * r2(i, :) .* (popBestX(GroupNum,:) - x(i,:));
            else
                % 如果个体比活力粒子好，使用原来的pBestX
                vel3 = (c - 0.45 * GroupNum / popNum) * vel(i,:)...
                        + (c1 - 2 * GroupNum / popNum) * r1(i, :) .* (pBestX(i,:) - x(i,:))...
                        + (c2 + 2 * GroupNum / popNum) * r2(i, :) .* (popBestX(GroupNum,:) - x(i,:));
            end
            
            % 三策略分配：根据动态比例分配
            cumRatio1 = ratio1(GroupNum);
            cumRatio2 = cumRatio1 + ratio2(GroupNum);
            % ratio3 = 1 - cumRatio2 (自动满足)
            
            if localIdx <= floor(popSize * cumRatio1)
                % 使用策略1：竞争学习
                vel(i,:) = vel1;
            elseif localIdx <= floor(popSize * cumRatio2)
                % 使用策略2：双驱动力学习
                vel(i,:) = vel2;
            else
                % 使用策略3：活力粒子引导PSO
                vel(i,:) = vel3;
            end
            
            %% 截取速度超界值
            vel(i,:) = max(vel(i,:), VelMin + 0.1 * (GroupNum / popNum)  * (ub - lb));
            vel(i,:) = min(vel(i,:), velMax - 0.1 * (GroupNum / popNum) * (ub - lb));
            % 更新位置
            x(i,:) = x(i,:) + vel(i,:);
            % 超出界限反向走(速度取反) 更改了
            if any(x(i,:)<lb) || any(x(i,:)>ub)
                vel(i,:) = -vel(i,:);
            end
            % 截取个体x超出界限部分
            x(i,:) = max(x(i,:),lb);
            x(i,:) = min(x(i,:),ub);

            %% 计算适应值
            fit(i) = fobj(x(i,:));
            
            if fit(i) < pBestFit(i)
                pBestFit(i) = fit(i);
                pBestX(i,:) = x(i,:);
            end
            % 更新种群优个体[x, f(x)]
            if fit(i) < popBestFit(GroupNum)
                popBestFit(GroupNum) = fit(i);
                popBestX(GroupNum,:) = x(i,:);
                % popBestIndex(GroupNum) = i - (GroupNum - 1) * popSize;
            elseif fit(i) > popWorstFit(GroupNum)
                popWorstFit(GroupNum) = fit(i);
                popWorstX(GroupNum,:) = x(i,:);
                popWorstIndex(GroupNum) = i - (GroupNum - 1) * popSize;
            end
            % 更新全局优
            if fit(i) < gBestFit
                gBestFit = fit(i);
                gBestX = x(i,:);
            end
        end

        %% 计算各种群多样性 - 针对各种群独立进行
        diversity = calculatePopulationDiversity(x, popSize, popNum);
        
        %% 多样性增强机制 - 针对各种群独立检测和触发
        for g = 1:popNum
            % 计算当前种群的多样性
            currentDiversity = diversity(g);
            
            % 检查当前种群是否需要多样性增强
            if currentDiversity < diversity_threshold || no_improve_count(g) >= no_improve_threshold
                % 选择当前种群中适应度最差的一部分个体进行重初始化
                startIdx = (g-1) * popSize + 1;
                endIdx = g * popSize;
                
                % 获取当前种群的适应度
                currentPopFit = fit(startIdx:endIdx);
                [~, sortedIndices] = sort(currentPopFit, 'descend'); % 降序排列，最差的在前面
                
                % 选择最差的reinit_percent比例进行重初始化
                numToReinit = ceil(popSize * reinit_percent);
                for j = 1:numToReinit
                    idx = startIdx + sortedIndices(j) - 1;
                    
                    % 重初始化策略：强化多样性的三种方案
                    r = rand();
                    if r < 0.25
                        % 全局探索：完全随机重初始化
                        x(idx,:) = lb + rand(1, dim) .* (ub - lb);
                    elseif r < 0.5
                        % 局部利用：在全局最优附近随机搜索
                        perturbRange = (ub - lb) * 0.2; % 扰动范围增加到20%
                        x(idx,:) = gBestX + (rand(1, dim) - 0.5) .* perturbRange;
                        % 边界处理
                        x(idx,:) = max(x(idx,:), lb);
                        x(idx,:) = min(x(idx,:), ub);
                    elseif r < 0.75
                        % 反向学习：远离当前解
                        x(idx,:) = lb + ub - x(idx,:) + rand(1, dim) * 0.1 .* (ub - lb);
                        % 边界处理
                        x(idx,:) = max(x(idx,:), lb);
                        x(idx,:) = min(x(idx,:), ub);
                    else
                        % 反向学习靠前：选择种群中适应度较好的个体进行学习
                        % 通过改索引的方式，选择排序靠前的个体
                        forwardIdx = startIdx + sortedIndices(end - j + 1) - 1; % 选择靠前的个体
                        x(idx,:) = x(forwardIdx,:) + rand(1, dim) * 0.1 .* (ub - lb);                        
                        % 边界处理
                        x(idx,:) = max(x(idx,:), lb);
                        x(idx,:) = min(x(idx,:), ub);
                    end
                    
                    % 重置速度
                    vel(idx,:) = (rand(1, dim) - 0.5) .* (ub - lb) * 0.1;
                    
                    % 评估新位置
                    fit(idx) = fobj(x(idx,:));
                    
                    % 更新个体最优
                    if fit(idx) < pBestFit(idx)
                        pBestFit(idx) = fit(idx);
                        pBestX(idx,:) = x(idx,:);
                    end
                    
                    % 更新种群最优
                    if fit(idx) < popBestFit(g)
                        popBestFit(g) = fit(idx);
                        popBestX(g,:) = x(idx,:);
                    end
                    
                    % 更新全局最优
                    if fit(idx) < gBestFit
                        gBestFit = fit(idx);
                        gBestX = x(idx,:);
                    end
                end
                
                % 重置当前种群的无改进计数器
                no_improve_count(g) = 0;
            end
        end

        %% 动态调整三策略比例 - 基于各种群的改进情况
        for g = 1:popNum
            startIdx = (g-1) * popSize + 1;
            endIdx = g * popSize;
            
            % 计算当前种群中三种策略的改进情况
            bnd1 = floor(popSize * ratio1(g));
            bnd2 = floor(popSize * (ratio1(g) + ratio2(g)));
            
            % 统计策略1（竞争学习）的改进数量
            strategy1_improvements = 0;
            for i = startIdx:startIdx + bnd1 - 1
                if fit(i) < oldPBestFit(i)
                    strategy1_improvements = strategy1_improvements + 1;
                end
            end
            
            % 统计策略2（双驱动力学习）的改进数量
            strategy2_improvements = 0;
            for i = startIdx + bnd1:startIdx + bnd2 - 1
                if fit(i) < oldPBestFit(i)
                    strategy2_improvements = strategy2_improvements + 1;
                end
            end
            
            % 统计策略3（活力粒子引导PSO）的改进数量
            strategy3_improvements = 0;
            for i = startIdx + bnd2:endIdx
                if fit(i) < oldPBestFit(i)
                    strategy3_improvements = strategy3_improvements + 1;
                end
            end
            
            % 动态调整比例
            total_improvements = strategy1_improvements + strategy2_improvements + strategy3_improvements;
            if total_improvements > 0
                ratio1(g) = strategy1_improvements / total_improvements;
                ratio2(g) = strategy2_improvements / total_improvements;
                ratio3(g) = strategy3_improvements / total_improvements;
                
                % 限制比例范围在[0.1, 0.8]，确保每种策略都有最小分配
                ratio1(g) = max(0.1, min(0.8, ratio1(g)));
                ratio2(g) = max(0.1, min(0.8, ratio2(g)));
                ratio3(g) = max(0.1, min(0.8, ratio3(g)));
                
                % 归一化确保总和为1
                total_ratio = ratio1(g) + ratio2(g) + ratio3(g);
                ratio1(g) = ratio1(g) / total_ratio;
                ratio2(g) = ratio2(g) / total_ratio;
                ratio3(g) = ratio3(g) / total_ratio;
                
            end
        end

        curve(it) = gBestFit;
        
    end
        
    bestX = gBestX;
    bestFitness = gBestFit;
end

% 计算新颖性函数 - 双驱动力学习策略使用
function novelty = computeNovelty(popX, popSize, num_neighbors)
    novelty = zeros(popSize, 1);
    
    for i = 1:popSize
        distances = zeros(1, popSize);
        % 计算个体i与所有其他个体的欧几里得距离
        for j = 1:popSize
            if i ~= j
                distances(j) = norm(popX(i,:) - popX(j,:));
            else
                distances(j) = inf; % 排除自身
            end
        end
        
        % 排序找到最近的邻居
        [sortedDist, ~] = sort(distances, 'ascend');
        
        % 计算与最近num_neighbors个邻居的平均距离
        neighbor_count = min(num_neighbors, sum(distances < inf));
        if neighbor_count > 0
            novelty(i) = mean(sortedDist(1:neighbor_count));
        else
            novelty(i) = 0;
        end
    end
    
    % 归一化新颖性值
    if max(novelty) > 0
        novelty = novelty / max(novelty);
    end
end

% 计算各种群多样性函数 - 针对各种群独立计算
function diversity = calculatePopulationDiversity(x, popSize, popNum)
    diversity = zeros(popNum, 1);
    
    for g = 1:popNum
        startIdx = (g-1) * popSize + 1;
        endIdx = g * popSize;
        
        % 获取当前种群的个体
        currentPop = x(startIdx:endIdx, :);
        
        % 计算当前种群的中心点
        center = mean(currentPop, 1);
        
        % 计算每个粒子到中心的欧氏距离
        distances = zeros(popSize, 1);
        for i = 1:popSize
            distances(i) = norm(currentPop(i,:) - center);
        end
        
        % 多样性定义为平均距离
        diversity(g) = mean(distances);
    end
end

function Positions=Initialization(SearchAgents_no,dim,ub,lb)

    Boundary_no= size(ub,2); % 边界数量
    
    % 如果所有变量的边界都相等，用户为ub和lb输入单个数字
    if Boundary_no==1
        Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
    end
    
    % 如果每个变量都有不同的lb和ub
    if Boundary_no>1
        for i=1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
        end
    end
end