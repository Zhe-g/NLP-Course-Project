// ===== ABSA 细粒度情感分析 — 前端逻辑 =====

let charts = {};
let currentData = null;
let selectedFiles = [];  // 已选文件列表
let historyData = [];    // 历史分析记录（用于统计）

// ===== 分页筛选变量 =====
let currentPage = 1;
let pageSize = 10;
let sentimentFilter = 'all';  // 'all', 'positive', 'neutral', 'negative'
let searchKeyword = '';
let filteredResults = [];

// ===== 暂存列表变量 =====
let stashList = [];  // 存储暂存的分析结果

// DOM elements
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const modelStatus = document.getElementById('modelStatus');
const fileListEl = document.getElementById('fileList');
const fileListContent = document.getElementById('fileListContent');

// ===== 初始化 =====
checkModelHealth();
analyzeBtn.addEventListener('click', handleTextAnalysis);
fileInput.addEventListener('change', handleFileSelect);
initDragDrop();

// ===== 模式切换 =====
function switchMode(mode) {
    const textModeBtn = document.getElementById('textModeBtn');
    const fileModeBtn = document.getElementById('fileModeBtn');
    const textModePanel = document.getElementById('textModePanel');
    const fileModePanel = document.getElementById('fileModePanel');

    if (mode === 'text') {
        textModeBtn.classList.add('active');
        fileModeBtn.classList.remove('active');
        textModePanel.style.display = 'block';
        fileModePanel.style.display = 'none';
    } else {
        textModeBtn.classList.remove('active');
        fileModeBtn.classList.add('active');
        textModePanel.style.display = 'none';
        fileModePanel.style.display = 'block';
    }
}

// ===== 拖拽上传初始化 =====
function initDragDrop() {
    // 拖拽进入
    dropZone.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
    });

    // 拖拽悬停
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
    });

    // 拖拽离开
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
    });

    // 拖拽放下
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            addFiles(files);
        }
    });
}

// ===== 文件选择处理 =====
function handleFileSelect() {
    const files = fileInput.files;
    if (files.length > 0) {
        addFiles(files);
    }
}

// ===== 添加文件到列表 =====
function addFiles(files) {
    for (let file of files) {
        // 检查文件格式
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['txt', 'csv', 'xlsx'].includes(ext)) {
            showError(`不支持的文件格式: ${file.name}`);
            continue;
        }
        // 检查是否已存在
        if (!selectedFiles.some(f => f.name === file.name && f.size === file.size)) {
            selectedFiles.push(file);
        }
    }
    updateFileList();
}

// ===== 更新文件列表显示 =====
function updateFileList() {
    if (selectedFiles.length === 0) {
        fileListEl.style.display = 'none';
        return;
    }

    fileListEl.style.display = 'block';
    fileListContent.innerHTML = '';

    selectedFiles.forEach((file, index) => {
        const ext = file.name.split('.').pop().toLowerCase();
        const icon = ext === 'txt' ? '📝' : ext === 'csv' ? '📊' : '📑';

        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <div class="file-item-info">
                <span class="file-icon">${icon}</span>
                <span class="file-name">${file.name}</span>
                <span class="file-size">${formatFileSize(file.size)}</span>
            </div>
            <button class="file-remove" onclick="removeFile(${index})">×</button>
        `;
        fileListContent.appendChild(item);
    });
}

// ===== 移除单个文件 =====
function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFileList();
}

// ===== 清空所有文件 =====
function clearFiles() {
    selectedFiles = [];
    fileInput.value = '';
    updateFileList();
}

// ===== 格式化文件大小 =====
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ===== 模型健康检查 =====
async function checkModelHealth() {
    try {
        const resp = await fetch('/health');
        const data = await resp.json();
        if (data.model_api === 'connected') {
            modelStatus.innerHTML = '<span class="dot green"></span> 模型服务已连接 (RoBERTa-wwm-ext)';
        } else {
            modelStatus.innerHTML = '<span class="dot red"></span> 模型服务未连接 — 请先启动: python scripts/run_api.py --port 5000';
        }
    } catch (e) {
        modelStatus.innerHTML = '<span class="dot red"></span> Web 服务异常';
    }
}

// ===== 文件上传分析 =====
async function handleFileUpload() {
    if (selectedFiles.length === 0) {
        showError('请选择或拖拽文件到上传区域');
        return;
    }

    showLoading();
    setProgress(0, '准备上传...');
    
    if (selectedFiles.length === 1) {
        await uploadSingle(selectedFiles[0]);
    } else {
        await uploadMultipleWithProgress(selectedFiles);
    }
    hideLoading();
}

async function uploadSingle(file) {
    const fd = new FormData();
    fd.append('file', file);
    try {
        const totalLines = await countLinesInFile(file);
        let processed = 0;
        
        setProgress(0, `正在分析文件: ${file.name}`);
        
        const resp = await fetch('/upload', { method: 'POST', body: fd });
        const data = await resp.json();
        
        // 设置完成状态
        setProgress(100, '分析完成');
        
        if (resp.ok) { 
            currentData = data; 
            displayResults(data); 
        }
        else { 
            showError(data.error || '分析失败'); 
        }
    } catch (e) { 
        showError('网络错误: ' + e.message); 
    } finally {
        // 确保隐藏加载状态
        hideLoading();
    }
}

// 带进度的批量上传
async function uploadMultipleWithProgress(files) {
    const fd = new FormData();
    for (let f of files) fd.append('files', f);

    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/batch-upload');
        
        // 流式处理响应
        xhr.onprogress = () => {
            if (xhr.responseText) {
                const response = xhr.responseText;
                const lastNewline = response.lastIndexOf('\n\n');
                if (lastNewline !== -1) {
                    const completeEvents = response.substring(0, lastNewline);
                    const events = completeEvents.split('\n\n').filter(e => e.trim());
                    
                    for (const event of events) {
                        if (event.startsWith('data: ')) {
                            const dataStr = event.substring(6);
                            try {
                                const data = JSON.parse(dataStr);
                                processProgressEvent(data);
                            } catch (e) {
                                console.log('解析事件失败:', e);
                            }
                        }
                    }
                }
            }
        };

        xhr.onreadystatechange = () => {
            if (xhr.readyState === XMLHttpRequest.DONE) {
                // 设置完成状态
                setProgress(100, '分析完成');
                
                if (xhr.status === 200) {
                    try {
                        const responseText = xhr.responseText;
                        const events = responseText.split('\n\n').filter(e => e.trim());
                        
                        for (const event of events) {
                            if (event.startsWith('data: ')) {
                                const dataStr = event.substring(6);
                                try {
                                    const data = JSON.parse(dataStr);
                                    processProgressEvent(data);
                                } catch (e) {
                                    console.log('解析事件失败:', e);
                                }
                            }
                        }
                    } catch (e) {
                        console.error('解析响应失败:', e);
                    }
                    resolve();
                } else {
                    showError('批量分析失败');
                    reject();
                }
                
                // 确保隐藏加载状态
                hideLoading();
            }
        };

        xhr.onerror = () => {
            showError('网络错误');
            hideLoading();
            reject();
        };

        xhr.send(fd);
    });
}

// 处理进度事件
function processProgressEvent(data) {
    if (!data) return;
    
    switch (data.stage) {
        case 'upload':
            setProgress(Math.round(data.current / data.total * 100), 
                `上传文件: ${data.file} (${data.current}/${data.total})`);
            break;
        case 'analyze':
            setProgress(data.progress || 0, data.message);
            break;
        case 'merge':
            setProgress(95, data.message);
            break;
        case 'complete':
            setProgress(100, '分析完成');
            currentData = data.data;
            displayResults(data.data);
            break;
        case 'error':
            showError(data.error || '分析失败');
            break;
    }
}

// 设置进度显示
function setProgress(percent, message) {
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    const loadingText = document.getElementById('loadingText');
    
    if (progressFill) progressFill.style.width = percent + '%';
    if (progressPercent) progressPercent.textContent = percent + '%';
    if (loadingText) loadingText.textContent = message;
}

// 统计文件行数（用于估算进度）
function countLinesInFile(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            const lines = content.split('\n').filter(line => line.trim());
            resolve(lines.length);
        };
        reader.readAsText(file);
    });
}

// ===== 手动输入 =====
async function handleTextAnalysis() {
    const text = textInput.value.trim();
    if (!text) { showError('请输入评论内容'); return; }
    showLoading();
    setProgress(0, '正在分析...');
    
    try {
        const resp = await fetch('/input-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await resp.json();
        
        // 设置完成状态
        setProgress(100, '分析完成');
        
        if (resp.ok) {
            // 包装成批量格式
            const wrapped = {
                summary: {
                    total_reviews: 1,
                    total_aspects: data.aspects ? data.aspects.length : 0,
                    avg_aspects_per_review: data.aspects ? data.aspects.length : 0,
                    review_sentiment_dist: {
                        positive: data.summary.positive_count > data.summary.negative_count ? 1 : 0,
                        neutral: 0,
                        negative: data.summary.negative_count >= data.summary.positive_count ? 1 : 0,
                        positive_ratio: data.summary.positive_count > data.summary.negative_count ? 100 : 0,
                        negative_ratio: data.summary.negative_count >= data.summary.positive_count ? 100 : 0,
                    },
                    aspect_sentiment_dist: {
                        positive: data.summary.positive_count,
                        neutral: data.summary.neutral_count,
                        negative: data.summary.negative_count,
                        positive_ratio: data.summary.total_aspects > 0 ? Math.round(data.summary.positive_count / data.summary.total_aspects * 100) : 0,
                        negative_ratio: data.summary.total_aspects > 0 ? Math.round(data.summary.negative_count / data.summary.total_aspects * 100) : 0,
                    },
                    group_stats: _buildGroupStats(data.aspects || []),
                    category_rankings: _buildCategoryRankings(data.aspects || []),
                },
                results: [{ ...data, text: textInput.value }]
            };
            currentData = wrapped;
            displayResults(wrapped);
        } else {
            showError(data.error || '分析失败');
        }
    } catch (e) { 
        showError('网络错误: ' + e.message); 
    } finally {
        // 确保隐藏加载状态
        hideLoading();
    }
}

// ===== 显示结果 =====
function displayResults(data) {
    // 将新数据追加到历史记录
    historyData.push({
        timestamp: new Date().toLocaleString('zh-CN'),
        data: JSON.parse(JSON.stringify(data))
    });
    
    // 更新累计统计
    updateCumulativeStats();
    
    resultsSection.style.display = 'block';
    updateSummary(data.summary);
    updateCharts(data);
    updateGroupDetail(data.summary.group_stats || {});
    
    // 获取所有历史评论
    const allReviews = getAllReviews();
    updateReviewsList(allReviews);
    
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// ===== 获取所有历史评论 =====
function getAllReviews() {
    const allReviews = [];
    historyData.forEach((record, index) => {
        const results = record.data.results || [];
        results.forEach((review, idx) => {
            allReviews.push({
                ...review,
                analysisIndex: index + 1,
                analysisTime: record.timestamp
            });
        });
    });
    return allReviews;
}

// ===== 获取累计维度排名数据 =====
function getCumulativeRankings() {
    const cumulativeStats = {};
    
    historyData.forEach(record => {
        // 尝试从 category_rankings 获取
        let rankings = record.data.summary.category_rankings || [];
        
        // 如果 category_rankings 为空，从 results 重新计算
        if (rankings.length === 0) {
            rankings = _buildCategoryRankingsFromResults(record.data.results || []);
        }
        
        rankings.forEach(r => {
            const key = r.category || r.category_zh;
            if (!cumulativeStats[key]) {
                cumulativeStats[key] = {
                    category: r.category,
                    category_zh: r.category_zh || r.category,
                    total: 0,
                    positive: 0,
                    negative: 0,
                    neutral: 0
                };
            }
            // 兼容多种字段名
            cumulativeStats[key].total += r.total || 0;
            cumulativeStats[key].positive += r.positive || r.positive_count || 0;
            cumulativeStats[key].negative += r.negative || r.negative_count || 0;
            cumulativeStats[key].neutral += r.neutral || r.neutral_count || 0;
        });
    });
    
    return Object.values(cumulativeStats).map(item => ({
        ...item,
        positive_ratio: item.total > 0 
            ? Math.round((item.positive / item.total) * 100) 
            : 0,
        negative_ratio: item.total > 0 
            ? Math.round((item.negative / item.total) * 100) 
            : 0,
        neutral_ratio: item.total > 0 
            ? Math.round((item.neutral / item.total) * 100) 
            : 0
    }));
}

// ===== 从 results 重新构建 category_rankings =====
function _buildCategoryRankingsFromResults(results) {
    const cats = {};
    results.forEach(r => {
        const aspects = r.aspects || [];
        aspects.forEach(a => {
            const key = a.category;
            if (!cats[key]) cats[key] = {
                category: key,
                category_zh: a.category_zh,
                total: 0,
                positive: 0,
                negative: 0,
                neutral: 0
            };
            cats[key].total++;
            if (a.sentiment === 'positive') cats[key].positive++;
            else if (a.sentiment === 'negative') cats[key].negative++;
            else cats[key].neutral++;
        });
    });
    return Object.values(cats);
}

// ===== 更新累计统计 =====
function updateCumulativeStats() {
    if (historyData.length === 0) {
        document.getElementById('cumulativeSection')?.remove();
        return;
    }
    
    let totalReviews = 0;
    let totalPositive = 0;
    let totalNegative = 0;
    
    historyData.forEach(record => {
        const s = record.data.summary;
        totalReviews += s.total_reviews || 0;
        totalPositive += s.review_sentiment_dist?.positive || 0;
        totalNegative += s.review_sentiment_dist?.negative || 0;
    });
    
    const statsEl = document.getElementById('cumulativeStats');
    if (statsEl) {
        statsEl.innerHTML = `
            <div class="stat-item">
                <span class="stat-value">${historyData.length}</span>
                <span class="stat-label">分析次数</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${totalReviews}</span>
                <span class="stat-label">累计评论</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${totalPositive}</span>
                <span class="stat-label">好评次数</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${totalNegative}</span>
                <span class="stat-label">差评次数</span>
            </div>
        `;
    }
}

// ===== 概览卡片 =====
function updateSummary(s) {
    document.getElementById('totalCount').textContent = s.total_reviews || 0;
    document.getElementById('totalAspects').textContent = s.total_aspects || 0;
    document.getElementById('avgAspects').textContent = s.avg_aspects_per_review || 0;
    const rd = s.review_sentiment_dist || {};
    document.getElementById('posReviews').textContent = rd.positive || 0;
    document.getElementById('posReviewsPct').textContent = (rd.positive_ratio || 0) + '%';
    document.getElementById('negReviews').textContent = rd.negative || 0;
    document.getElementById('negReviewsPct').textContent = (rd.negative_ratio || 0) + '%';
}

// ===== 图表 =====
function updateCharts(data) {
    Object.values(charts).forEach(c => c.destroy());
    charts = {};
    const s = data.summary;

    // 1. 评论整体情感饼图
    const rd = s.review_sentiment_dist || {};
    charts.reviewPie = new Chart(document.getElementById('reviewPieChart').getContext('2d'), {
        type: 'pie',
        data: {
            labels: ['正向', '中性', '负向'],
            datasets: [{
                data: [rd.positive || 0, rd.neutral || 0, rd.negative || 0],
                backgroundColor: ['#48bb78', '#4299e1', '#f56565'],
            }]
        },
        options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
    });

    // 2. 维度情感饼图
    const ad = s.aspect_sentiment_dist || {};
    charts.aspectPie = new Chart(document.getElementById('aspectPieChart').getContext('2d'), {
        type: 'pie',
        data: {
            labels: ['正向', '中性', '负向'],
            datasets: [{
                data: [ad.positive || 0, ad.neutral || 0, ad.negative || 0],
                backgroundColor: ['#48bb78', '#4299e1', '#f56565'],
            }]
        },
        options: { responsive: true, plugins: { legend: { position: 'bottom' } } }
    });

    // 3. 5 个一级维度雷达图
    const groups = s.group_stats || {};
    const gNames = Object.keys(groups);
    if (gNames.length > 0) {
        // 雷达图：好评率
        const radarData = gNames.map(g => groups[g].positive_ratio || 0);
        
        // 创建渐变颜色函数（蓝色到红色）
        const getGradientColor = (value) => {
            const ratio = value / 100;
            const r = Math.round(59 + ratio * 196);  // 59 -> 255 (蓝色->红色)
            const g = Math.round(130 - ratio * 130); // 130 -> 0
            const b = Math.round(246 - ratio * 200); // 246 -> 46
            return `rgb(${r}, ${g}, ${b})`;
        };

        // 生成每个点的颜色
        const pointColors = radarData.map(v => getGradientColor(v));
        
        // 获取画布上下文用于创建渐变
        const ctx = document.getElementById('radarChart').getContext('2d');
        const gradient = ctx.createRadialGradient(0, 0, 0, 0, 0, 1);
        gradient.addColorStop(0, 'rgba(255, 99, 99, 0.3)');    // 中心红色
        gradient.addColorStop(0.5, 'rgba(255, 170, 99, 0.2)');  // 中间橙色
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0.1)');    // 边缘蓝色

        charts.radarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: gNames,
                datasets: [{
                    label: '好评率 (%)',
                    data: radarData,
                    backgroundColor: gradient,
                    borderColor: '#667eea',
                    borderWidth: 3,
                    borderJoinStyle: 'round',
                    pointBackgroundColor: pointColors,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#667eea',
                    pointHoverBorderWidth: 3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        min: 0,
                        ticks: {
                            stepSize: 20,
                            callback: v => v + '%',
                            font: { size: 11 },
                            color: '#666',
                        },
                        grid: {
                            color: 'rgba(102, 126, 234, 0.15)',
                            circular: true,
                        },
                        angleLines: {
                            color: 'rgba(102, 126, 234, 0.2)',
                        },
                        pointLabels: {
                            font: { size: 13, weight: 'bold' },
                            color: '#333',
                        },
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { size: 12 },
                            padding: 20,
                            usePointStyle: true,
                            pointStyle: 'circle',
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleFont: { size: 14, weight: 'bold' },
                        bodyFont: { size: 13 },
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            title: ctx => `维度: ${ctx[0].label}`,
                            label: ctx => `好评率: ${ctx.raw.toFixed(1)}%`,
                            afterLabel: ctx => {
                                const value = ctx.raw;
                                if (value >= 80) return '⭐ 优秀';
                                if (value >= 60) return '👍 良好';
                                if (value >= 40) return '🤔 一般';
                                return '⚠️ 需要改进';
                            }
                        }
                    }
                },
                animation: {
                    duration: 1500,
                    easing: 'easeOutQuart',
                }
            }
        });

        // 柱状图：情感分布
        charts.groupChart = new Chart(document.getElementById('groupChart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: gNames,
                datasets: [
                    { 
                        label: '正向', 
                        data: gNames.map(g => groups[g].positive || 0), 
                        backgroundColor: 'rgba(72, 187, 120, 0.85)',
                        borderColor: 'rgba(72, 187, 120, 1)',
                        borderWidth: 1,
                        borderRadius: 4,
                    },
                    { 
                        label: '中性', 
                        data: gNames.map(g => groups[g].neutral || 0), 
                        backgroundColor: 'rgba(66, 153, 225, 0.85)',
                        borderColor: 'rgba(66, 153, 225, 1)',
                        borderWidth: 1,
                        borderRadius: 4,
                    },
                    { 
                        label: '负向', 
                        data: gNames.map(g => groups[g].negative || 0), 
                        backgroundColor: 'rgba(245, 101, 101, 0.85)',
                        borderColor: 'rgba(245, 101, 101, 1)',
                        borderWidth: 1,
                        borderRadius: 4,
                    },
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            font: { size: 12, weight: 'bold' },
                            padding: 20,
                            usePointStyle: true,
                            pointStyle: 'rect',
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.85)',
                        titleFont: { size: 14, weight: 'bold' },
                        bodyFont: { size: 13 },
                        padding: 12,
                        cornerRadius: 8,
                        callbacks: {
                            title: (ctx) => `维度: ${ctx[0].label}`,
                            afterBody: (ctx) => {
                                const idx = ctx[0].dataIndex;
                                const group = groups[gNames[idx]];
                                const total = group.total || 1;
                                const posRate = Math.round(group.positive / total * 100);
                                return `好评率: ${posRate}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        stacked: true,
                        ticks: {
                            font: { size: 12, weight: 'bold' },
                            color: '#333',
                        },
                        grid: {
                            display: false,
                        },
                        title: {
                            display: true,
                            text: '维度名称',
                            font: { size: 13, weight: 'bold' },
                            color: '#333',
                        }
                    },
                    y: {
                        stacked: true,
                        ticks: {
                            font: { size: 11 },
                            color: '#666',
                        },
                        grid: {
                            color: 'rgba(226, 232, 240, 0.5)',
                        },
                        title: {
                            display: true,
                            text: '数量',
                            font: { size: 13, weight: 'bold' },
                            color: '#333',
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart',
                }
            }
        });
    }

    // 4. 18 维度好评率排名水平柱状图（优化版）- 使用累计数据
    const rankings = getCumulativeRankings();
    if (rankings.length > 0) {
        // 按好评率降序排序
        const sortedRankings = [...rankings].sort((a, b) => (b.positive_ratio || 0) - (a.positive_ratio || 0));
        
        const labels = sortedRankings.map(r => r.category_zh || r.category);
        const posData = sortedRankings.map(r => r.positive_ratio || 0);
        const negData = sortedRankings.map(r => -(r.negative_ratio || 0));
        
        // 生成渐变颜色（好评率越高越绿，越低越红）
        const getBarColor = (ratio, isPositive) => {
            if (isPositive) {
                const r = Math.round(72 - ratio * 0.72);   // 72 -> 0
                const g = Math.round(187 + ratio * 0.68);  // 187 -> 255
                const b = Math.round(120 - ratio * 0.5);   // 120 -> 70
                return `rgb(${r}, ${g}, ${b})`;
            } else {
                const r = Math.round(245 - ratio * 0.45);  // 245 -> 200
                const g = Math.round(101 - ratio * 0.5);   // 101 -> 50
                const b = Math.round(101 - ratio * 0.3);   // 101 -> 70
                return `rgb(${r}, ${g}, ${b})`;
            }
        };
        
        const posColors = posData.map(v => getBarColor(v, true));
        const negColors = negData.map(v => getBarColor(Math.abs(v), false));

        // 生成排名标签（TOP1, TOP2, TOP3, 其余空）
        const generateRankLabels = () => {
            return sortedRankings.map((r, i) => {
                if (i === 0) return '🥇';
                if (i === 1) return '🥈';
                if (i === 2) return '🥉';
                return '';
            });
        };

        const ctx = document.getElementById('rankingChart').getContext('2d');
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(72, 187, 120, 0.1)');
        gradient.addColorStop(0.5, 'rgba(245, 101, 101, 0.05)');
        gradient.addColorStop(1, 'rgba(245, 101, 101, 0.1)');

        charts.rankingChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    { 
                        label: '差评率 %', 
                        data: negData, 
                        backgroundColor: negColors,
                        borderColor: 'rgba(245, 101, 101, 0.3)',
                        borderWidth: 1,
                        borderRadius: 0,
                        barThickness: 24,
                    },
                    { 
                        label: '好评率 %', 
                        data: posData, 
                        backgroundColor: posColors,
                        borderColor: 'rgba(72, 187, 120, 0.3)',
                        borderWidth: 1,
                        borderRadius: 0,
                        barThickness: 24,
                    }
                ]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    intersect: false,
                    mode: 'index',
                },
                plugins: {
                    legend: { 
                        position: 'top',
                        labels: {
                            font: { size: 13, weight: 'bold' },
                            padding: 20,
                            usePointStyle: true,
                            pointStyle: 'rect',
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.85)',
                        titleFont: { size: 14, weight: 'bold' },
                        bodyFont: { size: 13 },
                        padding: 15,
                        cornerRadius: 8,
                        displayColors: true,
                        callbacks: {
                            title: (ctx) => {
                                const idx = ctx[0].dataIndex;
                                const rank = idx + 1;
                                const ranking = sortedRankings[idx];
                                let rankLabel = '';
                                if (rank === 1) rankLabel = ' 🥇 TOP1';
                                else if (rank === 2) rankLabel = ' 🥈 TOP2';
                                else if (rank === 3) rankLabel = ' 🥉 TOP3';
                                return `${ctx[0].label}${rankLabel}`;
                            },
                            label: (ctx) => {
                                const v = Math.abs(ctx.raw);
                                const label = ctx.datasetIndex === 0 ? '差评率' : '好评率';
                                return `${label}: ${v.toFixed(1)}%`;
                            },
                            afterBody: (ctx) => {
                                const idx = ctx[0].dataIndex;
                                const ranking = sortedRankings[idx];
                                const posRatio = ranking.positive_ratio || 0;
                                const total = ranking.total || 0;
                                let level = '';
                                if (posRatio >= 80) level = '⭐ 优秀';
                                else if (posRatio >= 60) level = '👍 良好';
                                else if (posRatio >= 40) level = '🤔 一般';
                                else level = '⚠️ 需要改进';
                                return [`评价等级: ${level}`, `样本数: ${total} 条`];
                            }
                        }
                    },
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                xMin: 0,
                                xMax: 0,
                                yMin: 0,
                                yMax: labels.length,
                                borderColor: '#e2e8f0',
                                borderWidth: 2,
                                borderDash: [5, 5],
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        min: -100,
                        max: 100,
                        ticks: { 
                            callback: (v) => Math.abs(v) + '%',
                            font: { size: 11 },
                            color: '#666',
                        },
                        grid: {
                            color: 'rgba(226, 232, 240, 0.5)',
                            drawBorder: false,
                        },
                        title: {
                            display: true,
                            text: '好评率 / 差评率',
                            font: { size: 13, weight: 'bold' },
                            color: '#333',
                        }
                    },
                    y: {
                        ticks: { 
                            font: { size: 12 },
                            color: '#333',
                            padding: 15,
                            callback: (value, index) => {
                                const ranks = generateRankLabels();
                                const rank = ranks[index];
                                return rank ? `${rank} ${value}` : value;
                            }
                        },
                        grid: {
                            display: false,
                        },
                        title: {
                            display: true,
                            text: '维度名称',
                            font: { size: 13, weight: 'bold' },
                            color: '#333',
                        }
                    }
                },
                animation: {
                    duration: 1200,
                    easing: 'easeOutQuart',
                },
                hover: {
                    mode: 'nearest',
                    intersect: true,
                },
            }
        });
    }
}

// ===== 一级维度详情 =====
function updateGroupDetail(groups) {
    const el = document.getElementById('groupDetail');
    let html = '';
    for (const [name, stats] of Object.entries(groups)) {
        html += `<div class="group-card">
            <strong>${name}</strong>
            <span class="pos">正向 ${stats.positive || 0} (${stats.positive_ratio || 0}%)</span>
            <span class="neu">中性 ${stats.neutral || 0}</span>
            <span class="neg">负向 ${stats.negative || 0} (${stats.negative_ratio || 0}%)</span>
        </div>`;
    }
    el.innerHTML = html;
}

// ===== 评论详情列表（支持分页筛选） =====
function updateReviewsList(results) {
    // 重置筛选状态
    currentPage = 1;
    sentimentFilter = 'all';
    searchKeyword = '';

    // 清空搜索框
    const searchInput = document.getElementById('searchInput');
    if (searchInput) searchInput.value = '';

    // 重置筛选按钮状态
    updateFilterButtons('all');

    // 筛选并显示
    applyFiltersAndRender(results);
}

// ===== 应用筛选并渲染 =====
function applyFiltersAndRender(results) {
    // 1. 情感筛选
    filteredResults = results.filter(r => {
        const overall = (r.summary?.overall_sentiment || 'neutral');
        if (sentimentFilter === 'all') return true;
        return overall === sentimentFilter;
    });

    // 2. 关键词搜索
    if (searchKeyword.trim()) {
        const keyword = searchKeyword.toLowerCase();
        filteredResults = filteredResults.filter(r => {
            const text = (r.text || '').toLowerCase();
            const aspects = (r.aspects || []).map(a => (a.category_zh || '').toLowerCase()).join(' ');
            return text.includes(keyword) || aspects.includes(keyword);
        });
    }

    // 3. 分页计算
    const totalPages = Math.ceil(filteredResults.length / pageSize) || 1;
    if (currentPage > totalPages) currentPage = totalPages;

    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    const pageResults = filteredResults.slice(startIndex, endIndex);

    // 4. 渲染评论列表
    renderReviews(pageResults);

    // 5. 更新筛选信息
    document.getElementById('filterCount').textContent =
        `显示 ${pageResults.length} 条 / 共 ${filteredResults.length} 条`;

    // 6. 更新分页控件
    updatePagination(totalPages);
}

// ===== 渲染评论 =====
function renderReviews(results) {
    const el = document.getElementById('reviewsList');
    el.innerHTML = '';

    if (results.length === 0) {
        el.innerHTML = '<p class="no-aspect">没有符合条件的评论</p>';
        return;
    }

    results.forEach((r, i) => {
        const aspects = r.aspects || [];
        const summary = r.summary || {};
        const overall = summary.overall_sentiment || 'neutral';

        let aspectsHtml = '';
        aspects.forEach(a => {
            const cls = 'sentiment-' + a.sentiment;
            aspectsHtml += `<span class="aspect-tag ${cls}">${a.category_zh}: ${a.sentiment_zh} (${(a.sentiment_confidence*100).toFixed(0)}%)</span>`;
        });

        const div = document.createElement('div');
        div.className = `review-item ${overall}`;
        
        const analysisInfo = r.analysisIndex ? `
            <div class="analysis-tag">第 ${r.analysisIndex} 次分析 · ${r.analysisTime}</div>
        ` : '';
        
        div.innerHTML = `
            ${analysisInfo}
            <div class="review-text">${r.text || '(空)'}</div>
            ${aspectsHtml ? '<div class="aspects-list">' + aspectsHtml + '</div>' : '<p class="no-aspect">未检测到评价维度</p>'}
            <div class="review-meta">
                <span class="sentiment-badge ${overall}">整体: ${overall === 'positive' ? '正向' : overall === 'negative' ? '负向' : '中性'}</span>
                <span>${aspects.length} 个维度</span>
            </div>
        `;
        el.appendChild(div);
    });
}

// ===== 更新分页控件 =====
function updatePagination(totalPages) {
    const pagination = document.getElementById('pagination');
    const pageInfo = document.getElementById('pageInfo');

    if (filteredResults.length <= pageSize) {
        pagination.style.display = 'none';
        return;
    }

    pagination.style.display = 'flex';
    pageInfo.textContent = `第 ${currentPage} / ${totalPages} 页`;

    // 更新按钮状态
    const prevBtn = pagination.querySelector('.page-btn:first-child');
    const nextBtn = pagination.querySelector('.page-btn:last-child');

    prevBtn.disabled = currentPage <= 1;
    nextBtn.disabled = currentPage >= totalPages;
}

// ===== 分页操作 =====
function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        applyFiltersAndRender(currentData?.results || []);
    }
}

function nextPage() {
    const totalPages = Math.ceil(filteredResults.length / pageSize);
    if (currentPage < totalPages) {
        currentPage++;
        applyFiltersAndRender(currentData?.results || []);
    }
}

// ===== 筛选操作 =====
function setSentimentFilter(filter) {
    sentimentFilter = filter;
    currentPage = 1;
    updateFilterButtons(filter);
    applyFiltersAndRender(currentData?.results || []);
}

function updateFilterButtons(activeFilter) {
    const buttons = document.querySelectorAll('.filter-btn');
    buttons.forEach(btn => {
        const filter = btn.getAttribute('onclick').match(/'(\w+)'/)?.[1];
        if (filter === activeFilter) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

function filterReviews() {
    searchKeyword = document.getElementById('searchInput').value;
    currentPage = 1;
    applyFiltersAndRender(currentData?.results || []);
}

// ===== 手动输入结果时构建统计 =====
function _buildGroupStats(aspects) {
    const groups = {};
    aspects.forEach(a => {
        const g = a.group || '其他';
        if (!groups[g]) groups[g] = { total: 0, positive: 0, neutral: 0, negative: 0 };
        groups[g].total++;
        groups[g][a.sentiment]++;
    });
    for (const [k, v] of Object.entries(groups)) {
        v.positive_ratio = v.total > 0 ? Math.round(v.positive / v.total * 100) : 0;
        v.negative_ratio = v.total > 0 ? Math.round(v.negative / v.total * 100) : 0;
    }
    return groups;
}

function _buildCategoryRankings(aspects) {
    const cats = {};
    aspects.forEach(a => {
        const key = a.category;
        if (!cats[key]) cats[key] = { category: key, category_zh: a.category_zh, total: 0, positive: 0, negative: 0 };
        cats[key].total++;
        if (a.sentiment === 'positive') cats[key].positive++;
        if (a.sentiment === 'negative') cats[key].negative++;
    });
    const rankings = Object.values(cats).map(c => ({
        ...c,
        positive_ratio: c.total > 0 ? Math.round(c.positive / c.total * 100) : 0,
        negative_ratio: c.total > 0 ? Math.round(c.negative / c.total * 100) : 0,
    }));
    rankings.sort((a, b) => b.positive_ratio - a.positive_ratio);
    return rankings;
}

// ===== 工具函数 =====
function showLoading() { loadingIndicator.style.display = 'block'; resultsSection.style.display = 'none'; hideError(); }
function hideLoading() { loadingIndicator.style.display = 'none'; }
function showError(msg) { errorText.textContent = msg; errorMessage.style.display = 'block'; resultsSection.style.display = 'none'; }
function hideError() { errorMessage.style.display = 'none'; }

// ===== 暂存功能 =====

// 暂存当前结果
function stashCurrentResults() {
    if (!currentData) {
        showError('当前没有可暂存的分析结果');
        return;
    }

    const stashItem = {
        id: Date.now(),
        timestamp: new Date().toLocaleString('zh-CN'),
        name: selectedFiles.length > 0 ? `批量分析 (${selectedFiles.length}个文件)` : '单条评论分析',
        data: JSON.parse(JSON.stringify(currentData))
    };

    stashList.push(stashItem);
    updateStashCount();
    alert(`已暂存分析结果，当前共有 ${stashList.length} 条暂存记录`);
}

// 更新暂存计数
function updateStashCount() {
    document.getElementById('stashCount').textContent = stashList.length;
}

// 显示暂存列表
function showStashList() {
    const modal = document.getElementById('stashModal');
    const listContainer = document.getElementById('stashList');

    if (stashList.length === 0) {
        listContainer.innerHTML = '<div class="stash-empty">暂无暂存的分析结果<br><br>点击上方"暂存结果"按钮保存当前分析</div>';
    } else {
        listContainer.innerHTML = stashList.map((item, index) => `
            <div class="stash-item">
                <div class="stash-item-info">
                    <div class="stash-item-title">${item.name}</div>
                    <div class="stash-item-meta">${item.timestamp} | ${item.data.summary?.total_reviews || 0} 条评论</div>
                </div>
                <div class="stash-item-actions">
                    <button class="stash-item-btn view" onclick="loadStashed(${index})">查看</button>
                    <button class="stash-item-btn delete" onclick="deleteStashed(${index})">删除</button>
                </div>
            </div>
        `).join('');
    }

    modal.classList.add('show');
}

// 关闭暂存模态框
function closeStashModal(event) {
    if (!event || event.target.id === 'stashModal') {
        document.getElementById('stashModal').classList.remove('show');
    }
}

// 加载单条暂存
function loadStashed(index) {
    const item = stashList[index];
    if (item) {
        currentData = JSON.parse(JSON.stringify(item.data));
        displayResults(currentData);
        closeStashModal();
    }
}

// 删除单条暂存
function deleteStashed(index) {
    stashList.splice(index, 1);
    updateStashCount();
    showStashList();
}

// 加载全部暂存数据
function loadAllStashed() {
    if (stashList.length === 0) {
        alert('没有暂存的数据');
        return;
    }

    // 合并所有暂存数据
    let allResults = [];
    let groupStats = {};
    let categoryRankings = {};

    stashList.forEach(item => {
        const results = item.data.results || [];
        allResults = allResults.concat(results.map(r => ({
            ...r,
            source: item.name
        })));
    });

    // 重新计算汇总
    const mergedData = {
        summary: _computeMergedSummary(allResults),
        results: allResults
    };

    currentData = mergedData;
    displayResults(mergedData);
    closeStashModal();
    alert(`已加载 ${allResults.length} 条评论进行统一分析`);
}

// 计算合并后的汇总
function _computeMergedSummary(results) {
    const allAspects = [];
    results.forEach(r => {
        allAspects.push(...(r.aspects || []));
    });

    // 按组统计
    const groupStats = {};
    allAspects.forEach(a => {
        const g = a.group || '其他';
        if (!groupStats[g]) groupStats[g] = { total: 0, positive: 0, neutral: 0, negative: 0 };
        groupStats[g].total++;
        groupStats[g][a.sentiment]++;
    });

    // 按维度统计
    const catStats = {};
    allAspects.forEach(a => {
        const cat = a.category;
        if (!catStats[cat]) catStats[cat] = { total: 0, positive: 0, neutral: 0, negative: 0, category_zh: a.category_zh };
        catStats[cat].total++;
        catStats[cat][a.sentiment]++;
    });

    const groupResult = {};
    for (const [g, s] of Object.entries(groupStats)) {
        const t = Math.max(s.total, 1);
        groupResult[g] = {
            total: s.total,
            positive: s.positive,
            neutral: s.neutral,
            negative: s.negative,
            positive_ratio: Math.round(s.positive / t * 100, 1),
            negative_ratio: Math.round(s.negative / t * 100, 1),
        };
    }

    const rankings = Object.entries(catStats).map(([cat, s]) => {
        const t = Math.max(s.total, 1);
        return {
            category: cat,
            category_zh: s.category_zh,
            total: s.total,
            positive_ratio: Math.round(s.positive / t * 100, 1),
            negative_ratio: Math.round(s.negative / t * 100, 1),
        };
    });
    rankings.sort((a, b) => b.positive_ratio - a.positive_ratio);

    // 整体情感分布
    const reviewSentiments = results.map(r => r.summary?.overall_sentiment || 'neutral');
    const sentimentCount = { positive: 0, neutral: 0, negative: 0 };
    reviewSentiments.forEach(s => sentimentCount[s]++);

    const aspectSentiments = { positive: 0, neutral: 0, negative: 0 };
    allAspects.forEach(a => aspectSentiments[a.sentiment]++);

    return {
        total_reviews: results.length,
        total_aspects: allAspects.length,
        avg_aspects_per_review: results.length > 0 ? (allAspects.length / results.length).toFixed(1) : 0,
        review_sentiment_dist: {
            positive: sentimentCount.positive,
            neutral: sentimentCount.neutral,
            negative: sentimentCount.negative,
            positive_ratio: Math.round(sentimentCount.positive / results.length * 100, 1),
            negative_ratio: Math.round(sentimentCount.negative / results.length * 100, 1),
        },
        aspect_sentiment_dist: {
            positive: aspectSentiments.positive,
            neutral: aspectSentiments.neutral,
            negative: aspectSentiments.negative,
            positive_ratio: Math.round(aspectSentiments.positive / Math.max(allAspects.length, 1) * 100, 1),
            negative_ratio: Math.round(aspectSentiments.negative / Math.max(allAspects.length, 1) * 100, 1),
        },
        group_stats: groupResult,
        category_rankings: rankings,
    };
}

// 清空全部暂存
function clearAllStashed() {
    if (stashList.length === 0) {
        alert('没有可清空的数据');
        return;
    }
    if (confirm(`确定要清空全部 ${stashList.length} 条暂存记录吗？`)) {
        stashList = [];
        updateStashCount();
        closeStashModal();
        alert('已清空全部暂存记录');
    }
}

// ===== 导出功能 =====

// 切换导出菜单
function toggleExportMenu() {
    const menu = document.getElementById('exportMenu');
    menu.classList.toggle('show');
}

// 点击其他区域关闭导出菜单
document.addEventListener('click', (e) => {
    if (!e.target.closest('.export-dropdown')) {
        document.getElementById('exportMenu')?.classList.remove('show');
    }
});

// 导出为 Excel
function exportToExcel() {
    document.getElementById('exportMenu').classList.remove('show');

    if (!currentData || !currentData.results) {
        showError('没有可导出的数据');
        return;
    }

    const data = currentData.results;

    // 构建导出数据
    const exportData = data.map((r, i) => {
        const aspects = r.aspects || [];
        const summary = r.summary || {};

        return {
            '序号': i + 1,
            '评论内容': r.text || '',
            '整体情感': summary.overall_sentiment === 'positive' ? '正向' : summary.overall_sentiment === 'negative' ? '负向' : '中性',
            '检测维度数': aspects.length,
            '维度详情': aspects.map(a => `${a.category_zh}:${a.sentiment_zh}`).join('; '),
            '来源': r.source || '当前分析',
        };
    });

    const ws = XLSX.utils.json_to_sheet(exportData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, '评论分析结果');

    // 设置列宽
    ws['!cols'] = [
        { wch: 6 },   // 序号
        { wch: 50 },  // 评论内容
        { wch: 10 },  // 整体情感
        { wch: 10 },  // 检测维度数
        { wch: 40 },  // 维度详情
        { wch: 20 },  // 来源
    ];

    const timestamp = new Date().toISOString().slice(0, 10);
    XLSX.writeFile(wb, `ABSA分析结果_${timestamp}.xlsx`);
}

// 导出为 CSV
function exportToCSV() {
    document.getElementById('exportMenu').classList.remove('show');

    if (!currentData || !currentData.results) {
        showError('没有可导出的数据');
        return;
    }

    const data = currentData.results;

    // CSV 头部
    let csv = '\uFEFF'; // BOM for UTF-8
    csv += '序号,评论内容,整体情感,检测维度数,维度详情\n';

    // CSV 数据行
    data.forEach((r, i) => {
        const aspects = r.aspects || [];
        const summary = r.summary || {};
        const sentiment = summary.overall_sentiment === 'positive' ? '正向' : summary.overall_sentiment === 'negative' ? '负向' : '中性';
        const aspectsText = aspects.map(a => `${a.category_zh}:${a.sentiment_zh}`).join(';');

        // 处理评论内容中的特殊字符
        const text = (r.text || '').replace(/"/g, '""');

        csv += `${i + 1},"${text}",${sentiment},${aspects.length},"${aspectsText}"\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const timestamp = new Date().toISOString().slice(0, 10);
    a.href = url;
    a.download = `ABSA分析结果_${timestamp}.csv`;
    a.click();
    URL.revokeObjectURL(url);
}

// 导出暂存的全部数据
function exportStashedToExcel() {
    if (stashList.length === 0) {
        alert('没有可导出的暂存数据');
        return;
    }

    // 合并所有数据
    let allResults = [];
    stashList.forEach(item => {
        const results = item.data.results || [];
        allResults = allResults.concat(results.map(r => ({
            ...r,
            source: item.name
        })));
    });

    const exportData = allResults.map((r, i) => {
        const aspects = r.aspects || [];
        const summary = r.summary || {};

        return {
            '序号': i + 1,
            '评论内容': r.text || '',
            '整体情感': summary.overall_sentiment === 'positive' ? '正向' : summary.overall_sentiment === 'negative' ? '负向' : '中性',
            '检测维度数': aspects.length,
            '维度详情': aspects.map(a => `${a.category_zh}:${a.sentiment_zh}`).join('; '),
            '来源': r.source || '',
        };
    });

    const ws = XLSX.utils.json_to_sheet(exportData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, '评论分析结果');

    ws['!cols'] = [
        { wch: 6 },
        { wch: 50 },
        { wch: 10 },
        { wch: 10 },
        { wch: 40 },
        { wch: 20 },
    ];

    const timestamp = new Date().toISOString().slice(0, 10);
    XLSX.writeFile(wb, `ABSA分析结果_全部暂存_${timestamp}.xlsx`);
    closeStashModal();
}
