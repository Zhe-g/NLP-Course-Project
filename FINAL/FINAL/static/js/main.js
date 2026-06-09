// ===== ABSA 细粒度情感分析 — 前端逻辑 =====

let charts = {};
let currentData = null;

// DOM elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const modelStatus = document.getElementById('modelStatus');

// ===== 初始化 =====
checkModelHealth();
fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        dropZone.style.backgroundColor = '#f0f4ff';
        dropZone.style.borderColor = '#667eea';
    }
});
uploadBtn.addEventListener('click', handleFileUpload);
analyzeBtn.addEventListener('click', handleTextAnalysis);

// 拖拽上传
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        handleFileUpload();
    }
});
dropZone.addEventListener('click', () => fileInput.click());

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

// ===== 文件上传 =====
async function handleFileUpload() {
    const files = fileInput.files;
    if (!files || files.length === 0) { showError('请选择文件'); return; }

    showLoading();
    if (files.length === 1) {
        await uploadSingle(files[0]);
    } else {
        await uploadMultiple(files);
    }
    hideLoading();
}

async function uploadSingle(file) {
    const fd = new FormData();
    fd.append('file', file);
    try {
        const resp = await fetch('/upload', { method: 'POST', body: fd });
        const data = await resp.json();
        if (resp.ok) { currentData = data; displayResults(data); }
        else { showError(data.error || '分析失败'); }
    } catch (e) { showError('网络错误: ' + e.message); }
}

async function uploadMultiple(files) {
    const fd = new FormData();
    for (let f of files) fd.append('files', f);
    try {
        const resp = await fetch('/batch-upload', { method: 'POST', body: fd });
        const data = await resp.json();
        if (resp.ok) { currentData = data; displayResults(data); }
        else { showError(data.error || '批量分析失败'); }
    } catch (e) { showError('网络错误: ' + e.message); }
}

// ===== 手动输入 =====
async function handleTextAnalysis() {
    const text = textInput.value.trim();
    if (!text) { showError('请输入评论内容'); return; }
    showLoading();
    try {
        const resp = await fetch('/input-text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        const data = await resp.json();
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
    } catch (e) { showError('网络错误: ' + e.message); }
}

// ===== 显示结果 =====
function displayResults(data) {
    resultsSection.style.display = 'block';
    updateSummary(data.summary);
    updateCharts(data);
    updateGroupDetail(data.summary.group_stats || {});
    updateReviewsList(data.results || []);
    resultsSection.scrollIntoView({ behavior: 'smooth' });
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

    // 3. 5 个一级维度分组柱状图
    const groups = s.group_stats || {};
    const gNames = Object.keys(groups);
    if (gNames.length > 0) {
        charts.groupChart = new Chart(document.getElementById('groupChart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: gNames,
                datasets: [
                    { label: '正向', data: gNames.map(g => groups[g].positive || 0), backgroundColor: '#48bb78' },
                    { label: '中性', data: gNames.map(g => groups[g].neutral || 0), backgroundColor: '#4299e1' },
                    { label: '负向', data: gNames.map(g => groups[g].negative || 0), backgroundColor: '#f56565' },
                ]
            },
            options: {
                responsive: true,
                scales: { x: { stacked: true }, y: { stacked: true } },
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    // 4. 18 维度好评率排名水平柱状图
    const rankings = s.category_rankings || [];
    if (rankings.length > 0) {
        const labels = rankings.map(r => r.category_zh || r.category);
        const posData = rankings.map(r => r.positive_ratio || 0);
        const negData = rankings.map(r => -(r.negative_ratio || 0)); // 负值向左

        charts.rankingChart = new Chart(document.getElementById('rankingChart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    { label: '好评率 %', data: posData, backgroundColor: '#48bb78' },
                    { label: '差评率 %', data: negData, backgroundColor: '#f56565' },
                ]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                scales: {
                    x: {
                        min: -100,
                        max: 100,
                        ticks: { callback: v => Math.abs(v) + '%' }
                    }
                },
                plugins: {
                    legend: { position: 'bottom' },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const v = Math.abs(ctx.raw);
                                return (ctx.datasetIndex === 0 ? '好评率: ' : '差评率: ') + v.toFixed(1) + '%';
                            }
                        }
                    }
                }
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

// ===== 评论详情列表 =====
function updateReviewsList(results) {
    const el = document.getElementById('reviewsList');
    el.innerHTML = '';
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
        div.innerHTML = `
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
