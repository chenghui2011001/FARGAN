#!/usr/bin/env python3
"""
FarGanRVQ 实时 Web 监控界面
基于 Streamlit 的动态训练可视化
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import glob
import os
import torch
import time
from datetime import datetime
import json

# 设置页面配置
st.set_page_config(
    page_title="FarGanRVQ 训练监控",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-running {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .status-stopped {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=5)  # 缓存5秒
def load_checkpoint_data(checkpoint_dir):
    """加载检查点数据"""
    pattern = os.path.join(checkpoint_dir, "optfargan_*.pt")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None
    
    data = []
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        try:
            step = int(filename.split('_')[1].split('.')[0])
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            mtime = os.path.getmtime(file_path)
            
            # 尝试加载检查点信息
            try:
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
                loss = checkpoint.get('loss', 0)
                epoch = checkpoint.get('epoch', 0)
            except:
                loss = 0
                epoch = 0
            
            data.append({
                'step': step,
                'loss': loss,
                'epoch': epoch,
                'size_mb': size_mb,
                'timestamp': mtime,
                'datetime': datetime.fromtimestamp(mtime),
                'file': file_path
            })
        except (ValueError, IndexError):
            continue
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    df = df.sort_values('step')
    return df

def create_loss_chart(df):
    """创建损失曲线图"""
    fig = go.Figure()
    
    # 原始损失曲线
    fig.add_trace(go.Scatter(
        x=df['step'],
        y=df['loss'],
        mode='lines+markers',
        name='训练损失',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    
    # 移动平均
    if len(df) > 10:
        window = min(50, len(df) // 4)
        ma_loss = df['loss'].rolling(window=window, center=True).mean()
        fig.add_trace(go.Scatter(
            x=df['step'],
            y=ma_loss,
            mode='lines',
            name=f'移动平均({window})',
            line=dict(color='#ff7f0e', width=3, dash='dash')
        ))
    
    fig.update_layout(
        title="训练损失曲线",
        xaxis_title="训练步数",
        yaxis_title="L1 损失",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_speed_chart(df):
    """创建训练速度图"""
    if len(df) < 2:
        return go.Figure()
    
    # 计算训练速度
    df_copy = df.copy()
    df_copy['time_diff'] = df_copy['timestamp'].diff()
    df_copy['step_diff'] = df_copy['step'].diff()
    df_copy['speed'] = (df_copy['step_diff'] / df_copy['time_diff']) * 60  # 步数/分钟
    
    # 去除异常值
    df_speed = df_copy[df_copy['speed'].between(0, df_copy['speed'].quantile(0.95))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_speed['step'],
        y=df_speed['speed'],
        mode='lines+markers',
        name='训练速度',
        line=dict(color='#2ca02c', width=2),
        marker=dict(size=4)
    ))
    
    # 平均速度线
    avg_speed = df_speed['speed'].mean()
    fig.add_hline(
        y=avg_speed, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"平均: {avg_speed:.1f} 步/分"
    )
    
    fig.update_layout(
        title="训练速度",
        xaxis_title="训练步数",
        yaxis_title="步数/分钟",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_system_metrics(df):
    """创建系统指标图"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("文件大小变化", "训练频率分布"),
        vertical_spacing=0.1
    )
    
    # 文件大小
    fig.add_trace(
        go.Scatter(
            x=df['step'],
            y=df['size_mb'],
            mode='lines+markers',
            name='检查点大小',
            line=dict(color='#d62728', width=2)
        ),
        row=1, col=1
    )
    
    # 训练时间分布（每小时）
    df_copy = df.copy()
    df_copy['hour'] = df_copy['datetime'].dt.hour
    hourly_counts = df_copy['hour'].value_counts().sort_index()
    
    fig.add_trace(
        go.Bar(
            x=hourly_counts.index,
            y=hourly_counts.values,
            name='每小时训练量',
            marker_color='#9467bd'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="训练步数", row=1, col=1)
    fig.update_yaxes(title_text="文件大小 (MB)", row=1, col=1)
    fig.update_xaxes(title_text="小时", row=2, col=1)
    fig.update_yaxes(title_text="检查点数量", row=2, col=1)
    
    fig.update_layout(height=600, template="plotly_white")
    
    return fig

def main():
    # 标题和描述
    st.title("🎵 FarGanRVQ 训练监控面板")
    st.markdown("实时监控 FarGan 模型训练进度")
    
    # 侧边栏配置
    st.sidebar.header("⚙️ 配置")
    
    checkpoint_dir = st.sidebar.text_input(
        "检查点目录", 
        value="dnn/FarGanRVQ/checkpoints/test",
        help="包含训练检查点的目录路径"
    )
    
    refresh_interval = st.sidebar.slider(
        "自动刷新间隔 (秒)", 
        min_value=1, 
        max_value=60, 
        value=5
    )
    
    auto_refresh = st.sidebar.checkbox("自动刷新", value=True)
    
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    # 手动刷新按钮
    if st.sidebar.button("🔄 立即刷新"):
        st.cache_data.clear()
        st.rerun()
    
    # 加载数据
    df = load_checkpoint_data(checkpoint_dir)
    
    if df is None or len(df) == 0:
        st.error(f"❌ 在目录 {checkpoint_dir} 中未找到检查点文件")
        st.info("💡 请确保训练脚本正在运行并生成检查点文件")
        return
    
    # 基础统计信息
    latest_data = df.iloc[-1]
    total_steps = latest_data['step']
    total_checkpoints = len(df)
    latest_loss = latest_data['loss']
    best_loss = df['loss'].min()
    
    # 判断训练状态
    last_update_time = latest_data['timestamp']
    current_time = time.time()
    is_training = (current_time - last_update_time) < 60  # 1分钟内有更新
    
    # 状态卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_class = "status-running" if is_training else "status-stopped"
        status_text = "🟢 训练中" if is_training else "🔴 已停止"
        st.markdown(f"""
        <div class="metric-card {status_class}">
            <h3>{status_text}</h3>
            <p>最后更新: {latest_data['datetime'].strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            label="📊 总训练步数",
            value=f"{total_steps:,}",
            delta=f"{total_checkpoints} 个检查点"
        )
    
    with col3:
        loss_change = latest_loss - best_loss
        st.metric(
            label="📉 当前损失",
            value=f"{latest_loss:.6f}",
            delta=f"{loss_change:.6f}" if loss_change != 0 else None
        )
    
    with col4:
        st.metric(
            label="🏆 最佳损失",
            value=f"{best_loss:.6f}",
            delta=f"步数 {df[df['loss'] == best_loss]['step'].iloc[0]}"
        )
    
    # 主要图表
    st.header("📈 训练曲线")
    
    # 两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        loss_fig = create_loss_chart(df)
        st.plotly_chart(loss_fig, use_container_width=True)
    
    with col2:
        speed_fig = create_speed_chart(df)
        st.plotly_chart(speed_fig, use_container_width=True)
    
    # 系统指标
    st.header("🖥️ 系统指标")
    system_fig = create_system_metrics(df)
    st.plotly_chart(system_fig, use_container_width=True)
    
    # 详细数据表
    st.header("📋 详细数据")
    
    # 显示选项
    show_recent = st.checkbox("只显示最近50个检查点", value=True)
    
    display_df = df.tail(50) if show_recent else df
    
    # 格式化显示
    display_df_formatted = display_df.copy()
    display_df_formatted['时间'] = display_df_formatted['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df_formatted['损失'] = display_df_formatted['loss'].round(6)
    display_df_formatted['大小(MB)'] = display_df_formatted['size_mb'].round(2)
    
    st.dataframe(
        display_df_formatted[['step', '时间', '损失', 'epoch', '大小(MB)']].rename(columns={
            'step': '步数',
            'epoch': '轮次'
        }),
        use_container_width=True
    )
    
    # 导出功能
    st.header("💾 数据导出")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 导出 CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="下载 CSV 文件",
                data=csv,
                file_name=f"fargan_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 导出 JSON"):
            json_data = df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="下载 JSON 文件",
                data=json_data,
                file_name=f"fargan_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # 页脚信息
    st.markdown("---")
    st.markdown(f"""
    **📊 监控统计**  
    - 数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - 监控的检查点: {total_checkpoints} 个
    - 数据覆盖步数: {df['step'].min():,} - {df['step'].max():,}
    """)

if __name__ == "__main__":
    main() 