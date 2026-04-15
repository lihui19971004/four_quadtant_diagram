import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from matplotlib import colors as mcolors
from typing import Optional, Tuple
import io

st.set_page_config(page_title="波士顿矩阵图工具", layout="wide", page_icon="📊")

def transform_coordinate(value, origin, compress_factor=0.5):
    diff = value - origin
    if diff <= 0.5 and diff >= -1:
        return diff
    elif diff > 0.5:
        return 0.5 + (diff - 0.5) * compress_factor
    elif diff < -1:
        return -1 + (diff + 1) * compress_factor
    else:
        return diff

def format_number(value):
    """格式化数字：大于1亿用亿，大于100用万，否则保留小数"""
    if pd.isna(value) or value == 0:
        return "0"
    abs_val = abs(value)
    if abs_val >= 100000000:
        return f"{value/100000000:.2f}亿"
    elif abs_val >= 10000:
        return f"{value/10000:.1f}万"
    elif abs_val >= 100:
        return f"{value:.0f}"
    else:
        return f"{value:.1f}"

def create_boston_matrix(
    df: pd.DataFrame,
    title: str = "波士顿矩阵图",
    xy_origin: Optional[Tuple[float, float]] = None,
    height: int = 800,
    width: int = 1000,
    bubble_outer_multiplier: float = 150,
    bubble_outer_base: float = 20,
    bubble_inner_multiplier: float = 130,
    bubble_inner_base: float = 15,
    text_size: int = 12,
    category_col: str = None,
    x_col: str = None,
    y_col: str = None,
    outer_size_col: str = None,
    inner_size_col: str = None,
    has_inner_size: bool = True,
    text_positions: dict = None,
    compress_factor: float = 0.5
) -> go.Figure:
    
    df = df.copy()
    df = df.sort_values(outer_size_col, ascending=False).reset_index(drop=True)

    if xy_origin is None:
        x_origin = df[x_col].median()
        y_origin = df[y_col].median()
    else:
        x_origin, y_origin = xy_origin
    
    df['x_transformed'] = df[x_col].apply(lambda x: transform_coordinate(x, x_origin, compress_factor))
    df['y_transformed'] = df[y_col].apply(lambda y: transform_coordinate(y, y_origin, compress_factor))
    
    max_outer = df[outer_size_col].max()
    if max_outer == 0 or pd.isna(max_outer):
        max_outer = 1
    
    if has_inner_size and inner_size_col:
        max_inner = df[inner_size_col].max()
        if max_inner == 0 or pd.isna(max_inner):
            max_inner = max_outer
        df['bubble_size_inner'] = np.sqrt(np.abs(df[inner_size_col]) / max_inner) * bubble_inner_multiplier + bubble_inner_base
    else:
        max_inner = max_outer
        df['bubble_size_inner'] = np.sqrt(np.abs(df[outer_size_col]) / max_outer) * bubble_inner_multiplier + bubble_inner_base
    
    df['bubble_size_outer'] = np.sqrt(np.abs(df[outer_size_col]) / max_outer) * bubble_outer_multiplier + bubble_outer_base
    
    df['bubble_size_outer'] = df['bubble_size_outer'].fillna(bubble_outer_base)
    df['bubble_size_inner'] = df['bubble_size_inner'].fillna(bubble_inner_base)
    
    x_col_is_percent = df[x_col].abs().max() <= 1.0
    y_col_is_percent = df[y_col].abs().max() <= 1.0
    
    fig = go.Figure()
    
    colors = [
        '#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2', '#EF476F',
        '#073B4C', '#7209B7', '#3A86FF', '#FB5607', '#8338EC', '#FF006E',
        '#3A86FF', '#FFBE0B', '#FB5607', '#FF006E', '#8338EC', '#3A86FF'
    ]
    
    fig.add_shape(
        type="rect",
        x0=df['x_transformed'].min() - 0.5,
        y0=df['y_transformed'].min() - 0.5,
        x1=0,
        y1=0,
        fillcolor="rgba(255, 182, 193, 0.15)",
        line=dict(width=0)
    )
    
    fig.add_shape(
        type="rect",
        x0=0,
        y0=df['y_transformed'].min() - 0.5,
        x1=df['x_transformed'].max() + 0.5,
        y1=0,
        fillcolor="rgba(144, 238, 144, 0.15)",
        line=dict(width=0)
    )
    
    fig.add_shape(
        type="rect",
        x0=df['x_transformed'].min() - 0.5,
        y0=0,
        x1=0,
        y1=df['y_transformed'].max() + 0.5,
        fillcolor="rgba(173, 216, 230, 0.15)",
        line=dict(width=0)
    )
    
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=df['x_transformed'].max() + 0.5,
        y1=df['y_transformed'].max() + 0.5,
        fillcolor="rgba(255, 255, 153, 0.15)",
        line=dict(width=0)
    )
    
    fig.add_shape(
        type="line",
        x0=df['x_transformed'].min() - 0.5,
        y0=0,
        x1=df['x_transformed'].max() + 0.5,
        y1=0,
        line=dict(color="gray", width=1.5, dash="dot")
    )
    
    fig.add_shape(
        type="line",
        x0=0,
        y0=df['y_transformed'].min() - 0.5,
        x1=0,
        y1=df['y_transformed'].max() + 0.5,
        line=dict(color="gray", width=1.5, dash="dot")
    )

    for idx, row in df.iterrows():
        category = row[category_col]
        color_idx = idx % len(colors)
        base_color = colors[color_idx]
        
        rgb = mcolors.hex2color(base_color)
        inner_color = f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.9)'
        outer_color = f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.3)'
        
        outer_size_pct = (row[outer_size_col] / df[outer_size_col].sum()) * 100
        outer_size_str = format_number(row[outer_size_col])
        if has_inner_size and inner_size_col:
            inner_size_pct = (row[inner_size_col] / df[inner_size_col].sum()) * 100
            inner_size_str = format_number(row[inner_size_col])
            x_val_str = f"{row[x_col]*100:.1f}%" if x_col_is_percent else format_number(row[x_col])
            y_val_str = f"{row[y_col]*100:.1f}%" if y_col_is_percent else format_number(row[y_col])
            hover_text = (
                f"<b>{category}</b><br><br>"
                f"<b>{x_col}:</b> {x_val_str}<br>"
                f"<b>{y_col}:</b> {y_val_str}<br>"
                f"<b>{outer_size_col}:</b> {outer_size_str} ({outer_size_pct:.1f}%)<br>"
                f"<b>{inner_size_col}:</b> {inner_size_str} ({inner_size_pct:.1f}%)"
            )
        else:
            x_val_str = f"{row[x_col]*100:.1f}%" if x_col_is_percent else format_number(row[x_col])
            y_val_str = f"{row[y_col]*100:.1f}%" if y_col_is_percent else format_number(row[y_col])
            hover_text = (
                f"<b>{category}</b><br><br>"
                f"<b>{x_col}:</b> {x_val_str}<br>"
                f"<b>{y_col}:</b> {y_val_str}<br>"
                f"<b>{outer_size_col}:</b> {outer_size_str} ({outer_size_pct:.1f}%)"
            )
        
        if x_col_is_percent and y_col_is_percent:
            display_text = f"{category}<br>{row[x_col]*100:.0f}%/{row[y_col]*100:.0f}%"
        elif x_col_is_percent:
            display_text = f"{category}<br>{row[x_col]*100:.0f}%/{format_number(row[y_col])}"
        elif y_col_is_percent:
            display_text = f"{category}<br>{format_number(row[x_col])}/{row[y_col]*100:.0f}%"
        else:
            display_text = f"{category}<br>{format_number(row[x_col])}/{format_number(row[y_col])}"
        
        if text_positions and category in text_positions:
            text_position = text_positions[category]
        else:
            text_position = 'middle center'
        
        # 添加外层气泡
        fig.add_trace(go.Scatter(
            x=[row['x_transformed']],
            y=[row['y_transformed']],
            mode='markers+text',
            name=category,
            marker=dict(
                size=row['bubble_size_outer'],
                color=outer_color,
                line=dict(width=0),
                opacity=0.7
            ),
            text='',
            textposition='middle center',
            hoverinfo='skip',
            showlegend=True,
            legendgroup=category
        ))
        
        # 添加内层气泡
        fig.add_trace(go.Scatter(
            x=[row['x_transformed']],
            y=[row['y_transformed']],
            mode='markers+text',
            name='',
            marker=dict(
                size=row['bubble_size_inner'],
                color=inner_color,
                line=dict(width=0),
                opacity=0.9
            ),
            text=display_text,
            textposition=text_position,
            textfont=dict(
                size=text_size,
                color='black',
                family="Arial, sans-serif"
            ),
            hoverinfo='text',
            hovertext=hover_text,
            showlegend=False,
            legendgroup=category
        ))

    min_x_val = df[x_col].min()
    max_x_val = df[x_col].max()
    min_y_val = df[y_col].min()
    max_y_val = df[y_col].max()
    
    x_col_is_percent = df[x_col].abs().max() <= 1.0
    y_col_is_percent = df[y_col].abs().max() <= 1.0
    
    x_tick_values = np.linspace(min_x_val, max_x_val, 8)
    x_ticks_positions = []
    x_ticks_labels = []
    for value in x_tick_values:
        tick_pos = transform_coordinate(value, x_origin, compress_factor)
        x_ticks_positions.append(tick_pos)
        if x_col_is_percent:
            x_ticks_labels.append(f"{value*100:.1f}%")
        else:
            x_ticks_labels.append(format_number(value))
    
    y_tick_values = np.linspace(min_y_val, max_y_val, 8)
    y_ticks_positions = []
    y_ticks_labels = []
    for value in y_tick_values:
        tick_pos = transform_coordinate(value, y_origin, compress_factor)
        y_ticks_positions.append(tick_pos)
        if y_col_is_percent:
            y_ticks_labels.append(f"{value*100:.1f}%")
        else:
            y_ticks_labels.append(format_number(value))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(size=20, family="Arial, sans-serif", color="#2c3e50"),
            xanchor="center"
        ),
        height=height,
        width=width,
        xaxis=dict(
            title=dict(text=f"<b>{x_col}</b>", font=dict(size=14)),
            tickmode='array',
            tickvals=x_ticks_positions,
            ticktext=x_ticks_labels,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
            showgrid=True,
            gridwidth=1,
            range=[df['x_transformed'].min() - 0.3, df['x_transformed'].max() + 0.3],
            showline=True,
            linewidth=1,
            linecolor='gray',
            mirror=False,
            zerolinewidth=0
        ),
        yaxis=dict(
            title=dict(text=f"<b>{y_col}</b>", font=dict(size=14)),
            tickmode='array',
            tickvals=y_ticks_positions,
            ticktext=y_ticks_labels,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
            showgrid=True,
            gridwidth=1,
            range=[df['y_transformed'].min() - 0.3, df['y_transformed'].max() + 0.3],
            showline=True,
            linewidth=1,
            linecolor='gray',
            mirror=False,
            zerolinewidth=0
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            title=dict(text=f"<b>{category_col}</b>", font=dict(size=12)),
            bordercolor="rgba(255,255,255,0)",
            borderwidth=0,
            bgcolor="rgba(255, 255, 255, 0.8)",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=11)
        ),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial, sans-serif",
            bordercolor="#2c3e50"
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        autosize=True
    )
    
    x_origin_text = f"{x_origin*100:.1f}%" if x_col_is_percent else format_number(x_origin)
    y_origin_text = f"{y_origin*100:.1f}%" if y_col_is_percent else format_number(y_origin)
    
    fig.add_annotation(
        x=0,
        y=df['y_transformed'].min() - 0.2,
        xref="x",
        yref="y",
        text=x_origin_text,
        showarrow=False,
        font=dict(size=13, color="gray"),
        xanchor="center",
        yanchor="top"
    )

    fig.add_annotation(
        x=df['x_transformed'].min() - 0.2,
        y=0,
        xref="x",
        yref="y",
        text=y_origin_text,
        showarrow=False,
        font=dict(size=13, color="gray"),
        xanchor="right",
        yanchor="middle"
    )

    return fig

st.title("📊 四象限矩阵图可视化工具")

with st.sidebar:
    st.header("📁 数据输入")
    input_method = st.radio("选择输入方式", ["粘贴数据", "上传文件"])
    
    if input_method == "粘贴数据":
        pasted_data = st.text_area(
            "粘贴数据（支持CSV格式或制表符分隔）",
            height=150,
            placeholder="模式,广告引入金额_同比,广告收入_同比,广告引入金额,广告收入\nL1,0.15,0.12,100000,50000\nL2,0.08,-0.03,80000,35000\n..."
        )
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("上传CSV或Excel文件", type=['csv', 'xlsx', 'xls'])
        pasted_data = None
    
    st.header("⚙️ 基本设置")
    title = st.text_input("图表标题", value="波士顿矩阵图")
    height = st.number_input("图表高度", min_value=400, max_value=2000, value=800, step=50)
    width = st.number_input("图表宽度", min_value=400, max_value=2000, value=1000, step=50)
    
    st.header("📍 坐标原点")
    use_custom_origin = st.checkbox("自定义原点")
    col1, col2 = st.columns(2)
    with col1:
        x_origin_input = st.number_input("X轴原点（小数形式）", value=0.0, format="%.3f")
    with col2:
        y_origin_input = st.number_input("Y轴原点（小数形式）", value=0.0, format="%.3f")

    st.header("🔵 气泡大小调节")
    col3, col4 = st.columns(2)
    with col3:
        bubble_outer_multiplier = st.slider("外层气泡乘数", min_value=50, max_value=300, value=150, step=10)
        bubble_outer_base = st.slider("外层气泡基数", min_value=5, max_value=50, value=20, step=5)
    with col4:
        bubble_inner_multiplier = st.slider("内层气泡乘数", min_value=50, max_value=300, value=130, step=10)
        bubble_inner_base = st.slider("内层气泡基数", min_value=5, max_value=50, value=15, step=5)
    
    st.header("📝 文字设置")
    text_size = st.slider("文本大小", min_value=8, max_value=24, value=12, step=1)
    
    st.header("📐 坐标压缩设置")
    compress_factor = st.slider(
        "压缩倍数",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="越远离中心的坐标压缩越多，数值越小压缩越厉害"
    )

df = None
if pasted_data and pasted_data.strip():
    try:
        df = pd.read_csv(io.StringIO(pasted_data), sep=None, engine='python')
        st.success(f"成功读取 {len(df)} 行数据")
    except Exception as e:
        st.error(f"解析粘贴数据出错: {str(e)}")
elif uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"处理文件时出错: {str(e)}")

if df is not None and len(df) > 0:
    st.subheader("📋 数据预览")
    st.dataframe(df.head(10))
    
    columns = df.columns.tolist()
    
    st.subheader("🔧 列映射设置")
    c1, c2 = st.columns(2)
    with c1:
        category_col = st.selectbox("第1列：分类维度", columns, index=0)
        x_col = st.selectbox("第2列：横坐标", columns, index=1)
    with c2:
        y_col = st.selectbox("第3列：纵坐标", columns, index=2)
    
    if len(columns) >= 4:
        outer_size_col = st.selectbox("第4列：外层气泡大小", columns, index=3)
    else:
        st.error("至少需要4列数据")
        st.stop()
    
    if len(columns) >= 5:
        inner_size_col = st.selectbox("第5列：内层气泡大小（可选）", columns, index=4)
        has_inner_size = True
    else:
        inner_size_col = None
        has_inner_size = False
        st.info("未检测到第5列，内层气泡将使用外层气泡数据")
    
    categories = df[category_col].unique().tolist()
    
    st.subheader("📍 文本位置设置")
    st.markdown("点击展开为每个类别设置文本位置")
    
    position_options = [
        ('middle center', '居中'),
        ('middle left', '左中'),
        ('middle right', '右中'),
        ('top center', '上中'),
        ('top left', '左上'),
        ('top right', '右上'),
        ('bottom center', '下中'),
        ('bottom left', '左下'),
        ('bottom right', '右下')
    ]
    position_labels = [f"{p[1]} - {p[0]}" for p in position_options]
    position_values = [p[0] for p in position_options]
    
    with st.expander("设置文本位置", expanded=False):
        text_positions = {}
        cols = st.columns(3)
        for i, cat in enumerate(categories):
            with cols[i % 3]:
                idx = st.selectbox(
                    f"{cat}",
                    range(len(position_options)),
                    index=0,
                    key=f"pos_{cat}",
                    format_func=lambda x: position_labels[x]
                )
                text_positions[cat] = position_values[idx]
    
    xy_origin = (x_origin_input, y_origin_input) if use_custom_origin else None
    
    fig = create_boston_matrix(
        df=df,
        title=title,
        xy_origin=xy_origin,
        height=height,
        width=width,
        bubble_outer_multiplier=bubble_outer_multiplier,
        bubble_outer_base=bubble_outer_base,
        bubble_inner_multiplier=bubble_inner_multiplier,
        bubble_inner_base=bubble_inner_base,
        text_size=text_size,
        category_col=category_col,
        x_col=x_col,
        y_col=y_col,
        outer_size_col=outer_size_col,
        inner_size_col=inner_size_col,
        has_inner_size=has_inner_size,
        text_positions=text_positions,
        compress_factor=compress_factor
    )
        
    st.subheader("📊 可视化结果")
    st.plotly_chart(fig, use_container_width=True)
    
    html_buffer = io.StringIO()
    fig.write_html(html_buffer, include_plotlyjs='cdn')
    html_buffer.seek(0)
    
    st.download_button(
        label="📥 导出为HTML",
        data=html_buffer.getvalue(),
        file_name=f"{title}.html",
        mime="text/html"
    )
else:
    st.info("👈 请在左侧上传数据文件或粘贴数据开始使用")
    
    with st.expander("📝 查看示例数据格式", expanded=False):
        st.markdown("**5列数据格式：**")
        sample_data_5 = {
            '模式': ['L1', 'L2', 'L3', 'L4', 'L5'],
            '广告引入金额_同比': [0.15, 0.08, -0.05, 0.25, -0.10],
            '广告收入_同比': [0.12, -0.03, 0.10, 0.20, -0.08],
            '广告引入金额': [100000, 80000, 60000, 120000, 50000],
            '广告收入': [50000, 35000, 40000, 70000, 25000]
        }
        st.dataframe(pd.DataFrame(sample_data_5))
        
        st.markdown("**4列数据格式（无内层气泡）：**")
        sample_data_4 = {
            '模式': ['L1', 'L2', 'L3', 'L4', 'L5'],
            '广告引入金额_同比': [0.15, 0.08, -0.05, 0.25, -0.10],
            '广告收入_同比': [0.12, -0.03, 0.10, 0.20, -0.08],
            '广告引入金额': [100000, 80000, 60000, 120000, 50000]
        }
        st.dataframe(pd.DataFrame(sample_data_4))
        st.markdown("""
        **数据要求：**
        - 第1列：分类维度（如：产品线、品类名称）
        - 第2列：横坐标值（小数形式，如 0.15 表示 15%）
        - 第3列：纵坐标值（小数形式，如 0.12 表示 12%）
        - 第4列：外层气泡大小指标（绝对值）
        - 第5列（可选）：内层气泡大小指标（绝对值）
        """)
