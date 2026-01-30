#!/usr/bin/env python3
"""
배치 크기별 전력 소비 및 샘플당 에너지 시각화 (개별 파일 처리)

각 CSV 파일마다 graph_batch_size와 batch_size 기준으로 두 개의 그래프를 생성합니다.

사용법:
    python plot_power_energy_individual.py logs/gpu_profile_164454.csv
    python plot_power_energy_individual.py logs/gpu_profile_*.csv  # 여러 파일 개별 분석
"""

import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def load_and_aggregate_data(csv_file, x_axis='batch_size'):
    """CSV 파일을 로드하고 배치별로 집계
    
    Args:
        csv_file: CSV 파일 경로
        x_axis: 'batch_size' 또는 'graph_batch_size'
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise ValueError(f"Failed to load {csv_file}: {e}")
    
    # cudagraph_mode 가 "FULL"인 행만 필터링
    df = df[df['cudagraph_mode'] == 'FULL']
    # df = df[df['batch_size'].isin([1,2,4,5,8,9,16,32])]
    # 샘플 개수 먼저 계산: index==length인 행의 개수 * batch_size
    sample_counts = df[df['index'] == df['length']].groupby(x_axis).agg({
        'batch_size': lambda x: len(x) * x.iloc[0]  # 행 개수 * batch_size
    }).rename(columns={'batch_size': 'total_samples'})
    
    # index/length 비율이 0.5 이하인 항목 제거 (샘플 개수 계산 후에 적용)
    df = df[df['index'] / df['length'] > 0.5]
    
    # 배치별로 집계
    agg_dict = {
        'power': ['mean', 'std', 'min', 'max'],
        'during_time': 'mean',
        'throughput': 'first',
        'gpu_util': 'mean',
        'temperature': 'mean'
    }
    
    # x_axis가 아닌 컬럼만 추가
    other_axis = 'graph_batch_size' if x_axis == 'batch_size' else 'batch_size'
    agg_dict[other_axis] = 'first'
    
    grouped = df.groupby(x_axis).agg(agg_dict).reset_index()
    
    # 컬럼명 정리 (멀티인덱스 컬럼 평탄화)
    new_columns = [x_axis]
    for col in grouped.columns[1:]:
        if isinstance(col, tuple):
            if col[1]:  # 집계 함수가 있는 경우
                if col[0] == 'power':
                    new_columns.append(f'power_{col[1]}')
                else:
                    new_columns.append(col[0])
            else:
                new_columns.append(col[0])
        else:
            new_columns.append(col)
    
    grouped.columns = new_columns
    
    # x_axis가 아닌 컬럼이 없으면 x_axis 값으로 채우기
    if other_axis not in grouped.columns:
        grouped[other_axis] = grouped[x_axis]
    
    # 샘플 개수 병합
    grouped = grouped.merge(sample_counts, on=x_axis, how='inner')
    
    # power와 during_time을 이용한 에너지 계산
    # during_time은 600번 반복한 시간이므로 600으로 나눔
    # Energy (J) = Power (W) × Time (s)
    grouped['avg_power'] = grouped['power_mean']
    grouped['time_per_iteration'] = grouped['during_time'] / 600.0
    grouped['total_energy'] = grouped['power_mean'] * grouped['time_per_iteration']
    # Energy per Sample = 1회 iteration의 총 에너지 / batch_size (항상 batch_size 사용!)
    if 'batch_size' in grouped.columns:
        grouped['energy_per_sample'] = grouped['total_energy'] / grouped['batch_size']
    else:
        # batch_size 컬럼이 없으면 x_axis 값 사용 (batch_size 기준일 때)
        grouped['energy_per_sample'] = grouped['total_energy'] / grouped[x_axis]
    
    # NaN이 있는 행 제거
    grouped = grouped.dropna(subset=['total_samples', 'energy_per_sample'])
    
    return grouped.sort_values(x_axis)

def plot_power_and_energy(data, output_file, x_axis='batch_size', csv_filename=''):
    """전력과 에너지를 이중 축으로 시각화
    
    Args:
        data: 집계된 데이터프레임
        output_file: 출력 파일명
        x_axis: X축으로 사용할 컬럼 ('batch_size' 또는 'graph_batch_size')
        csv_filename: 원본 CSV 파일명 (제목에 표시)
    """
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    batch_sizes = data[x_axis].values
    x_pos = np.arange(len(batch_sizes))
    
    # X축 라벨 결정
    x_label = 'Batch Size' if x_axis == 'batch_size' else 'Graph Batch Size'
    
    # 왼쪽 축: 평균 전력 (histogram/bar)
    color1 = '#2E86AB'
    bars = ax1.bar(x_pos, data['power_mean'], 
                   yerr=data['power_std'],
                   alpha=0.7, 
                   color=color1,
                   edgecolor='black',
                   linewidth=1.5,
                   capsize=5,
                   label='Average Power (W)')
    
    ax1.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Power (W)', fontsize=14, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{int(bs)}' for bs in batch_sizes])
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, data['power_mean'].max() * 1.3)

    # 막대 위에 정확한 값 표시
    for i, (bar, val, std) in enumerate(zip(bars, data['power_mean'], data['power_std'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}W\n±{std:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 오른쪽 축 1: 샘플당 에너지 (line graph)
    ax2 = ax1.twinx()
    color2 = '#A23B72'
    j_p_sample = data['energy_per_sample']  # power × time 기반 계산값 사용
    line2 = ax2.plot(x_pos, j_p_sample, 
                    color=color2, 
                    marker='o', 
                    markersize=10,
                    linewidth=3,
                    label='Energy per Sample (J/sample)')
    
    ax2.set_ylabel('Energy per Sample (J/sample)', fontsize=14, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)
    
    # 라인 위에 정확한 값 표시 (마커보다 위에)
    for i, (x, y) in enumerate(zip(x_pos, j_p_sample)):
        ax2.annotate(f'{y:.4f}', xy=(x, y),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color=color2)
    
    # 오른쪽 축 2: During Time (line graph)
    ax3 = ax1.twinx()
    # ax3의 spine을 오른쪽으로 이동
    ax3.spines['right'].set_position(('outward', 70))
    color3 = '#F18F01'
    during_time_vals = data['during_time']
    line3 = ax3.plot(x_pos, during_time_vals,
                    color=color3,
                    marker='s',
                    markersize=8,
                    linewidth=2.5,
                    linestyle='--',
                    label='During Time (s)')
    
    ax3.set_ylabel('During Time (s)', fontsize=14, fontweight='bold', color=color3)
    ax3.tick_params(axis='y', labelcolor=color3, labelsize=12)
    
    # 라인 아래에 정확한 값 표시 (마커보다 아래에)
    for i, (x, y) in enumerate(zip(x_pos, during_time_vals)):
        ax3.annotate(f'{y:.3f}', xy=(x, y),
                    xytext=(0, -8), textcoords='offset points',
                    ha='center', va='top', fontsize=8, fontweight='bold',
                    color=color3)
    
    # 제목 및 범례
    title = f'Power Consumption and Energy Efficiency by {x_label}\n'
    if csv_filename:
        title += f'({csv_filename} - Power × Time/600)'
    else:
        title += '(Power × Time/600 calculation)'
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 범례 통합
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
              loc='lower left', fontsize=11, framealpha=0.9)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 파일 저장
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    
    # 화면 표시 하지 않음 (여러 파일 처리 시 창이 많이 뜸)
    plt.close()

def print_summary_table(data, x_axis='batch_size', csv_filename=''):
    """요약 테이블 출력
    
    Args:
        data: 집계된 데이터프레임
        x_axis: X축으로 사용할 컬럼
        csv_filename: 원본 CSV 파일명
    """
    x_label = 'Batch' if x_axis == 'batch_size' else 'Graph'
    
    print("\n" + "="*140)
    if csv_filename:
        print(f"{csv_filename} - {x_label} 크기별 성능 요약 (Power × Time 기반)")
    else:
        print(f"{x_label} 크기별 성능 요약 (Power × Time 기반)")
    print("="*140)
    print(f"{x_label:>6} | {'Batch':>6} | {'Graph':>6} | {'Samples':>8} | {'Avg Power':>10} | "
          f"{'Power StdDev':>12} | {'Time/Iter':>10} | {'Total Energy':>13} | {'Energy/Sample':>15} | "
          f"{'Throughput':>12} | {'GPU Util':>9} | {'Temp':>7}")
    print("-"*140)
    
    for _, row in data.iterrows():
        if x_axis == 'batch_size':
            print(f"{int(row[x_axis]):6d} | "
                f"{int(row['batch_size']):6d} | "
                f"{'N/A':>6s} | "
                f"{int(row['total_samples']):8d} | "
                f"{row['power_mean']:10.2f}W | "
                f"{row['power_std']:12.2f}W | "
                f"{row['time_per_iteration']:10.4f}s | "
                f"{row['total_energy']:13.4f}J | "
                f"{row['energy_per_sample']:15.6f}J | "
                f"{row['throughput']:12.2f}/s | "
                f"{row['gpu_util']:9.1f}% | "
                f"{row['temperature']:7.1f}°C")
        else:
            print(f"{int(row[x_axis]):6d} | "
                f"{int(row['batch_size']):6d} | "
                f"{int(row['graph_batch_size']):6d} | "
                f"{int(row['total_samples']):8d} | "
                f"{row['power_mean']:10.2f}W | "
                f"{row['power_std']:12.2f}W | "
                f"{row['time_per_iteration']:10.4f}s | "
                f"{row['total_energy']:13.4f}J | "
                f"{row['energy_per_sample']:15.6f}J | "
                f"{row['throughput']:12.2f}/s | "
                f"{row['gpu_util']:9.1f}% | "
                f"{row['temperature']:7.1f}°C")

    print("="*140)
    
    # 효율성 분석
    best_power_efficiency = data.loc[data['energy_per_sample'].idxmin()]
    print(f"\n✓ 최고 에너지 효율: {x_label} {int(best_power_efficiency[x_axis])} "
          f"({best_power_efficiency['energy_per_sample']:.6f} J/sample, "
          f"{int(best_power_efficiency['total_samples'])} samples)")
    
    highest_throughput = data.loc[data['throughput'].idxmax()]
    print(f"✓ 최고 처리량: {x_label} {int(highest_throughput[x_axis])} "
          f"({highest_throughput['throughput']:.2f} samples/s)")
    
    print(f"\n[계산 방식]")
    print(f"  - Time per iteration = during_time / 600")
    print(f"  - Total Energy (J) = Average Power (W) × Time per iteration (s)")
    print(f"  - Energy per Sample = Total Energy / Total Samples")
    print()

def process_single_file(csv_file, show_table=True):
    """단일 CSV 파일 처리 - graph_batch_size와 batch_size 기준 두 개의 그래프 생성
    
    Args:
        csv_file: CSV 파일 경로
        show_table: 요약 테이블 출력 여부
    """
    # 파일명에서 확장자 제거
    file_path = Path(csv_file)
    base_name = 'case0_' + file_path.stem  # 확장자 없는 파일명
    output_dir = file_path.parent  # 원본 파일과 같은 디렉토리
    
    print(f"\n{'='*120}")
    print(f"Processing: {csv_file}")
    print(f"{'='*120}")
    
    # 1. graph_batch_size 기준 그래프
    try:
        print(f"\n[1/2] Generating graph_batch_size plot...")
        data_graph = load_and_aggregate_data(csv_file, x_axis='graph_batch_size')
        output_file_graph = output_dir / f"{base_name}_graph_batch.png"
        
        if show_table:
            print_summary_table(data_graph, x_axis='graph_batch_size', csv_filename=file_path.name)
        
        plot_power_and_energy(data_graph, str(output_file_graph), 
                            x_axis='graph_batch_size', csv_filename=file_path.name)
    except Exception as e:
        print(f"  ✗ Error generating graph_batch_size plot: {e}")
    
    # 2. batch_size 기준 그래프
    try:
        print(f"\n[2/2] Generating batch_size plot...")
        data_batch = load_and_aggregate_data(csv_file, x_axis='batch_size')
        output_file_batch = output_dir / f"{base_name}_batch.png"
        
        if show_table:
            print_summary_table(data_batch, x_axis='batch_size', csv_filename=file_path.name)
        
        plot_power_and_energy(data_batch, str(output_file_batch), 
                            x_axis='batch_size', csv_filename=file_path.name)
    except Exception as e:
        print(f"  ✗ Error generating batch_size plot: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='배치 크기별 전력 소비 및 샘플당 에너지 시각화 (개별 파일 처리)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_power_energy_individual.py logs/gpu_profile_164454.csv
  python plot_power_energy_individual.py logs/gpu_profile_*.csv
  python plot_power_energy_individual.py logs/gpu_profile_*.csv --no-table
  
각 CSV 파일마다 2개의 그래프가 생성됩니다:
  - {파일명}_graph_batch.png (graph_batch_size 기준)
  - {파일명}_batch.png (batch_size 기준)
        """
    )
    
    parser.add_argument('csv_pattern', nargs='+', help='CSV 파일 경로 또는 패턴 (예: logs/gpu_profile_*.csv)')
    parser.add_argument('--no-table', action='store_true',
                       help='요약 테이블 출력 생략')
    
    args = parser.parse_args()
    
    # CSV 파일 찾기 (각 패턴을 glob으로 확장)
    csv_files = []
    for pattern in args.csv_pattern:
        expanded = glob.glob(pattern)
        if expanded:
            csv_files.extend(expanded)
        else:
            # glob 확장이 안되면 직접 파일로 처리 (쉘이 이미 확장한 경우)
            csv_files.append(pattern)
    
    if not csv_files:
        print(f"Error: No CSV files found matching pattern: {args.csv_pattern}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {f}")
    
    # 각 파일 개별 처리
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n\n{'#'*120}")
        print(f"# File {i}/{len(csv_files)}")
        print(f"{'#'*120}")
        process_single_file(csv_file, show_table=not args.no_table)
    
    print(f"\n\n{'='*120}")
    print(f"✓ All files processed! Total: {len(csv_files)} files, {len(csv_files)*2} plots generated")
    print(f"{'='*120}\n")

if __name__ == "__main__":
    main()
