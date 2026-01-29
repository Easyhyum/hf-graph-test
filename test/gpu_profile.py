# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import multiprocessing
import os
import signal
import sys, time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TextIO

import psutil

import torch

# logger = init_logger(__name__)
CYAN = "\033[0;36m"
RESET = "\033[0;0m"
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

csv_path = f"measured_metrics"

class GPUMonitor:
    def __init__(self, device_id=0, batch_size=None, graph_batch_size=None, decoding_steps=None, cudagraph_mode=None, manipulated=False):
        self.device_id = device_id
        self.handle = None
        self.metrics = []
        self.execution_time = None  # 실행 시간 (초)
        self.num_samples = None  # 처리된 샘플 수
        self.batch_size = batch_size
        self.graph_batch_size = graph_batch_size
        self.decoding_steps = decoding_steps
        self.cudagraph_mode = cudagraph_mode
        self.manipulated = manipulated
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            except Exception as e:
                print(f"NVML 초기화 실패: {e}")
                self.handle = None
    def get_metrics(self):
        """현재 GPU 메트릭 수집"""
        if not NVML_AVAILABLE or self.handle is None:
            return None
        
        try:
        # 1. 전력 제한(Power Cap)에 걸린 시간 가져오기
            power_violation = pynvml.nvmlDeviceGetViolationStatus(self.handle, pynvml.NVML_PERF_POLICY_POWER)
            
            # 반환값은 객체이며 .violationTime과 .referenceTime 속성을 가집니다.
            # 단위는 나노초(nanoseconds)입니다.
            p_violation_time_ns = power_violation.violationTime
            p_reference_time_ns = power_violation.referenceTime
            
            # 보기 좋게 밀리초(ms)나 초(s)로 변환
            
            # 2. 온도 제한(Thermal)에 걸린 시간 가져오기
            thermal_violation = pynvml.nvmlDeviceGetViolationStatus(self.handle, pynvml.NVML_PERF_POLICY_THERMAL)
            
            t_violation_time_ns = thermal_violation.violationTime 
    
            # 전력 (mW)
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # W로 변환
            
            # 온도 (°C)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # 그래픽 클럭 (MHz) - 일반적으로 SM 클럭과 동일
            graphics_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_GRAPHICS)
            
            # SM 클럭 (MHz) - Streaming Multiprocessor 클럭
            # 일부 GPU에서는 지원되지 않을 수 있으므로 try-except 사용
            sm_clock = None
            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            except (pynvml.NVMLError, AttributeError):
                # NVML_CLOCK_SM이 지원되지 않으면 graphics_clock 사용
                sm_clock = graphics_clock
            
            # 메모리 클럭 (MHz)
            memory_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            
            # 메모리 사용량 (MB)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used = mem_info.used / (1024 ** 2)  # MB
            memory_total = mem_info.total / (1024 ** 2)  # MB
            
            # GPU 활용률 (utilization)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = util.gpu
            memory_util = util.memory
            
            return {
                'power': power,
                'temperature': temp,
                'graphics_clock': graphics_clock,
                'sm_clock': sm_clock,
                'memory_clock': memory_clock,
                'memory_used': memory_used,
                'memory_total': memory_total,
                'gpu_util': gpu_util,
                'memory_util': memory_util,
                'power_violation_time_ns': p_violation_time_ns,
                'power_reference_time_ns': p_reference_time_ns,
                'thermal_violation_time_ns': t_violation_time_ns
            }
        except Exception as e:
            print(f"메트릭 수집 오류: {e}")
            return None
    def start_monitoring(self, interval=0.1, duration=None):
        """모니터링 시작"""
        self.metrics = []
        if not NVML_AVAILABLE or self.handle is None:
            return
        
        start_time = time.time()
        while True:
            metric = self.get_metrics()
            if metric:
                metric['timestamp'] = time.time()
                self.metrics.append(metric)
            
            if duration and (time.time() - start_time) >= duration:
                break
            
            time.sleep(interval)
    
    def collect_during_execution(self, func, *args, num_samples=None, **kwargs):
        """함수 실행 중 메트릭 수집 (multiprocessing 사용)"""
        self.metrics = []
        self.num_samples = num_samples
        
        if not NVML_AVAILABLE or self.handle is None:
            start_time = time.time()
            result = func(*args, **kwargs)
            self.execution_time = time.time() - start_time
            return result
        
        from multiprocessing import Process, Queue, Event
        
        metrics_queue = Queue()
        stop_event = Event()
        
        def monitor_process(device_id, metrics_queue, stop_event):
            """별도 프로세스에서 GPU 메트릭 수집"""
            try:
                # 프로세스 내에서 NVML 재초기화
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                start_time = time.time()
                metrics_queue.put(('start_time', start_time))
                
                while not stop_event.is_set():
                    try:
                        # 전력 측정
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                        
                        sm_clock = None
                        try:
                            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                        except (pynvml.NVMLError, AttributeError):
                            sm_clock = graphics_clock
                        
                        memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        
                        power_violation = pynvml.nvmlDeviceGetViolationStatus(handle, pynvml.NVML_PERF_POLICY_POWER)
                        thermal_violation = pynvml.nvmlDeviceGetViolationStatus(handle, pynvml.NVML_PERF_POLICY_THERMAL)
                        
                        perf_state= pynvml.nvmlDeviceGetPerformanceState(handle)
                        power_state = pynvml.nvmlDeviceGetPowerState(handle)
                        total_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                        metric = {
                            'timestamp': time.time(),
                            'power': power,
                            'temperature': temp,
                            'graphics_clock': graphics_clock,
                            'sm_clock': sm_clock,
                            'memory_clock': memory_clock,
                            'memory_used': mem_info.used / (1024 ** 2),
                            'memory_total': mem_info.total / (1024 ** 2),
                            'gpu_util': util.gpu,
                            'memory_util': util.memory,
                            'power_violation_time_ns': power_violation.violationTime,
                            'power_reference_time_ns': power_violation.referenceTime,
                            'thermal_violation_time_ns': thermal_violation.violationTime,
                            'power_state': power_state,
                            'perf_state': perf_state,
                            'total_energy': total_energy
                        }
                        
                        metrics_queue.put(('metric', metric))
                        
                    except Exception as e:
                        metrics_queue.put(('error', str(e)))
                    
                    time.sleep(0.1)  # 10ms 간격
                
                pynvml.nvmlShutdown()
                
            except Exception as e:
                metrics_queue.put(('error', f"Monitor process error: {e}"))
        
        # 모니터링 프로세스 시작
        monitor_proc = Process(target=monitor_process, args=(self.device_id, metrics_queue, stop_event))
        monitor_proc.start()
        
        # start_time 받기
        start_time = None
        try:
            msg_type, data = metrics_queue.get(timeout=2.0)
            if msg_type == 'start_time':
                start_time = data
        except:
            pass
        
        try:
            # 메인 함수 실행
            result = func(*args, **kwargs)
        finally:
            # 모니터링 중지
            stop_event.set()
            monitor_proc.join(timeout=5.0)
            if monitor_proc.is_alive():
                monitor_proc.terminate()
                monitor_proc.join(timeout=1.0)
            
            # 수집된 메트릭 가져오기
            while not metrics_queue.empty():
                try:
                    msg_type, data = metrics_queue.get_nowait()
                    if msg_type == 'metric':
                        self.metrics.append(data)
                    elif msg_type == 'error':
                        print(f"Monitoring error: {data}")
                except:
                    break
            
            if start_time:
                self.execution_time = time.time() - start_time
        
        return result
    
    def get_statistics(self):
        """수집된 메트릭의 통계 계산"""
        if not self.metrics:
            return None
        
        stats = {}
        for key in ['power', 'temperature', 'graphics_clock', 'sm_clock', 'memory_clock', 
                   'gpu_util', 'memory_util', 'power_violation_time_ns', 'power_reference_time_ns', 'thermal_violation_time_ns']:
            values = [m[key] for m in self.metrics if key in m and m[key] is not None]
            if values:
                stats[key] = {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'count' : len(values),
                    'current': values[-1] if values else None
                }
        
        return stats
    
    def calculate_energy_per_sample(self):
        """샘플당 에너지 (J/sample) 계산"""
        if not self.metrics or not self.execution_time or not self.num_samples:
            return None
        
        # 평균 전력 계산
        power_values = [m['power'] for m in self.metrics if 'power' in m]
        if not power_values:
            return None
        
        avg_power = sum(power_values) / len(power_values)  # W
        
        # 총 에너지 = 평균 전력 × 시간 (Joule)
        total_energy = avg_power * self.execution_time  # J
        
        # 샘플당 에너지 (주의: num_samples는 이미 총 처리 샘플 수)
        energy_per_sample = total_energy / self.num_samples  # J/sample
        
        # 처리량 계산
        throughput = self.num_samples / self.execution_time  # samples/s
        
        return {
            'total_energy': total_energy,
            'energy_per_sample': energy_per_sample,
            'avg_power': avg_power,
            'execution_time': self.execution_time,
            'num_samples': self.num_samples,
            'throughput': throughput
        }
            
    def update_environment_variables(envs_dict: dict[str, str]):
        """Update multiple environment variables with logging."""
        for k, v in envs_dict.items():
            if k in os.environ and os.environ[k] != v:
                logger.warning(
                    "Overwriting environment variable %s from '%s' to '%s'",
                    k,
                    os.environ[k],
                    v,
                )
            os.environ[k] = v
    def save_statistics_to_csv(self, path=csv_path, during_time=None, decoding_steps=None):
        """수집된 메트릭을 CSV 파일로 저장"""
        if not self.metrics:
            print("저장할 메트릭 데이터가 없습니다.")
            return
        
        import csv
        
        # 통계 계산
        energy_info = self.calculate_energy_per_sample()
        avg_power_js = energy_info['avg_power'] if energy_info else None
        total_energy = energy_info['total_energy'] if energy_info else None
        energy_per_sample = energy_info['energy_per_sample'] if energy_info else None
        throughput = energy_info['throughput'] if energy_info else None
        
        keys = ['cudagraph_mode', 'manipulated', 'batch_size', 'graph_batch_size', 'during_time', 'index', 'length', 
                'avg_power_js', 'total_energy_j', 'energy_per_sample', 'throughput',
                'timestamp', 'power', 'temperature', 'graphics_clock', 'sm_clock', 
                'memory_clock', 'memory_used', 'memory_total', 'gpu_util', 'memory_util', 
                'power_violation_time_ns', 'power_reference_time_ns', 'thermal_violation_time_ns', 'decoding_steps', 'perf_state', 'power_state', 'total_energy']
        
        os.makedirs(path, exist_ok=True)

        file_path = os.path.join(path, f"gpu_profile_{os.getpid()}.csv")
        length = len(self.metrics)
        
        try:
            # file_path 파일이 존재하는지 확인
            if not os.path.exists(file_path):
                with open(file_path, mode='w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=keys)
                    writer.writeheader()
            with open(file_path, mode='a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                for i, metric in enumerate(self.metrics):
                    # build row for this metric
                    metric_row = {k: metric.get(k, None) for k in keys}
                    # update with contextual fields and statistics
                    metric_row.update({
                        "cudagraph_mode": self.cudagraph_mode, 
                        "manipulated": self.manipulated,
                        "batch_size": self.batch_size, 
                        "graph_batch_size": self.graph_batch_size, 
                        "during_time": during_time, 
                        "length": length, 
                        "index": i + 1,
                        "avg_power_js": avg_power_js,
                        "total_energy_j": total_energy,
                        "energy_per_sample": energy_per_sample,
                        "throughput": throughput,
                        "decoding_steps": decoding_steps,
                        "perf_state": metric.get("perf_state", None),
                        "power_state": metric.get("power_state", None),
                        "total_energy": metric.get("total_energy", None)
                    })
                    writer.writerow(metric_row)
        except Exception as e:
            print(f"메트릭 CSV 저장 중 오류: {e}")
    def print_statistics(self, label=""):
        """통계 출력"""
        stats = self.get_statistics()
        if not stats:
            print(f"{label}: 메트릭 데이터가 없습니다.")
            return

        print(f"{label} GPU 메트릭: {self.batch_size} 배치 크기, {self.graph_batch_size} 그래프 배치 크기")
        print(f"  전력:")
        print(f"    평균: {stats['power']['avg']:.2f} W")
        print(f"    최소: {stats['power']['min']:.2f} W")
        print(f"    최대: {stats['power']['max']:.2f} W")
        print(f"    개수: {stats['power']['count']}")
        print(f"    현재: {stats['power']['current']:.2f} W")
        
        energy_info = self.calculate_energy_per_sample()
        print(f"  에너지 소비:")
        print(f"    총 에너지: {energy_info['total_energy']:.4f} J")
        print(f"    샘플당 에너지: {energy_info['energy_per_sample']:.6f} J/sample")
        print(f"    실행 시간: {energy_info['execution_time']:.4f} s")
        print(f"    처리 샘플 수: {energy_info['num_samples']:,}")
        print(f"    처리량(Throughput): {energy_info['throughput']:.2f} samples/s")
        
        print(f"  온도:")
        print(f"    평균: {stats['temperature']['avg']:.1f} °C")
        print(f"    최소: {stats['temperature']['min']:.1f} °C")
        print(f"    최대: {stats['temperature']['max']:.1f} °C")
        print(f"    현재: {stats['temperature']['current']:.1f} °C")
        
        print(f"  SM 클럭 (Streaming Multiprocessor):")
        print(f"    평균: {stats['sm_clock']['avg']:.0f} MHz")
        print(f"    최소: {stats['sm_clock']['min']:.0f} MHz")
        print(f"    최대: {stats['sm_clock']['max']:.0f} MHz")
        print(f"    현재: {stats['sm_clock']['current']:.0f} MHz")
        
        print(f"  메모리 클럭:")
        print(f"    평균: {stats['memory_clock']['avg']:.0f} MHz")
        print(f"    최소: {stats['memory_clock']['min']:.0f} MHz")
        print(f"    최대: {stats['memory_clock']['max']:.0f} MHz")
        print(f"    현재: {stats['memory_clock']['current']:.0f} MHz")
        
        print(f"  GPU 활용률:")
        print(f"    평균: {stats['gpu_util']['avg']:.1f} %")
        print(f"    최소: {stats['gpu_util']['min']:.1f} %")
        print(f"    최대: {stats['gpu_util']['max']:.1f} %")
        print(f"    현재: {stats['gpu_util']['current']:.1f} %")
        
        print(f"  메모리 활용률:")
        print(f"    평균: {stats['memory_util']['avg']:.1f} %")
        print(f"    최소: {stats['memory_util']['min']:.1f} %")
        print(f"    최대: {stats['memory_util']['max']:.1f} %")
        print(f"    현재: {stats['memory_util']['current']:.1f} %")
        
        if self.metrics:
            mem_info = self.metrics[-1]
            if 'memory_used' in mem_info and 'memory_total' in mem_info:
                print(f"  메모리 사용량:")
                print(f"    사용: {mem_info['memory_used']:.0f} MB / {mem_info['memory_total']:.0f} MB")
                print(f"    사용률: {mem_info['memory_used'] / mem_info['memory_total'] * 100:.1f} %")


# Alias for backwards compatibility
NVMLContext = GPUMonitor
