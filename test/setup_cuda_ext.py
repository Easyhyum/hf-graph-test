"""
CUDA Graph API 확장 모듈 빌드 스크립트

기능:
1. DOT 파일 저장 (cudaGraphDebugDotPrint)
2. 그래프 노드 수정 (Grid dimension 변경 등)

사용법:
    python setup_cuda_ext.py install
    또는
    python setup_cuda_ext.py build_ext --inplace
    
빌드 후 사용:
    import cuda_graph_api
    
    # DOT 파일 저장 테스트
    cuda_graph_api.test_empty_graph_dot("test.dot")
    
    # 모델 캡처 및 DOT 저장
    graph, exec, output = cuda_graph_api.capture_graph_and_save_dot(
        model.forward, input_tensor, "model_graph.dot"
    )
    
    # 그래프 노드 정보 조회
    nodes = cuda_graph_api.get_graph_nodes_info(graph)
    
    # 그래프 batch size 수정
    cuda_graph_api.modify_graph_for_batch(graph, 16, 9)
    new_exec = cuda_graph_api.reinstantiate_graph(graph)
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# CUDA 경로 설정
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
cuda_include = os.path.join(cuda_home, 'include')

# 현재 디렉토리
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='cuda_graph_api',
    version='1.0.0',
    description='CUDA Graph manipulation and DOT file saving utilities',
    ext_modules=[
        CUDAExtension(
            name='cuda_graph_api',
            sources=[
                os.path.join(current_dir, 'graph_api.cu'),
            ],
            include_dirs=[cuda_include],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '--expt-relaxed-constexpr',
                    '-std=c++17',
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
