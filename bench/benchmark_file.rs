use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use fast_sssp::{FastSSSP, ClassicDijkstra, examples::create_random_graph};
use std::hint::black_box;

fn benchmark_sssp(c: &mut Criterion) {
    let mut group = c.benchmark_group("SSSP Algorithms");
    
    // Test different graph sizes
    let sizes = vec![
        (100, 300),    // Small graph
        (500, 1500),   // Medium graph
        (1000, 3000),  // Large graph
        (2000, 8000),  // Very large graph
    ];
    
    for (n, m) in sizes {
        let graph = create_random_graph(n, m, 100.0);
        
        // Benchmark Fast SSSP
        group.bench_with_input(
            BenchmarkId::new("FastSSSP", format!("n={}_m={}", n, m)),
            &graph,
            |b, graph| {
                b.iter(|| {
                    let mut fast_sssp = FastSSSP::new(graph.clone());
                    fast_sssp.solve(0);
                    let distances = fast_sssp.get_distances();
                    black_box(distances[0]) // Just access first element to avoid borrowing issues
                })
            },
        );
        
        // Benchmark Classic Dijkstra
        group.bench_with_input(
            BenchmarkId::new("Dijkstra", format!("n={}_m={}", n, m)),
            &graph,
            |b, graph| {
                b.iter(|| {
                    let dijkstra = ClassicDijkstra::new(graph.clone());
                    black_box(dijkstra.solve(0))
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_sparse_vs_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sparse vs Dense Graphs");
    let n = 1000;
    
    // Sparse graph (m ≈ n)
    let sparse_graph = create_random_graph(n, n, 100.0);
    
    // Dense graph (m ≈ n²/4)
    let dense_graph = create_random_graph(n, n * n / 4, 100.0);
    
    group.bench_function("FastSSSP_Sparse", |b| {
        b.iter(|| {
            let mut fast_sssp = FastSSSP::new(sparse_graph.clone());
            fast_sssp.solve(0);
            let distances = fast_sssp.get_distances();
            black_box(distances[0])
        })
    });
    
    group.bench_function("Dijkstra_Sparse", |b| {
        b.iter(|| {
            let dijkstra = ClassicDijkstra::new(sparse_graph.clone());
            let distances = dijkstra.solve(0);
            black_box(distances[0])
        })
    });
    
    group.bench_function("FastSSSP_Dense", |b| {
        b.iter(|| {
            let mut fast_sssp = FastSSSP::new(dense_graph.clone());
            fast_sssp.solve(0);
            let distances = fast_sssp.get_distances();
            black_box(distances[0])
        })
    });
    
    group.bench_function("Dijkstra_Dense", |b| {
        b.iter(|| {
            let dijkstra = ClassicDijkstra::new(dense_graph.clone());
            let distances = dijkstra.solve(0);
            black_box(distances[0])
        })
    });
    
    group.finish();
}

fn benchmark_memory_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Mapping vs In-Memory");
    
    // Test larger graphs where memory mapping benefits are more apparent
    let sizes = vec![
        (1000, 3000),   // Medium graph
        (2000, 8000),   // Large graph
        (5000, 15000),  // Very large graph
    ];
    
    for (n, m) in sizes {
        let graph = create_random_graph(n, m, 100.0);
        
        // Benchmark in-memory version
        group.bench_with_input(
            BenchmarkId::new("InMemory", format!("n={}_m={}", n, m)),
            &graph,
            |b, graph| {
                b.iter(|| {
                    let mut fast_sssp = FastSSSP::new(graph.clone());
                    fast_sssp.solve(0);
                    let distances = fast_sssp.get_distances();
                    black_box(distances[0])
                })
            },
        );
        
        // Benchmark memory-mapped version with SIMD
        group.bench_with_input(
            BenchmarkId::new("MemoryMapped_SIMD", format!("n={}_m={}", n, m)),
            &graph,
            |b, graph| {
                b.iter(|| {
                    let mut fast_sssp_mmap = FastSSSP::new_with_mmap(graph.clone()).unwrap();
                    fast_sssp_mmap.solve(0);
                    let distances = fast_sssp_mmap.get_distances();
                    black_box(distances[0])
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_sssp, benchmark_sparse_vs_dense, benchmark_memory_mapping);
criterion_main!(benches);