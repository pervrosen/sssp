use fast_sssp::examples;

fn main() {
    println!("Fast Single-Source Shortest Path Algorithm");
    println!("==========================================");
    
    // Run examples
    examples::social_network_example();
    examples::city_network_example();
    examples::network_latency_example();
    
    // Run benchmarks
    println!("\n=== Performance Benchmarks ===");
    examples::benchmark_algorithms(100, 300);
    examples::benchmark_algorithms(500, 1500);
    examples::benchmark_algorithms(1000, 5000);
}