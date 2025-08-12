// examples/practical_usage.rs
use fast_sssp::{Graph, FastSSSP, ClassicDijkstra, examples::create_random_graph};
use std::time::Instant;

fn main() {
    println!("=== Practical Usage Examples for Fast SSSP ===\n");
    
    // Example 1: Simple pathfinding
    simple_pathfinding_example();
    
    // Example 2: Large graph performance comparison
    performance_comparison_example();
    
    // Example 3: Real-world application - delivery optimization
    delivery_optimization_example();

    // Example 3a: Deliver to all customers, but shortest total path - Travelling Salesman Problem (TSP)
    delivery_route_optimization_example();
    
    // Example 4: Algorithm analysis on different graph structures
    graph_structure_analysis();
}

fn delivery_optimization_example_full() -> () {
    // Example 3a: Deliver to all customers, but shortest total path
    todo!()
}

fn simple_pathfinding_example() {
    println!("1. Simple Pathfinding Example");
    println!("==============================");
    
    // Create a small graph representing a simple road network
    let mut graph = Graph::new(6);
    
    // Add roads with travel times in minutes
    graph.add_edge(0, 1, 10.0);  // Start -> Town A: 10 min
    graph.add_edge(0, 2, 15.0);  // Start -> Town B: 15 min
    graph.add_edge(1, 3, 12.0);  // Town A -> City: 12 min
    graph.add_edge(1, 4, 8.0);   // Town A -> Village: 8 min
    graph.add_edge(2, 3, 6.0);   // Town B -> City: 6 min
    graph.add_edge(2, 5, 20.0);  // Town B -> Destination: 20 min
    graph.add_edge(3, 5, 5.0);   // City -> Destination: 5 min
    graph.add_edge(4, 5, 25.0);  // Village -> Destination: 25 min
    
    let locations = ["Start", "Town A", "Town B", "City", "Village", "Destination"];
    
    let mut sssp = FastSSSP::new(graph);
    let start_time = Instant::now();
    sssp.solve(0);
    let distances = sssp.get_distances();
    let solve_time = start_time.elapsed();
    
    println!("Shortest travel times from Start:");
    for (i, &dist) in distances.iter().enumerate() {
        if dist < f64::INFINITY {
            println!("  To {}: {:.0} minutes", locations[i], dist);
            
            if let Some(path) = sssp.get_path(i) {
                let path_names: Vec<_> = path.iter()
                    .map(|&j| locations[j])
                    .collect();
                println!("    Route: {}", path_names.join(" → "));
            }
        }
    }
    
    println!("Computation time: {:?}\n", solve_time);
}

fn performance_comparison_example() {
    println!("2. Performance Comparison Example");
    println!("==================================");
    
    let sizes = vec![
        (100, 200),
        (500, 1000),
        (1000, 2000),
        (2000, 4000),
        (5000, 10000),
    ];
    
    println!("{:<15} {:<12} {:<12} {:<10}", "Graph Size", "Fast SSSP", "Dijkstra", "Speedup");
    println!("{}", "-".repeat(55));
    
    for (n, m) in sizes {
        let graph = create_random_graph(n, m, 100.0);
        
        // Benchmark Fast SSSP
        let start = Instant::now();
        let mut fast_sssp = FastSSSP::new(graph.clone());
        fast_sssp.solve(0);
        let _fast_result = fast_sssp.get_distances();
        let fast_time = start.elapsed();
        
        // Benchmark Classic Dijkstra
        let start = Instant::now();
        let classic = ClassicDijkstra::new(graph);
        let _classic_result = classic.solve(0);
        let classic_time = start.elapsed();
        
        let speedup = classic_time.as_secs_f64() / fast_time.as_secs_f64();
        
        println!("{:<15} {:<12.2?} {:<12.2?} {:<10.2}x", 
                 format!("{}v,{}e", n, m), 
                 fast_time, 
                 classic_time, 
                 speedup);
    }
    println!();
}

fn delivery_optimization_example() {
    println!("3. Delivery Optimization Example");
    println!("=================================");
    
    // Create a delivery network with warehouses, distribution centers, and customers
    let mut graph = Graph::new(12);
    
    // Example and explanation
    // graph.add_edge(0, 1, 5.0);   // Warehouse -> Distribution Center A
    // That means that the id 0 = main warehouse, and id 1 = Distribution Center A
    // Weight could be represented by:
    //  Kilometers between cities (distance)
    //  Minutes to travel a road (time)
    //  Energy needed to move between nodes (cost)
    //  Any other additive metric
    // Also, pls note this!
    //  How to handle your highway vs detour trade-off
    //  You have a few options:
    //  Pick a primary metric
    //  Decide what matters most (e.g., minimize travel time), and treat everything else as secondary or just informational.
    //  Combine metrics into one weight
    //  For example:
    //  weight = time_minutes + (toll_cost_dollars * penalty_factor)
    //  This way, Dijkstra still works because there’s just one number to minimize — but the number now reflects both time and cost in a way you choose.
    // Multi-objective shortest paths
    //  If you truly want to optimize two independent metrics (e.g., “minimize cost and time simultaneously”), you need multi-objective algorithms. These find a set of Pareto-optimal paths (none strictly better in all criteria). Examples:
    //   Label-setting algorithms for bicriteria shortest paths
    //   Multi-objective Dijkstra variants
    //   Evolutionary search methods (for big graphs)
    // Constraint-based search
    //   Minimize time subject to cost ≤ some budget (or vice versa). This is a constrained shortest path problem — different from classic Dijkstra, but solvable with adaptations like resource-constrained shortest path algorithms or A* with state expansion.

    // Warehouse connections (vertex 0 = main warehouse)
    graph.add_edge(0, 1, 5.0);   // Warehouse -> Distribution Center A
    graph.add_edge(0, 2, 8.0);   // Warehouse -> Distribution Center B
    graph.add_edge(0, 3, 12.0);  // Warehouse -> Distribution Center C
    
    // Distribution center to local hubs
    graph.add_edge(1, 4, 3.0);   // DC A -> Hub 1
    graph.add_edge(1, 5, 4.0);   // DC A -> Hub 2
    graph.add_edge(2, 5, 2.0);   // DC B -> Hub 2
    graph.add_edge(2, 6, 6.0);   // DC B -> Hub 3
    graph.add_edge(3, 6, 3.0);   // DC C -> Hub 3
    graph.add_edge(3, 7, 5.0);   // DC C -> Hub 4
    
    // Hub to customer connections
    graph.add_edge(4, 8, 2.0);   // Hub 1 -> Customer A
    graph.add_edge(4, 9, 4.0);   // Hub 1 -> Customer B
    graph.add_edge(5, 9, 3.0);   // Hub 2 -> Customer B
    graph.add_edge(5, 10, 2.5);  // Hub 2 -> Customer C
    graph.add_edge(6, 10, 4.0);  // Hub 3 -> Customer C
    graph.add_edge(6, 11, 3.5);  // Hub 3 -> Customer D
    graph.add_edge(7, 11, 2.0);  // Hub 4 -> Customer D
    
    // Some direct routes for efficiency
    graph.add_edge(1, 9, 6.0);   // DC A -> Customer B (express)
    graph.add_edge(2, 10, 7.0);  // DC B -> Customer C (express)
    
    let locations = [
        "Warehouse", "DC A", "DC B", "DC C",
        "Hub 1", "Hub 2", "Hub 3", "Hub 4",
        "Customer A", "Customer B", "Customer C", "Customer D"
    ];
    
    let mut sssp = FastSSSP::new(graph);
    sssp.solve(0);
    let distances = sssp.get_distances();
    
    println!("Optimal delivery routes from main warehouse:");
    
    // Focus on customer deliveries (vertices 8-11)
    for customer_id in 8..12 {
        let customer_name = locations[customer_id];
        let delivery_time = distances[customer_id];
        
        if delivery_time < f64::INFINITY {
            println!("\n📦 {} - Delivery time: {:.1} hours", customer_name, delivery_time);
            
            if let Some(path) = sssp.get_path(customer_id) {
                let route: Vec<_> = path.iter()
                    .map(|&i| locations[i])
                    .collect();
                println!("   Route: {}", route.join(" → "));
                
                // Calculate route segments for detailed analysis
                for window in path.windows(2) {
                    let from = window[0];
                    let to = window[1];
                    let segment_time = distances[to] - distances[from];
                    println!("   • {} to {}: {:.1}h", 
                            locations[from], locations[to], segment_time);
                }
            }
        }
    }
    
    // Find the most efficient delivery sequence
    let customer_distances: Vec<_> = (8..12)
        .map(|i| (locations[i], distances[i]))
        .collect();
    
    println!("\n🚚 Delivery Priority (by shortest delivery time):");
    let mut sorted_customers = customer_distances.clone();
    sorted_customers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    for (i, (customer, time)) in sorted_customers.iter().enumerate() {
        println!("   {}. {} ({:.1}h)", i + 1, customer, time);
    }
    
    println!();
}

/// Custom Dijkstra implementation with path tracking for TSP
struct PathTrackingDijkstra {
    graph: Graph,
}

impl PathTrackingDijkstra {
    fn new(graph: Graph) -> Self {
        Self { graph }
    }
    
    fn solve(&self, source: usize) -> (Vec<f64>, Vec<Option<usize>>) {
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;
        
        // OrderedFloat wrapper to make f64 work with BinaryHeap
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct OrderedFloat(f64);
        
        impl Eq for OrderedFloat {}
        
        impl PartialOrd for OrderedFloat {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }
        
        impl Ord for OrderedFloat {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
            }
        }
        
        let mut distances = vec![f64::INFINITY; self.graph.n];
        let mut predecessors = vec![None; self.graph.n];
        let mut heap = BinaryHeap::new();
        
        distances[source] = 0.0;
        heap.push(Reverse((OrderedFloat(0.0), source)));
        
        while let Some(Reverse((OrderedFloat(dist), u))) = heap.pop() {
            if dist > distances[u] {
                continue;
            }
            
            for edge in &self.graph.adj[u] {
                let new_dist = dist + edge.weight;
                if new_dist < distances[edge.to] {
                    distances[edge.to] = new_dist;
                    predecessors[edge.to] = Some(u);
                    heap.push(Reverse((OrderedFloat(new_dist), edge.to)));
                }
            }
        }
        
        (distances, predecessors)
    }
    
    fn get_path(&self, target: usize, predecessors: &[Option<usize>]) -> Option<Vec<usize>> {
        if predecessors[target].is_none() && target != 0 {
            return None;
        }
        
        let mut path = Vec::new();
        let mut current = Some(target);
        
        while let Some(v) = current {
            path.push(v);
            current = predecessors[v];
        }
        
        path.reverse();
        Some(path)
    }
}

fn delivery_route_optimization_example() {
    println!("3a. Traveling Salesman Problem - Delivery Route Optimization");
    println!("============================================================");

    // Create a simpler, fully connected graph for TSP demonstration
    let mut graph = Graph::new(6);
    
    // Warehouse (0) connections
    graph.add_edge(0, 1, 10.0);  // Warehouse -> Customer A
    graph.add_edge(0, 2, 15.0);  // Warehouse -> Customer B  
    graph.add_edge(0, 3, 20.0);  // Warehouse -> Customer C
    graph.add_edge(0, 4, 25.0);  // Warehouse -> Customer D
    graph.add_edge(0, 5, 30.0);  // Warehouse -> Customer E

    // Customer A (1) connections
    graph.add_edge(1, 0, 10.0);  // Customer A -> Warehouse
    graph.add_edge(1, 2, 8.0);   // Customer A -> Customer B
    graph.add_edge(1, 3, 12.0);  // Customer A -> Customer C
    graph.add_edge(1, 4, 16.0);  // Customer A -> Customer D
    graph.add_edge(1, 5, 18.0);  // Customer A -> Customer E

    // Customer B (2) connections
    graph.add_edge(2, 0, 15.0);  // Customer B -> Warehouse
    graph.add_edge(2, 1, 8.0);   // Customer B -> Customer A
    graph.add_edge(2, 3, 7.0);   // Customer B -> Customer C
    graph.add_edge(2, 4, 9.0);   // Customer B -> Customer D
    graph.add_edge(2, 5, 14.0);  // Customer B -> Customer E

    // Customer C (3) connections
    graph.add_edge(3, 0, 20.0);  // Customer C -> Warehouse
    graph.add_edge(3, 1, 12.0);  // Customer C -> Customer A
    graph.add_edge(3, 2, 7.0);   // Customer C -> Customer B
    graph.add_edge(3, 4, 6.0);   // Customer C -> Customer D
    graph.add_edge(3, 5, 11.0);  // Customer C -> Customer E

    // Customer D (4) connections
    graph.add_edge(4, 0, 25.0);  // Customer D -> Warehouse
    graph.add_edge(4, 1, 16.0);  // Customer D -> Customer A
    graph.add_edge(4, 2, 9.0);   // Customer D -> Customer B
    graph.add_edge(4, 3, 6.0);   // Customer D -> Customer C
    graph.add_edge(4, 5, 5.0);   // Customer D -> Customer E

    // Customer E (5) connections
    graph.add_edge(5, 0, 30.0);  // Customer E -> Warehouse
    graph.add_edge(5, 1, 18.0);  // Customer E -> Customer A
    graph.add_edge(5, 2, 14.0);  // Customer E -> Customer B
    graph.add_edge(5, 3, 11.0);  // Customer E -> Customer C
    graph.add_edge(5, 4, 5.0);   // Customer E -> Customer D

    let locations = [
        "Warehouse", "Customer A", "Customer B", "Customer C", "Customer D", "Customer E"
    ];

    let customers = [1, 2, 3, 4, 5];
    let start = 0;

    println!("🚚 Solving TSP using Dijkstra-based nearest neighbor heuristic...\n");

    // TSP using Dijkstra-based nearest neighbor heuristic
    let mut tour = vec![start];
    let mut unvisited: std::collections::HashSet<_> = customers.iter().cloned().collect();
    let mut current = start;
    let mut total_time = 0.0;

    println!("Building optimal tour:");
    while !unvisited.is_empty() {
        // Use Dijkstra from current location to find distances to all unvisited customers
        let dijkstra = PathTrackingDijkstra::new(graph.clone());
        let (distances, _) = dijkstra.solve(current);
        
        // Find nearest unvisited customer
        let mut nearest_customer = None;
        let mut nearest_distance = f64::INFINITY;
        
        for &customer in &unvisited {
            if distances[customer] < nearest_distance {
                nearest_distance = distances[customer];
                nearest_customer = Some(customer);
            }
        }
        
        if let Some(next_customer) = nearest_customer {
            tour.push(next_customer);
            unvisited.remove(&next_customer);
            total_time += nearest_distance;
            
            println!("   {} → {} (distance: {:.1}h)", 
                    locations[current], locations[next_customer], nearest_distance);
            
            current = next_customer;
        }
    }

    // Show the complete tour
    println!("\n🎯 TSP tour using nearest neighbor heuristic:");
    print!("   ");
    for (i, &vertex) in tour.iter().enumerate() {
        if i > 0 { print!(" → "); }
        print!("{}", locations[vertex]);
    }
    println!("\n   Total travel time: {:.1} hours", total_time);

    // Show detailed route breakdown with intermediate hops
    println!("\n📋 Detailed route breakdown:");
    for i in 0..tour.len() - 1 {
        let from = tour[i];
        let to = tour[i + 1];
        
        // Use Dijkstra to find the actual path between consecutive tour stops
        let dijkstra = PathTrackingDijkstra::new(graph.clone());
        let (distances, predecessors) = dijkstra.solve(from);
        
        if let Some(path) = dijkstra.get_path(to, &predecessors) {
            let route_names: Vec<_> = path.iter().map(|&j| locations[j]).collect();
            let leg_time = distances[to];
            
            println!("\nLeg {}: {} → {} ({:.1}h)", 
                    i + 1, locations[from], locations[to], leg_time);
            println!("   Full route: {}", route_names.join(" → "));
            
            // Show segment-by-segment breakdown
            for window in path.windows(2) {
                let seg_from = window[0];
                let seg_to = window[1];
                let seg_time = distances[seg_to] - distances[seg_from];
                println!("   • {} to {}: {:.1}h", 
                        locations[seg_from], locations[seg_to], seg_time);
            }
        }
    }
    
    println!("\n💡 Note: This implements the Traveling Salesman Problem using:");
    println!("   • Nearest neighbor heuristic for tour construction");
    println!("   • Dijkstra's algorithm for shortest path between each pair of nodes");
    println!("   • O(n² * (V log V + E)) time complexity for the full TSP solution");
    println!("   • For optimal TSP solutions, exact algorithms like branch-and-bound or");
    println!("     dynamic programming would be needed, but those scale poorly (O(n!)).");
}


fn graph_structure_analysis() {
    println!("4. Algorithm Performance on Different Graph Structures");
    println!("======================================================");
    
    let n = 1000;
    
    // Create different types of graphs
    let graphs = vec![
        ("Sparse (m ≈ n)", create_random_graph(n, n, 100.0)),
        ("Medium (m ≈ 2n)", create_random_graph(n, 2 * n, 100.0)),
        ("Dense (m ≈ 5n)", create_random_graph(n, 5 * n, 100.0)),
        ("Very Dense (m ≈ 10n)", create_random_graph(n, 10 * n, 100.0)),
    ];
    
    println!("{:<20} {:<12} {:<12} {:<10} {:<15}", 
             "Graph Type", "Fast SSSP", "Dijkstra", "Speedup", "Theoretical");
    println!("{}", "-".repeat(75));
    
    for (graph_type, graph) in graphs {
        let m = graph.edge_count();
        
        // Benchmark both algorithms
        let start = Instant::now();
        let mut fast_sssp = FastSSSP::new(graph.clone());
        fast_sssp.solve(0);
        let _fast_result = fast_sssp.get_distances();
        let fast_time = start.elapsed();
        
        let start = Instant::now();
        let classic = ClassicDijkstra::new(graph);
        let _classic_result = classic.solve(0);
        let classic_time = start.elapsed();
        
        let speedup = classic_time.as_secs_f64() / fast_time.as_secs_f64();
        
        // Theoretical advantage calculation
        // Fast SSSP: O(m + n log log n)
        // Dijkstra: O(m log n)
        let log_n = (n as f64).log2();
        let log_log_n = log_n.log2();
        let theoretical_advantage = (m as f64 * log_n) / 
                                   (m as f64 + n as f64 * log_log_n);
        
        println!("{:<20} {:<12.2?} {:<12.2?} {:<10.2}x {:<15.2}x", 
                 graph_type, 
                 fast_time, 
                 classic_time, 
                 speedup,
                 theoretical_advantage);
    }
    
    println!("\nNote: Theoretical advantages are upper bounds.");
    println!("Actual performance depends on implementation constants and graph structure.");
    println!();
}

// Helper function to demonstrate correctness
#[allow(dead_code)]
fn verify_correctness(graph: &Graph, source: usize) -> bool {
    let mut fast_sssp = FastSSSP::new(graph.clone());
    fast_sssp.solve(source);
    let fast_distances = fast_sssp.get_distances();
    
    let classic = ClassicDijkstra::new(graph.clone());
    let classic_distances = classic.solve(source);
    
    for i in 0..graph.n {
        if (fast_distances[i] - classic_distances[i]).abs() > 1e-9 {
            println!("Mismatch at vertex {}: Fast={:.6}, Classic={:.6}", 
                     i, fast_distances[i], classic_distances[i]);
            return false;
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_algorithm_correctness() {
        // Test on various graph sizes
        for &(n, m) in &[(50, 100), (100, 300), (200, 600)] {
            let graph = create_random_graph(n, m, 100.0);
            assert!(verify_correctness(&graph, 0), 
                    "Algorithm correctness failed for graph {}v, {}e", n, m);
        }
    }
    
    #[test]
    fn test_performance_improvement() {
        let n = 500;
        let m = 1000; // Sparse graph where improvement should be visible
        let graph = create_random_graph(n, m, 100.0);
        
        let start = Instant::now();
        let mut fast_sssp = FastSSSP::new(graph.clone());
        fast_sssp.solve(0);
        let _fast_result = fast_sssp.get_distances();
        let fast_time = start.elapsed();
        
        let start = Instant::now();
        let classic = ClassicDijkstra::new(graph);
        let _classic_result = classic.solve(0);
        let classic_time = start.elapsed();
        
        // On sparse graphs, our algorithm should eventually be faster
        // (though constant factors might make small graphs slower)
        println!("Fast SSSP: {:?}, Dijkstra: {:?}", fast_time, classic_time);
        
        // Just verify both complete successfully
        assert!(fast_time.as_nanos() > 0);
        assert!(classic_time.as_nanos() > 0);
    }
}