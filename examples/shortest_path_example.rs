use fast_sssp::{Graph, FastSSSP};
use std::io::{self, Write};
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Example program that finds the shortest path between two nodes
/// from the LiveJournal dataset
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("LiveJournal Dataset Shortest Path Finder");
    println!("========================================");
    
    // Get user input for source and target nodes
    print!("Enter source node (i): ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let source: usize = input.trim().parse()?;
    
    print!("Enter target node (j): ");
    io::stdout().flush()?;
    input.clear();
    io::stdin().read_line(&mut input)?;
    let target: usize = input.trim().parse()?;
    
    println!("\nLoading LiveJournal dataset...");
    let graph = load_livejournal_dataset("dataset/soc-LiveJournal1.txt")?;
    
    println!("Graph loaded: {} vertices, {} edges", graph.n, graph.edge_count());
    
    // Check if nodes exist in the graph
    if source >= graph.n {
        println!("Error: Source node {} does not exist (max: {})", source, graph.n - 1);
        return Ok(());
    }
    if target >= graph.n {
        println!("Error: Target node {} does not exist (max: {})", target, graph.n - 1);
        return Ok(());
    }
    
    println!("Finding shortest path from {} to {}...", source, target);
    
    // Solve shortest paths from source
    let mut sssp = FastSSSP::new(graph);
    let start = std::time::Instant::now();
    sssp.solve(source);
    let solve_time = start.elapsed();
    
    // Get results
    let distances = sssp.get_distances();
    let distance = distances[target];
    
    if distance == f64::INFINITY {
        println!("No path exists from {} to {}", source, target);
    } else {
        println!("Shortest distance from {} to {}: {}", source, target, distance as usize);
        
        if let Some(path) = sssp.get_path(target) {
            println!("Shortest path: {:?}", path);
            println!("Path length: {} hops", path.len() - 1);
            
            // Show path in a more readable format
            if path.len() <= 10 {
                print!("Path: ");
                for (i, &node) in path.iter().enumerate() {
                    if i > 0 { print!(" -> "); }
                    print!("{}", node);
                }
                println!();
            } else {
                println!("Path (first 5 and last 5 nodes): {} -> {} -> {} -> {} -> {} -> ... -> {} -> {} -> {} -> {} -> {}", 
                    path[0], path[1], path[2], path[3], path[4],
                    path[path.len()-5], path[path.len()-4], path[path.len()-3], path[path.len()-2], path[path.len()-1]);
            }
        }
    }
    
    println!("Computation time: {:?}", solve_time);
    
    Ok(())
}

/// Load the LiveJournal dataset from file
/// Format: Each line is "from_node\tto_node" (tab-separated)
/// All edges have weight = 1.0 (unweighted graph)
fn load_livejournal_dataset(filename: &str) -> Result<Graph, Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    
    let mut edges = Vec::new();
    let mut max_node = 0;
    let mut line_count = 0;
    
    println!("Reading edges from {}...", filename);
    
    for line in reader.lines() {
        let line = line?;
        line_count += 1;
        
        // Skip comments (lines starting with #)
        if line.starts_with('#') {
            continue;
        }
        
        // Parse edge: from_node \t to_node
        let parts: Vec<&str> = line.trim().split('\t').collect();
        if parts.len() != 2 {
            continue; // Skip malformed lines
        }
        
        if let (Ok(from), Ok(to)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
            edges.push((from, to));
            max_node = max_node.max(from).max(to);
        }
        
        // Progress indicator
        if line_count % 1_000_000 == 0 {
            println!("  Processed {} lines, found {} edges", line_count, edges.len());
        }
    }
    
    let num_vertices = max_node + 1;
    println!("Creating graph with {} vertices and {} edges...", num_vertices, edges.len());
    
    // Create graph
    let mut graph = Graph::new(num_vertices);
    
    // Add all edges with weight 1.0 (since the dataset is unweighted)
    for (i, (from, to)) in edges.into_iter().enumerate() {
        graph.add_edge(from, to, 1.0);
        
        // Progress indicator for edge addition
        if (i + 1) % 1_000_000 == 0 {
            println!("  Added {} edges to graph", i + 1);
        }
    }
    
    Ok(graph)
}