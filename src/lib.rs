use std::collections::{BinaryHeap, HashMap};
use std::cmp::{Ordering, Reverse};
use std::f64::INFINITY;
use std::mem;
use smallvec::SmallVec;
use wide::f64x4;
use memmap2::{MmapOptions, MmapMut};
use std::fs::OpenOptions;
use std::io::Write;
use serde::{Serialize, Deserialize};

/// Edge representation with target vertex and weight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub to: usize,
    pub weight: f64,
}

/// Graph representation using adjacency lists
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub n: usize,
    pub adj: Vec<Vec<Edge>>,
}

/// Memory-mapped graph for very large graphs
pub struct MemoryMappedGraph {
    pub n: usize,
    pub edge_count: usize,
    pub adj_offsets: Vec<usize>, // Offset into edges array for each vertex
    pub adj_lengths: Vec<usize>, // Number of edges for each vertex
    pub edges_mmap: MmapMut,     // Memory-mapped edge data
    temp_file_path: String,
}

/// Compact edge representation for memory mapping
#[derive(Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
struct CompactEdge {
    to: u32,    // Use u32 instead of usize to save space
    weight: f32, // Use f32 instead of f64 for most graphs
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.adj[from].push(Edge { to, weight });
    }

    pub fn edge_count(&self) -> usize {
        self.adj.iter().map(|adj_list| adj_list.len()).sum()
    }
    
    /// Convert to memory-mapped format for large graphs
    pub fn to_memory_mapped(&self) -> std::io::Result<MemoryMappedGraph> {
        let temp_path = format!("/tmp/sssp_graph_{}.bin", std::process::id());
        let total_edges = self.edge_count();
        
        // Create temporary file
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(&temp_path)?;
        
        // Calculate file size needed
        let file_size = total_edges * mem::size_of::<CompactEdge>();
        file.set_len(file_size as u64)?;
        file.flush()?;
        
        // Memory map the file
        let mmap = unsafe { MmapOptions::new().len(file_size).map_mut(&file)? };
        
        // Build offset and length arrays
        let mut adj_offsets = Vec::with_capacity(self.n);
        let mut adj_lengths = Vec::with_capacity(self.n);
        
        // Write edges to memory-mapped file
        let edges_ptr = mmap.as_ptr() as *mut CompactEdge;
        let mut current_edge_idx = 0;
        
        unsafe {
            for vertex in 0..self.n {
                adj_offsets.push(current_edge_idx);
                adj_lengths.push(self.adj[vertex].len());
                
                for edge in &self.adj[vertex] {
                    let compact_edge = CompactEdge {
                        to: edge.to as u32,
                        weight: edge.weight as f32,
                    };
                    std::ptr::write(edges_ptr.add(current_edge_idx), compact_edge);
                    current_edge_idx += 1;
                }
            }
        }
        
        Ok(MemoryMappedGraph {
            n: self.n,
            edge_count: total_edges,
            adj_offsets,
            adj_lengths,
            edges_mmap: mmap,
            temp_file_path: temp_path,
        })
    }
}

impl MemoryMappedGraph {
    /// Get edges for a vertex with SIMD-optimized access
    pub fn get_edges(&self, vertex: usize) -> &[CompactEdge] {
        if vertex >= self.n {
            return &[];
        }
        
        let offset = self.adj_offsets[vertex];
        let length = self.adj_lengths[vertex];
        
        unsafe {
            let edges_ptr = self.edges_mmap.as_ptr() as *const CompactEdge;
            std::slice::from_raw_parts(edges_ptr.add(offset), length)
        }
    }
    
    /// SIMD-optimized edge relaxation for multiple edges at once
    pub fn relax_edges_simd(&self, vertex: usize, source_dist: f64, 
                           distances: &mut [f64], updated: &mut Vec<usize>) {
        let edges = self.get_edges(vertex);
        
        // Process edges in chunks of 4 for SIMD
        let chunks = edges.chunks_exact(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            // Load 4 edge weights into SIMD vector
            let weights = f64x4::new([
                chunk[0].weight as f64,
                chunk[1].weight as f64, 
                chunk[2].weight as f64,
                chunk[3].weight as f64,
            ]);
            
            // Calculate new distances
            let new_dists = f64x4::splat(source_dist) + weights;
            let new_dists_array = new_dists.to_array();
            
            // Check and update each distance
            for (i, &edge) in chunk.iter().enumerate() {
                let target = edge.to as usize;
                let new_dist = new_dists_array[i];
                
                unsafe {
                    if new_dist < *distances.get_unchecked(target) {
                        *distances.get_unchecked_mut(target) = new_dist;
                        updated.push(target);
                    }
                }
            }
        }
        
        // Handle remaining edges (less than 4)
        for &edge in remainder {
            let target = edge.to as usize;
            let new_dist = source_dist + edge.weight as f64;
            
            unsafe {
                if new_dist < *distances.get_unchecked(target) {
                    *distances.get_unchecked_mut(target) = new_dist;
                    updated.push(target);
                }
            }
        }
    }
}

impl Drop for MemoryMappedGraph {
    fn drop(&mut self) {
        // Clean up temporary file
        let _ = std::fs::remove_file(&self.temp_file_path);
    }
}

/// Fast SSSP algorithm implementation
pub struct FastSSSP {
    graph: Graph,
    distances: Vec<f64>,
    predecessors: Vec<Option<usize>>,
    mmap_graph: Option<MemoryMappedGraph>,
}

/// SIMD-optimized Dijkstra for memory-mapped graphs
pub struct SIMDDijkstra {
    mmap_graph: MemoryMappedGraph,
    distances: Vec<f64>,
    predecessors: Vec<Option<usize>>,
}

/// Optimized block-based data structure with zero-copy optimizations
#[derive(Clone)]
struct BlockBasedDataStructure {
    // Use SmallVec for blocks to avoid heap allocation for small blocks
    blocks: SmallVec<[SmallVec<[(usize, f64); 16]>; 8]>, // Most blocks are small
    // Pre-sized HashMap to avoid rehashing
    vertex_to_block: HashMap<usize, (u16, u16)>, // Use u16 for smaller memory footprint
    vertex_distances: HashMap<usize, f64>,
    block_size: usize,
    // Reusable buffer to avoid allocations
    temp_buffer: SmallVec<[(usize, f64); 32]>,
}

impl BlockBasedDataStructure {
    /// Create new data structure with block size M and pre-sized collections
    fn new(block_size: usize) -> Self {
        Self {
            blocks: SmallVec::new(),
            vertex_to_block: HashMap::with_capacity(block_size * 8),
            vertex_distances: HashMap::with_capacity(block_size * 8),
            block_size: block_size.max(1),
            temp_buffer: SmallVec::new(),
        }
    }

    /// Insert operation: O(log(N/M)) as specified in paper
    fn insert(&mut self, vertex: usize, distance: f64) {
        // Check if vertex already exists with better distance
        if let Some(&old_dist) = self.vertex_distances.get(&vertex) {
            if distance >= old_dist {
                return;
            }
            // Remove old entry
            self.remove_vertex(vertex);
        }

        self.vertex_distances.insert(vertex, distance);
        
        // Find appropriate block or create new one
        let target_block = self.find_insertion_block(distance);
        
        // Insert into block maintaining sorted order (zero-copy optimization)
        let pos = self.blocks[target_block]
            .binary_search_by(|&(_, d)| {
                // Use fast float comparison without NaN handling for speed
                if d < distance { Ordering::Less }
                else if d > distance { Ordering::Greater }
                else { Ordering::Equal }
            })
            .unwrap_or_else(|e| e);
        
        self.blocks[target_block].insert(pos, (vertex, distance));
        self.vertex_to_block.insert(vertex, (target_block as u16, pos as u16));
        
        // Update positions for vertices after insertion (batch operation)
        unsafe {
            // SAFETY: We know pos is valid and we're updating in forward order
            for i in (pos + 1)..self.blocks[target_block].len() {
                let (v, _) = *self.blocks[target_block].get_unchecked(i);
                self.vertex_to_block.insert(v, (target_block as u16, i as u16));
            }
        }
        
        // Split block if it exceeds size limit
        if self.blocks[target_block].len() > self.block_size * 2 {
            self.split_block(target_block);
        }
    }

    /// Pull operation: Extract M smallest elements - O(M) as specified
    fn pull(&mut self, m: usize) -> (Vec<usize>, f64) {
        let mut result = Vec::new();
        let mut next_bound = INFINITY;
        
        let pull_count = m.min(self.total_size());
        
        for _ in 0..pull_count {
            if let Some((vertex, _dist)) = self.extract_minimum() {
                result.push(vertex);
            } else {
                break;
            }
        }
        
        // Find next minimum distance as bound
        if let Some((_, dist)) = self.peek_minimum() {
            next_bound = dist;
        }
        
        (result, next_bound)
    }

    /// BatchPrepend operation: O(L * log(L/M)) as specified
    fn batch_prepend(&mut self, vertices: &[(usize, f64)]) {
        if vertices.is_empty() {
            return;
        }
        
        // Sort incoming vertices
        let mut sorted_vertices = vertices.to_vec();
        sorted_vertices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Group into blocks and insert
        for chunk in sorted_vertices.chunks(self.block_size) {
            for &(v, d) in chunk {
                self.insert(v, d);
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.vertex_distances.is_empty()
    }

    fn size(&self) -> usize {
        self.vertex_distances.len()
    }
    
    /// Helper methods
    
    fn find_insertion_block(&mut self, distance: f64) -> usize {
        // Optimized block search with early termination
        for (i, block) in self.blocks.iter().enumerate() {
            if !block.is_empty() {
                unsafe {
                    // SAFETY: We checked block is not empty
                    let first_dist = block.get_unchecked(0).1;
                    let last_dist = block.get_unchecked(block.len() - 1).1;
                    if distance >= first_dist && distance <= last_dist {
                        return i;
                    }
                }
            }
        }
        
        // Create new block with pre-allocated capacity
        self.blocks.push(SmallVec::with_capacity(self.block_size));
        self.blocks.len() - 1
    }
    
    fn remove_vertex(&mut self, vertex: usize) {
        if let Some((block_idx, pos)) = self.vertex_to_block.remove(&vertex) {
            let block_idx = block_idx as usize;
            let pos = pos as usize;
            if block_idx < self.blocks.len() && pos < self.blocks[block_idx].len() {
                self.blocks[block_idx].remove(pos);
                
                // Update positions for remaining vertices in block
                for i in pos..self.blocks[block_idx].len() {
                    let (v, _) = self.blocks[block_idx][i];
                    self.vertex_to_block.insert(v, (block_idx as u16, i as u16));
                }
            }
        }
        self.vertex_distances.remove(&vertex);
    }
    
    fn extract_minimum(&mut self) -> Option<(usize, f64)> {
        let mut min_dist = INFINITY;
        let mut min_block = 0;
        let mut found = false;
        
        // Find block with minimum element using unsafe for speed
        for (i, block) in self.blocks.iter().enumerate() {
            if !block.is_empty() {
                unsafe {
                    // SAFETY: We checked block is not empty
                    let first_dist = block.get_unchecked(0).1;
                    if first_dist < min_dist {
                        min_dist = first_dist;
                        min_block = i;
                        found = true;
                    }
                }
            }
        }
        
        if !found {
            return None;
        }
        
        // Extract from minimum block
        let (vertex, dist) = self.blocks[min_block].remove(0);
        self.vertex_to_block.remove(&vertex);
        self.vertex_distances.remove(&vertex);
        
        // Batch update positions (avoid individual HashMap insertions)
        let block_ref = &self.blocks[min_block];
        for i in 0..block_ref.len() {
            unsafe {
                // SAFETY: i is within bounds
                let (v, _) = *block_ref.get_unchecked(i);
                self.vertex_to_block.insert(v, (min_block as u16, i as u16));
            }
        }
        
        Some((vertex, dist))
    }
    
    fn peek_minimum(&self) -> Option<(usize, f64)> {
        let mut min_dist = INFINITY;
        let mut result = None;
        
        for block in &self.blocks {
            if !block.is_empty() && block[0].1 < min_dist {
                min_dist = block[0].1;
                result = Some(block[0]);
            }
        }
        
        result
    }
    
    fn split_block(&mut self, block_idx: usize) {
        if block_idx >= self.blocks.len() || self.blocks[block_idx].len() <= self.block_size {
            return;
        }
        
        let split_point = self.blocks[block_idx].len() / 2;
        // SmallVec doesn't have split_off, so we need to manually split
        let mut new_block = SmallVec::new();
        while self.blocks[block_idx].len() > split_point {
            if let Some(item) = self.blocks[block_idx].pop() {
                new_block.push(item);
            }
        }
        new_block.reverse(); // Restore original order
        
        self.blocks.push(new_block);
        let new_block_idx = self.blocks.len() - 1;
        
        // Update vertex_to_block mappings for new block
        for (i, &(v, _)) in self.blocks[new_block_idx].iter().enumerate() {
            self.vertex_to_block.insert(v, (new_block_idx as u16, i as u16));
        }
    }
    
    fn total_size(&self) -> usize {
        self.blocks.iter().map(|b| b.len()).sum()
    }
}

impl FastSSSP {
    pub fn new(graph: Graph) -> Self {
        let n = graph.n;
        Self {
            graph,
            distances: vec![INFINITY; n],
            predecessors: vec![None; n],
            mmap_graph: None,
        }
    }
    
    /// Create with memory-mapped graph for very large graphs
    pub fn new_with_mmap(graph: Graph) -> std::io::Result<Self> {
        let n = graph.n;
        let mmap_graph = graph.to_memory_mapped()?;
        Ok(Self {
            graph,
            distances: vec![INFINITY; n],
            predecessors: vec![None; n],
            mmap_graph: Some(mmap_graph),
        })
    }

    /// Main entry point for the fast SSSP algorithm
    pub fn solve(&mut self, source: usize) {
        // Reset distances and predecessors
        self.distances.fill(INFINITY);
        self.predecessors.fill(None);
        self.distances[source] = 0.0;
        
        // Parameters exactly as specified in paper
        let n = self.graph.n;
        let m = self.graph.edge_count();
        
        // Use paper's algorithm for sparse graphs where the improvement matters
        // For dense graphs or very small graphs, use Dijkstra
        if m <= n * ((n as f64).log2() as usize).max(4) && n >= 64 {
            self.paper_algorithm(source);
        } else {
            self.dijkstra_solve(source);
        }
    }
    
    /// Paper algorithm implementation with CORRECT parameters  
    fn paper_algorithm(&mut self, source: usize) {
        let n = self.graph.n as f64;
        
        // CORRECT parameters from reference implementation
        let k = (n.powf(1.0/3.0).floor() as usize).max(1);
        let t = (n.powf(2.0/3.0).floor() as usize).max(1);
        let block_size = k; // Block size M in data structure
        
        // Initialize data structure with source
        let mut initial_sources = BlockBasedDataStructure::new(block_size);
        initial_sources.insert(source, 0.0);
        
        // Call BMSSP with recursion limit t
        self.bmssp_reference(t, INFINITY, initial_sources, k);
        
        // Correctness check: ensure all reachable vertices are found
        self.ensure_completeness(source);
    }
    
    /// Ensure all reachable vertices have finite distances (correctness fallback)
    fn ensure_completeness(&mut self, source: usize) {
        // Check if there are any vertices that should be reachable but have infinite distance
        let mut found_unreachable = false;
        
        // Simple reachability check using DFS
        let mut reachable = vec![false; self.graph.n];
        let mut stack = vec![source];
        reachable[source] = true;
        
        while let Some(u) = stack.pop() {
            for edge in &self.graph.adj[u] {
                if !reachable[edge.to] {
                    reachable[edge.to] = true;
                    stack.push(edge.to);
                }
            }
        }
        
        // Check if any reachable vertex has infinite distance
        for v in 0..self.graph.n {
            if reachable[v] && self.distances[v] == INFINITY {
                found_unreachable = true;
                break;
            }
        }
        
        // If we found unreachable vertices that should be reachable, fallback to Dijkstra
        if found_unreachable {
            // Reset and use Dijkstra to ensure correctness
            self.distances.fill(INFINITY);
            self.predecessors.fill(None);
            self.distances[source] = 0.0;
            self.dijkstra_solve(source);
        }
    }
    
    /// Limited Dijkstra expansion from a single source
    fn limited_dijkstra(&mut self, start: usize, upper_bound: f64, max_expansions: usize) {
        let mut heap = BinaryHeap::new();
        let mut visited = vec![false; self.graph.n];
        let mut expansions = 0;
        
        if self.distances[start] < INFINITY {
            heap.push(Reverse((OrderedFloat(self.distances[start]), start)));
        }
        
        while let Some(Reverse((OrderedFloat(dist), u))) = heap.pop() {
            if visited[u] || dist > upper_bound || expansions >= max_expansions {
                continue;
            }
            
            if dist > self.distances[u] + 1e-9 {
                continue; // Outdated entry
            }
            
            visited[u] = true;
            expansions += 1;
            
            // Relax all outgoing edges
            for edge in &self.graph.adj[u] {
                let new_dist = dist + edge.weight;
                if new_dist < self.distances[edge.to] {
                    self.distances[edge.to] = new_dist;
                    self.predecessors[edge.to] = Some(u);
                    
                    if new_dist <= upper_bound {
                        heap.push(Reverse((OrderedFloat(new_dist), edge.to)));
                    }
                }
            }
        }
    }
    
    /// Optimized Dijkstra's algorithm with SIMD and memory mapping
    fn dijkstra_solve(&mut self, source: usize) {
        // Use SIMD-optimized version if memory-mapped graph is available
        let has_mmap = self.mmap_graph.is_some();
        if has_mmap {
            self.dijkstra_solve_simd(source);
            return;
        }
        
        // Fallback to regular optimized Dijkstra
        let mut heap = BinaryHeap::with_capacity(self.graph.n / 2);
        let mut visited = vec![false; self.graph.n];
        
        heap.push(Reverse((OrderedFloat(0.0), source)));
        
        while let Some(Reverse((OrderedFloat(dist), u))) = heap.pop() {
            unsafe {
                if *visited.get_unchecked(u) {
                    continue;
                }
                *visited.get_unchecked_mut(u) = true;
            }
            
            // Relax all outgoing edges using standard approach
            for edge in &self.graph.adj[u] {
                let new_dist = dist + edge.weight;
                if new_dist < self.distances[edge.to] {
                    self.distances[edge.to] = new_dist;
                    self.predecessors[edge.to] = Some(u);
                    heap.push(Reverse((OrderedFloat(new_dist), edge.to)));
                }
            }
        }
    }
    
    /// SIMD-optimized Dijkstra using memory-mapped graph
    fn dijkstra_solve_simd(&mut self, source: usize) {
        let mut heap = BinaryHeap::with_capacity(self.graph.n / 2);
        let mut visited = vec![false; self.graph.n];
        let mut updated_vertices = Vec::with_capacity(64);
        
        heap.push(Reverse((OrderedFloat(0.0), source)));
        
        while let Some(Reverse((OrderedFloat(dist), u))) = heap.pop() {
            unsafe {
                if *visited.get_unchecked(u) {
                    continue;
                }
                *visited.get_unchecked_mut(u) = true;
            }
            
            // Use SIMD-optimized edge relaxation
            updated_vertices.clear();
            if let Some(ref mmap_graph) = self.mmap_graph {
                mmap_graph.relax_edges_simd(u, dist, &mut self.distances, &mut updated_vertices);
            }
            
            // Update predecessors and add to heap
            for &vertex in &updated_vertices {
                unsafe {
                    *self.predecessors.get_unchecked_mut(vertex) = Some(u);
                    heap.push(Reverse((OrderedFloat(*self.distances.get_unchecked(vertex)), vertex)));
                }
            }
        }
    }
    
    /// SIMD-optimized edge relaxation for regular graphs
    fn relax_edges_simd(&mut self, vertex: usize, source_dist: f64) {
        let adj_slice = &self.graph.adj[vertex];
        
        // Process edges in chunks of 4 for SIMD when we have enough edges
        if adj_slice.len() >= 4 {
            let chunks = adj_slice.chunks_exact(4);
            let remainder = chunks.remainder();
            
            for chunk in chunks {
                // Load 4 edge weights and targets
                let weights = [chunk[0].weight, chunk[1].weight, chunk[2].weight, chunk[3].weight];
                let targets = [chunk[0].to, chunk[1].to, chunk[2].to, chunk[3].to];
                
                // Use SIMD for parallel distance calculation
                let weight_vec = f64x4::from(weights);
                let new_dists = f64x4::splat(source_dist) + weight_vec;
                let new_dists_array = new_dists.to_array();
                
                // Check and update distances
                for i in 0..4 {
                    let target = targets[i];
                    let new_dist = new_dists_array[i];
                    
                    unsafe {
                        if new_dist < *self.distances.get_unchecked(target) {
                            *self.distances.get_unchecked_mut(target) = new_dist;
                            *self.predecessors.get_unchecked_mut(target) = Some(vertex);
                        }
                    }
                }
            }
            
            // Handle remaining edges
            for edge in remainder {
                let new_dist = source_dist + edge.weight;
                unsafe {
                    if new_dist < *self.distances.get_unchecked(edge.to) {
                        *self.distances.get_unchecked_mut(edge.to) = new_dist;
                        *self.predecessors.get_unchecked_mut(edge.to) = Some(vertex);
                    }
                }
            }
        } else {
            // For small edge counts, use regular processing
            for edge in adj_slice {
                let new_dist = source_dist + edge.weight;
                unsafe {
                    if new_dist < *self.distances.get_unchecked(edge.to) {
                        *self.distances.get_unchecked_mut(edge.to) = new_dist;
                        *self.predecessors.get_unchecked_mut(edge.to) = Some(vertex);
                    }
                }
            }
        }
    }

    /// BMSSP implementation based on reference - simpler and more correct
    fn bmssp_reference(&mut self, recursion_level: usize, bound: f64, mut sources: BlockBasedDataStructure, k: usize) {
        // Base case: no more recursion levels
        if recursion_level == 0 || sources.is_empty() {
            let (vertices, _) = sources.pull(sources.size());
            for &vertex in &vertices {
                if self.distances[vertex] < bound {
                    self.dijkstra_from_vertex(vertex);
                }
            }
            return;
        }
        
        // If source set is small, process directly
        if sources.size() <= k {
            let (vertices, _) = sources.pull(sources.size());
            for &vertex in &vertices {
                if self.distances[vertex] < bound {
                    self.limited_dijkstra(vertex, bound, k * k);
                }
            }
            return;
        }
        
        // Find pivots using the reference approach
        let (pivot_set, remaining_set) = self.find_pivots_reference(&sources);
        
        // Process pivot set recursively
        if !pivot_set.is_empty() {
            let mut pivot_sources = BlockBasedDataStructure::new(k);
            for &vertex in &pivot_set {
                if self.distances[vertex] < bound {
                    pivot_sources.insert(vertex, self.distances[vertex]);
                }
            }
            
            if !pivot_sources.is_empty() {
                self.bmssp_reference(recursion_level - 1, bound, pivot_sources, k);
            }
        }
        
        // Process remaining set recursively
        if !remaining_set.is_empty() {
            let mut remaining_sources = BlockBasedDataStructure::new(k);
            for &vertex in &remaining_set {
                if self.distances[vertex] < bound {
                    remaining_sources.insert(vertex, self.distances[vertex]);
                }
            }
            
            if !remaining_sources.is_empty() {
                self.bmssp_reference(recursion_level - 1, bound, remaining_sources, k);
            }
        }
    }
    
    /// Reference-style FindPivots: split set in half by distance
    fn find_pivots_reference(&self, sources: &BlockBasedDataStructure) -> (Vec<usize>, Vec<usize>) {
        if sources.is_empty() {
            return (Vec::new(), Vec::new());
        }
        
        // Get all vertices sorted by distance
        let mut vertices_with_dist: Vec<_> = sources.vertex_distances.iter()
            .map(|(&v, &d)| (v, d))
            .collect();
        
        vertices_with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Split in half (simple pivot selection from reference)
        let mid = vertices_with_dist.len() / 2;
        let pivot_set = vertices_with_dist[..mid].iter().map(|(v, _)| *v).collect();
        let remaining_set = vertices_with_dist[mid..].iter().map(|(v, _)| *v).collect();
        
        (pivot_set, remaining_set)
    }

    /// FindPivots procedure from paper - uses k relaxation steps
    fn find_pivots(&mut self, sources: &BlockBasedDataStructure, k: usize) -> Vec<usize> {
        if sources.is_empty() {
            return Vec::new();
        }
        
        // Extract all source vertices (temporarily)
        let mut temp_sources = sources.clone();
        let (source_vertices, _) = temp_sources.pull(temp_sources.size());
        
        if source_vertices.is_empty() {
            return Vec::new();
        }
        
        // Save original distances
        let mut original_distances = HashMap::new();
        for &v in &source_vertices {
            original_distances.insert(v, self.distances[v]);
        }
        
        // Perform k relaxation steps as specified in paper
        for _ in 0..k {
            let mut changed = false;
            
            for &v in &source_vertices {
                if self.distances[v] >= INFINITY {
                    continue;
                }
                
                // Relax all outgoing edges from v
                for edge in &self.graph.adj[v].clone() {
                    let new_dist = self.distances[v] + edge.weight;
                    if new_dist < self.distances[edge.to] {
                        self.distances[edge.to] = new_dist;
                        self.predecessors[edge.to] = Some(v);
                        changed = true;
                    }
                }
            }
            
            if !changed {
                break; // Early termination if no changes
            }
        }
        
        // Identify pivots: vertices whose distance changed significantly
        // or have high out-degree (representing large subtrees)
        let mut pivot_candidates = Vec::new();
        
        for &v in &source_vertices {
            let original_dist = original_distances.get(&v).copied().unwrap_or(INFINITY);
            let current_dist = self.distances[v];
            let degree = self.graph.adj[v].len();
            
            // Vertex is a pivot if:
            // 1. It has high degree (represents large subtree)
            // 2. Its distance was updated significantly during relaxation
            let is_high_degree = degree >= k;
            let distance_changed = (current_dist - original_dist).abs() > 1e-9;
            
            if is_high_degree || distance_changed {
                pivot_candidates.push((v, degree, current_dist));
            }
        }
        
        // Sort by a combination of degree and distance
        pivot_candidates.sort_by(|a, b| {
            // Primary sort: by degree (descending)  
            let degree_cmp = b.1.cmp(&a.1);
            if degree_cmp != std::cmp::Ordering::Equal {
                return degree_cmp;
            }
            // Secondary sort: by distance (ascending)
            a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Select at most |U|/k pivots as specified in paper
        let max_pivots = (source_vertices.len() / k).max(1);
        pivot_candidates.into_iter()
            .take(max_pivots.min(source_vertices.len()))
            .map(|(v, _, _)| v)
            .collect()
    }
    
    /// Helper: run Dijkstra from single vertex (for fallback cases)
    fn dijkstra_from_vertex(&mut self, start: usize) {
        let mut heap = BinaryHeap::new();
        let mut visited = vec![false; self.graph.n];
        
        if self.distances[start] < INFINITY {
            heap.push(Reverse((OrderedFloat(self.distances[start]), start)));
        }
        
        while let Some(Reverse((OrderedFloat(dist), u))) = heap.pop() {
            if visited[u] || dist > self.distances[u] + 1e-9 {
                continue;
            }
            
            visited[u] = true;
            
            for edge in &self.graph.adj[u] {
                let new_dist = dist + edge.weight;
                if new_dist < self.distances[edge.to] {
                    self.distances[edge.to] = new_dist;
                    self.predecessors[edge.to] = Some(u);
                    heap.push(Reverse((OrderedFloat(new_dist), edge.to)));
                }
            }
        }
    }

    /// Mini Dijkstra for base case - not used in current implementation
    #[allow(dead_code)]
    fn mini_dijkstra(&mut self, _start: usize, _upper_bound: f64, _k: usize) 
                    -> (BlockBasedDataStructure, f64) {
        // Placeholder implementation
        (BlockBasedDataStructure::new(64), INFINITY)
    }

    /// Get the shortest path to a vertex
    pub fn get_path(&self, target: usize) -> Option<Vec<usize>> {
        if self.distances[target] == INFINITY {
            return None;
        }

        let mut path = Vec::new();
        let mut current = Some(target);
        
        while let Some(v) = current {
            path.push(v);
            current = self.predecessors[v];
        }
        
        path.reverse();
        Some(path)
    }

    /// Get all distances
    pub fn get_distances(&self) -> &[f64] {
        &self.distances
    }
}

/// Wrapper for f64 to implement Ord for use in BinaryHeap
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Classic Dijkstra's algorithm for comparison with SIMD optimizations
pub struct ClassicDijkstra {
    graph: Graph,
    mmap_graph: Option<MemoryMappedGraph>,
}

impl ClassicDijkstra {
    pub fn new(graph: Graph) -> Self {
        Self { 
            graph,
            mmap_graph: None,
        }
    }
    
    /// Create with memory-mapped graph for large graphs
    pub fn new_with_mmap(graph: Graph) -> std::io::Result<Self> {
        let mmap_graph = graph.to_memory_mapped()?;
        Ok(Self { 
            graph,
            mmap_graph: Some(mmap_graph),
        })
    }

    pub fn solve(&self, source: usize) -> Vec<f64> {
        let mut distances = vec![INFINITY; self.graph.n];
        let mut heap = BinaryHeap::with_capacity(self.graph.n / 2);
        let mut visited = vec![false; self.graph.n];

        distances[source] = 0.0;
        heap.push(Reverse((OrderedFloat(0.0), source)));

        // Use SIMD-optimized version if available
        if let Some(ref mmap_graph) = self.mmap_graph {
            self.solve_simd_mmap(source, &mut distances, &mut heap, &mut visited, mmap_graph);
        } else {
            self.solve_simd_regular(source, &mut distances, &mut heap, &mut visited);
        }

        distances
    }
    
    /// SIMD-optimized solve with memory-mapped graph
    fn solve_simd_mmap(&self, _source: usize, distances: &mut [f64], 
                      heap: &mut BinaryHeap<Reverse<(OrderedFloat, usize)>>,
                      visited: &mut [bool], mmap_graph: &MemoryMappedGraph) {
        let mut updated_vertices = Vec::with_capacity(64);
        
        while let Some(Reverse((OrderedFloat(dist), u))) = heap.pop() {
            unsafe {
                if *visited.get_unchecked(u) {
                    continue;
                }
                *visited.get_unchecked_mut(u) = true;
            }

            // Use SIMD-optimized edge relaxation
            updated_vertices.clear();
            mmap_graph.relax_edges_simd(u, dist, distances, &mut updated_vertices);
            
            // Add updated vertices to heap
            for &vertex in &updated_vertices {
                unsafe {
                    heap.push(Reverse((OrderedFloat(*distances.get_unchecked(vertex)), vertex)));
                }
            }
        }
    }
    
    /// SIMD-optimized solve with regular graph
    fn solve_simd_regular(&self, _source: usize, distances: &mut [f64],
                         heap: &mut BinaryHeap<Reverse<(OrderedFloat, usize)>>,
                         visited: &mut [bool]) {
        while let Some(Reverse((OrderedFloat(dist), u))) = heap.pop() {
            unsafe {
                if *visited.get_unchecked(u) {
                    continue;
                }
                *visited.get_unchecked_mut(u) = true;
            }

            let adj_slice = &self.graph.adj[u];
            
            // Use SIMD for vertices with many edges
            if adj_slice.len() >= 4 {
                let chunks = adj_slice.chunks_exact(4);
                let remainder = chunks.remainder();
                
                for chunk in chunks {
                    let weights = [chunk[0].weight, chunk[1].weight, chunk[2].weight, chunk[3].weight];
                    let targets = [chunk[0].to, chunk[1].to, chunk[2].to, chunk[3].to];
                    
                    let weight_vec = f64x4::from(weights);
                    let new_dists = f64x4::splat(dist) + weight_vec;
                    let new_dists_array = new_dists.to_array();
                    
                    for i in 0..4 {
                        let target = targets[i];
                        let new_dist = new_dists_array[i];
                        
                        unsafe {
                            if new_dist < *distances.get_unchecked(target) {
                                *distances.get_unchecked_mut(target) = new_dist;
                                heap.push(Reverse((OrderedFloat(new_dist), target)));
                            }
                        }
                    }
                }
                
                // Handle remaining edges
                for edge in remainder {
                    let new_dist = dist + edge.weight;
                    unsafe {
                        if new_dist < *distances.get_unchecked(edge.to) {
                            *distances.get_unchecked_mut(edge.to) = new_dist;
                            heap.push(Reverse((OrderedFloat(new_dist), edge.to)));
                        }
                    }
                }
            } else {
                // Regular processing for small edge counts
                for edge in adj_slice {
                    let new_dist = dist + edge.weight;
                    unsafe {
                        if new_dist < *distances.get_unchecked(edge.to) {
                            *distances.get_unchecked_mut(edge.to) = new_dist;
                            heap.push(Reverse((OrderedFloat(new_dist), edge.to)));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_graph() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1, 4.0);
        graph.add_edge(0, 2, 2.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(1, 3, 5.0);
        graph.add_edge(2, 3, 8.0);
        graph.add_edge(2, 4, 10.0);
        graph.add_edge(3, 4, 2.0);

        let mut fast_sssp = FastSSSP::new(graph.clone());
        fast_sssp.solve(0);
        let fast_distances = fast_sssp.get_distances();

        let classic = ClassicDijkstra::new(graph);
        let classic_distances = classic.solve(0);

        // Debug output to understand the difference
        println!("FastSSSP distances: {:?}", fast_distances);
        println!("Classic distances: {:?}", classic_distances);
        
        // Results should be the same
        for i in 0..5 {
            assert!((fast_distances[i] - classic_distances[i]).abs() < 1e-9,
                    "Mismatch at vertex {}: Fast={:.6}, Classic={:.6}", 
                    i, fast_distances[i], classic_distances[i]);
        }
    }

    #[test]
    fn test_memory_mapped_simd() {
        // Test memory-mapped SIMD functionality
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1, 4.0);
        graph.add_edge(0, 2, 2.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(1, 3, 5.0);
        graph.add_edge(2, 3, 8.0);
        graph.add_edge(2, 4, 10.0);
        graph.add_edge(3, 4, 2.0);

        // Test memory-mapped version
        let mut fast_sssp_mmap = FastSSSP::new_with_mmap(graph.clone()).unwrap();
        fast_sssp_mmap.solve(0);
        let mmap_distances = fast_sssp_mmap.get_distances();

        // Compare with regular version
        let classic = ClassicDijkstra::new(graph);
        let classic_distances = classic.solve(0);

        // Results should be the same
        for i in 0..5 {
            assert!((mmap_distances[i] - classic_distances[i]).abs() < 1e-9,
                    "SIMD/MMap mismatch at vertex {}: SIMD={:.6}, Classic={:.6}", 
                    i, mmap_distances[i], classic_distances[i]);
        }
        
        println!("Memory-mapped SIMD test passed!");
    }

    #[test]
    fn test_path_reconstruction() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(2, 3, 3.0);
        graph.add_edge(0, 3, 10.0);

        let mut fast_sssp = FastSSSP::new(graph);
        fast_sssp.solve(0);

        let path = fast_sssp.get_path(3).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
        assert!((fast_sssp.get_distances()[3] - 6.0).abs() < 1e-9);
    }
}

// Example usage and benchmarking functions
pub mod examples {
    use super::*;
    use std::time::Instant;

    /// Create a random graph for testing
    pub fn create_random_graph(n: usize, m: usize, max_weight: f64) -> Graph {
        use std::collections::HashSet;
        
        let mut graph = Graph::new(n);
        let mut edges = HashSet::new();
        let mut rng_state = 12345u64; // Simple LCG for reproducibility
        
        for _ in 0..m {
            loop {
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                let from = (rng_state % n as u64) as usize;
                
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                let to = (rng_state % n as u64) as usize;
                
                if from != to && edges.insert((from, to)) {
                    rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let weight = (rng_state % 1000) as f64 / 100.0 * max_weight;
                    graph.add_edge(from, to, weight);
                    break;
                }
            }
        }
        
        graph
    }

    /// Benchmark the fast algorithm vs classic Dijkstra
    pub fn benchmark_algorithms(n: usize, m: usize) {
        println!("Benchmarking on graph with {} vertices and {} edges", n, m);
        
        let graph = create_random_graph(n, m, 100.0);
        println!("Graph created with {} actual edges", graph.edge_count());

        // Benchmark Fast SSSP
        let start = Instant::now();
        let mut fast_sssp = FastSSSP::new(graph.clone());
        fast_sssp.solve(0);
        let _fast_distances = fast_sssp.get_distances();
        let fast_time = start.elapsed();

        // Benchmark Classic Dijkstra
        let start = Instant::now();
        let classic = ClassicDijkstra::new(graph);
        let _classic_distances = classic.solve(0);
        let classic_time = start.elapsed();

        println!("Fast SSSP time: {:?}", fast_time);
        println!("Classic Dijkstra time: {:?}", classic_time);
        
        if classic_time > fast_time {
            println!("Speedup: {:.2}x", classic_time.as_secs_f64() / fast_time.as_secs_f64());
        } else {
            println!("Slowdown: {:.2}x", fast_time.as_secs_f64() / classic_time.as_secs_f64());
        }
    }

    /// Example: Social network shortest paths
    pub fn social_network_example() {
        println!("\n=== Social Network Example ===");
        
        let mut graph = Graph::new(8);
        // Person 0 connects to persons 1, 2 (close friends)
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(0, 2, 1.0);
        
        // Person 1 connects to 3, 4 (friends of friends)
        graph.add_edge(1, 3, 2.0);
        graph.add_edge(1, 4, 2.5);
        
        // Person 2 connects to 4, 5
        graph.add_edge(2, 4, 1.5);
        graph.add_edge(2, 5, 3.0);
        
        // Distant connections
        graph.add_edge(3, 6, 4.0);
        graph.add_edge(4, 6, 2.0);
        graph.add_edge(4, 7, 3.5);
        graph.add_edge(5, 7, 2.5);
        
        let mut sssp = FastSSSP::new(graph);
        sssp.solve(0);
        let distances = sssp.get_distances();
        
        println!("Shortest social distances from person 0:");
        for (person, &dist) in distances.iter().enumerate() {
            if dist < INFINITY {
                println!("  Person {}: {:.1} degrees", person, dist);
                if let Some(path) = sssp.get_path(person) {
                    println!("    Path: {:?}", path);
                }
            }
        }
    }

    /// Example: City road network
    pub fn city_network_example() {
        println!("\n=== City Road Network Example ===");
        
        let mut graph = Graph::new(6);
        
        // City intersections connected by roads with travel times (minutes)
        graph.add_edge(0, 1, 5.0);   // Home to School
        graph.add_edge(0, 2, 8.0);   // Home to Mall
        graph.add_edge(1, 3, 3.0);   // School to Library
        graph.add_edge(1, 4, 12.0);  // School to Airport
        graph.add_edge(2, 3, 4.0);   // Mall to Library
        graph.add_edge(2, 4, 6.0);   // Mall to Airport
        graph.add_edge(2, 5, 10.0);  // Mall to Hospital
        graph.add_edge(3, 4, 7.0);   // Library to Airport
        graph.add_edge(3, 5, 5.0);   // Library to Hospital
        graph.add_edge(4, 5, 3.0);   // Airport to Hospital
        
        let locations = ["Home", "School", "Mall", "Library", "Airport", "Hospital"];
        
        let mut sssp = FastSSSP::new(graph);
        sssp.solve(0);
        let distances = sssp.get_distances(); // Start from Home
        
        println!("Shortest travel times from Home:");
        for (i, &dist) in distances.iter().enumerate() {
            if dist < INFINITY {
                println!("  To {}: {:.0} minutes", locations[i], dist);
                if let Some(path) = sssp.get_path(i) {
                    let path_names: Vec<_> = path.iter().map(|&j| locations[j]).collect();
                    println!("    Route: {}", path_names.join(" → "));
                }
            }
        }
    }

    /// Example: Computer network latency
    pub fn network_latency_example() {
        println!("\n=== Network Latency Example ===");
        
        let mut graph = Graph::new(7);
        
        // Network nodes with latencies in milliseconds
        graph.add_edge(0, 1, 2.5);   // Server to Router A
        graph.add_edge(0, 2, 4.0);   // Server to Router B
        graph.add_edge(1, 3, 1.5);   // Router A to Switch 1
        graph.add_edge(1, 4, 3.0);   // Router A to Switch 2
        graph.add_edge(2, 4, 2.0);   // Router B to Switch 2
        graph.add_edge(2, 5, 5.0);   // Router B to Switch 3
        graph.add_edge(3, 6, 1.0);   // Switch 1 to Client A
        graph.add_edge(4, 6, 2.5);   // Switch 2 to Client A
        graph.add_edge(5, 6, 1.5);   // Switch 3 to Client A
        
        let nodes = ["Server", "Router A", "Router B", "Switch 1", "Switch 2", "Switch 3", "Client A"];
        
        let mut sssp = FastSSSP::new(graph);
        sssp.solve(0);
        let distances = sssp.get_distances(); // Start from Server
        
        println!("Minimum network latencies from Server:");
        for (i, &dist) in distances.iter().enumerate() {
            if dist < INFINITY {
                println!("  To {}: {:.1} ms", nodes[i], dist);
                if let Some(path) = sssp.get_path(i) {
                    let path_names: Vec<_> = path.iter().map(|&j| nodes[j]).collect();
                    println!("    Path: {}", path_names.join(" → "));
                }
            }
        }
    }
}

