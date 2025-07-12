#!/usr/bin/env python3
"""
ARC-AGI Challenge Solver
========================

This script implements a comprehensive approach to solving ARC-AGI tasks
by analyzing patterns in the training examples and applying them to test inputs.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional
import itertools

class ARCTaskSolver:
    def __init__(self):
        self.transformations = [
            self.tile_pattern,
            self.mirror_pattern,
            self.rotate_pattern,
            self.color_mapping,
            self.geometric_transform,
            self.symmetry_completion,
            self.pattern_fill,
            self.object_detection,
            self.rule_extraction,
            self.size_transformation
        ]
    
    def load_data(self, filepath: str) -> Dict[str, Any]:
        """Load ARC challenge data from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def grid_to_numpy(self, grid: List[List[int]]) -> np.ndarray:
        """Convert grid to numpy array."""
        return np.array(grid)
    
    def numpy_to_grid(self, arr: np.ndarray) -> List[List[int]]:
        """Convert numpy array to grid."""
        return arr.tolist()
    
    def analyze_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single task to extract patterns."""
        analysis = {
            'train_examples': len(task_data['train']),
            'test_inputs': len(task_data['test']),
            'input_shapes': [],
            'output_shapes': [],
            'color_patterns': [],
            'transformation_type': None
        }
        
        for example in task_data['train']:
            input_grid = self.grid_to_numpy(example['input'])
            output_grid = self.grid_to_numpy(example['output'])
            
            analysis['input_shapes'].append(input_grid.shape)
            analysis['output_shapes'].append(output_grid.shape)
            analysis['color_patterns'].append(self.analyze_colors(input_grid, output_grid))
        
        analysis['transformation_type'] = self.identify_transformation_type(task_data['train'])
        return analysis
    
    def analyze_colors(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze color patterns between input and output."""
        return {
            'input_colors': set(input_grid.flatten()),
            'output_colors': set(output_grid.flatten()),
            'color_mapping': self.find_color_mapping(input_grid, output_grid),
            'new_colors': set(output_grid.flatten()) - set(input_grid.flatten())
        }
    
    def find_color_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[int, int]:
        """Find direct color mappings between input and output."""
        mapping = {}
        if input_grid.shape == output_grid.shape:
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    in_color = input_grid[i, j]
                    out_color = output_grid[i, j]
                    if in_color in mapping and mapping[in_color] != out_color:
                        # Inconsistent mapping
                        return {}
                    mapping[in_color] = out_color
        return mapping
    
    def identify_transformation_type(self, train_examples: List[Dict[str, Any]]) -> str:
        """Identify the type of transformation based on training examples."""
        if not train_examples:
            return "unknown"
        
        # Check for tiling patterns
        for example in train_examples:
            input_grid = self.grid_to_numpy(example['input'])
            output_grid = self.grid_to_numpy(example['output'])
            
            if self.is_tiling_pattern(input_grid, output_grid):
                return "tiling"
            elif self.is_mirror_pattern(input_grid, output_grid):
                return "mirror"
            elif self.is_rotation_pattern(input_grid, output_grid):
                return "rotation"
            elif self.is_color_mapping_pattern(input_grid, output_grid):
                return "color_mapping"
        
        return "complex"
    
    def is_tiling_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if the transformation is a tiling pattern."""
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape
        
        # Check if output is a multiple of input
        if h_out % h_in == 0 and w_out % w_in == 0:
            tiles_v = h_out // h_in
            tiles_h = w_out // w_in
            
            # Check if the pattern repeats
            for i in range(tiles_v):
                for j in range(tiles_h):
                    start_row = i * h_in
                    start_col = j * w_in
                    tile = output_grid[start_row:start_row+h_in, start_col:start_col+w_in]
                    
                    # Check if tile matches input or its variations
                    if not (np.array_equal(tile, input_grid) or 
                           np.array_equal(tile, np.fliplr(input_grid)) or
                           np.array_equal(tile, np.flipud(input_grid))):
                        return False
            return True
        return False
    
    def is_mirror_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if the transformation is a mirror pattern."""
        return (np.array_equal(output_grid, np.fliplr(input_grid)) or
                np.array_equal(output_grid, np.flipud(input_grid)))
    
    def is_rotation_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if the transformation is a rotation pattern."""
        for k in range(1, 4):
            if np.array_equal(output_grid, np.rot90(input_grid, k)):
                return True
        return False
    
    def is_color_mapping_pattern(self, input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
        """Check if the transformation is a simple color mapping."""
        if input_grid.shape != output_grid.shape:
            return False
        
        mapping = self.find_color_mapping(input_grid, output_grid)
        return len(mapping) > 0
    
    def tile_pattern(self, input_grid: np.ndarray, task_data: Dict[str, Any]) -> List[np.ndarray]:
        """Apply tiling transformation."""
        solutions = []
        
        for example in task_data['train']:
            train_input = self.grid_to_numpy(example['input'])
            train_output = self.grid_to_numpy(example['output'])
            
            if self.is_tiling_pattern(train_input, train_output):
                h_in, w_in = train_input.shape
                h_out, w_out = train_output.shape
                
                tiles_v = h_out // h_in
                tiles_h = w_out // w_in
                
                # Apply the same tiling pattern to input
                tiled = np.tile(input_grid, (tiles_v, tiles_h))
                
                # Check for alternating patterns
                alt_pattern = np.zeros((h_out, w_out), dtype=int)
                for i in range(tiles_v):
                    for j in range(tiles_h):
                        start_row = i * h_in
                        start_col = j * w_in
                        
                        if (i + j) % 2 == 0:
                            alt_pattern[start_row:start_row+h_in, start_col:start_col+w_in] = input_grid
                        else:
                            alt_pattern[start_row:start_row+h_in, start_col:start_col+w_in] = np.fliplr(input_grid)
                
                solutions.extend([tiled, alt_pattern])
        
        return solutions
    
    def mirror_pattern(self, input_grid: np.ndarray, task_data: Dict[str, Any]) -> List[np.ndarray]:
        """Apply mirror transformation."""
        solutions = []
        
        # Try horizontal and vertical mirroring
        solutions.append(np.fliplr(input_grid))
        solutions.append(np.flipud(input_grid))
        
        # Try combining with original
        h, w = input_grid.shape
        combined_h = np.hstack([input_grid, np.fliplr(input_grid)])
        combined_v = np.vstack([input_grid, np.flipud(input_grid)])
        
        solutions.extend([combined_h, combined_v])
        
        return solutions
    
    def rotate_pattern(self, input_grid: np.ndarray, task_data: Dict[str, Any]) -> List[np.ndarray]:
        """Apply rotation transformation."""
        solutions = []
        
        for k in range(1, 4):
            solutions.append(np.rot90(input_grid, k))
        
        return solutions
    
    def color_mapping(self, input_grid: np.ndarray, task_data: Dict[str, Any]) -> List[np.ndarray]:
        """Apply color mapping transformation."""
        solutions = []
        
        for example in task_data['train']:
            train_input = self.grid_to_numpy(example['input'])
            train_output = self.grid_to_numpy(example['output'])
            
            mapping = self.find_color_mapping(train_input, train_output)
            if mapping:
                # Apply the mapping to the input
                mapped = input_grid.copy()
                for old_color, new_color in mapping.items():
                    mapped[input_grid == old_color] = new_color
                solutions.append(mapped)
        
        return solutions
    
    def geometric_transform(self, input_grid: np.ndarray, task_data: Dict[str, Any]) -> List[np.ndarray]:
        """Apply geometric transformations."""
        solutions = []
        
        # Try scaling
        for scale in [2, 3]:
            scaled = np.repeat(np.repeat(input_grid, scale, axis=0), scale, axis=1)
            solutions.append(scaled)
        
        return solutions
    
    def symmetry_completion(self, input_grid: np.ndarray, task_data: Dict[str, Any]) -> List[np.ndarray]:
        """Complete symmetry patterns."""
        solutions = []
        
        # Try to complete horizontal symmetry
        h, w = input_grid.shape
        if w % 2 == 1:  # Odd width, can have vertical axis of symmetry
            mid = w // 2
            left_half = input_grid[:, :mid]
            right_half = input_grid[:, mid+1:]
            
            # Complete from left
            completed_left = np.hstack([left_half, input_grid[:, mid:mid+1], np.fliplr(left_half)])
            solutions.append(completed_left)
            
            # Complete from right
            completed_right = np.hstack([np.fliplr(right_half), input_grid[:, mid:mid+1], right_half])
            solutions.append(completed_right)
        
        return solutions
    
    def pattern_fill(self, input_grid: np.ndarray, task_data: Dict[str, Any]) -> List[np.ndarray]:
        """Fill patterns based on surrounding context."""
        solutions = []
        
        # Fill zeros with most common non-zero neighbor
        result = input_grid.copy()
        h, w = result.shape
        
        for i in range(h):
            for j in range(w):
                if result[i, j] == 0:
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and result[ni, nj] != 0:
                                neighbors.append(result[ni, nj])
                    
                    if neighbors:
                        most_common = Counter(neighbors).most_common(1)[0][0]
                        result[i, j] = most_common
        
        solutions.append(result)
        return solutions
    
    def object_detection(self, input_grid: np.ndarray, task_data: Dict[str, Any]) -> List[np.ndarray]:
        """Detect and manipulate objects in the grid."""
        solutions = []
        
        # Find connected components
        objects = self.find_objects(input_grid)
        
        # Try moving objects
        for obj_coords in objects:
            moved_grid = input_grid.copy()
            # Move object down by 1
            for (i, j) in obj_coords:
                moved_grid[i, j] = 0
            for (i, j) in obj_coords:
                if i + 1 < input_grid.shape[0]:
                    moved_grid[i + 1, j] = input_grid[i, j]
            solutions.append(moved_grid)
        
        return solutions
    
    def find_objects(self, grid: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Find connected components (objects) in the grid."""
        visited = np.zeros_like(grid, dtype=bool)
        objects = []
        
        def dfs(i, j, color, obj_coords):
            if (i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1] or 
                visited[i, j] or grid[i, j] != color):
                return
            
            visited[i, j] = True
            obj_coords.append((i, j))
            
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                dfs(i + di, j + dj, color, obj_coords)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] != 0:
                    obj_coords = []
                    dfs(i, j, grid[i, j], obj_coords)
                    if obj_coords:
                        objects.append(obj_coords)
        
        return objects
    
    def rule_extraction(self, input_grid: np.ndarray, task_data: Dict[str, Any]) -> List[np.ndarray]:
        """Extract and apply rules from training examples."""
        solutions = []
        
        # Rule: If input has pattern A, output has pattern B
        # This is a simplified rule extraction
        for example in task_data['train']:
            train_input = self.grid_to_numpy(example['input'])
            train_output = self.grid_to_numpy(example['output'])
            
            # Look for simple position-based rules
            if input_grid.shape == train_input.shape:
                # Check if we can apply the same transformation
                diff = train_output - train_input
                applied = input_grid + diff
                
                # Ensure values are in valid range
                applied = np.clip(applied, 0, 9)
                solutions.append(applied)
        
        return solutions
    
    def size_transformation(self, input_grid: np.ndarray, task_data: Dict[str, Any]) -> List[np.ndarray]:
        """Handle size transformations."""
        solutions = []
        
        # Try cropping to different sizes
        h, w = input_grid.shape
        
        # Crop to center
        if h > 2 and w > 2:
            center_h, center_w = h // 2, w // 2
            cropped = input_grid[center_h-1:center_h+1, center_w-1:center_w+1]
            solutions.append(cropped)
        
        # Try extracting corners
        if h >= 3 and w >= 3:
            corners = [
                input_grid[:h//2, :w//2],  # Top-left
                input_grid[:h//2, w//2:],  # Top-right
                input_grid[h//2:, :w//2],  # Bottom-left
                input_grid[h//2:, w//2:]   # Bottom-right
            ]
            solutions.extend(corners)
        
        return solutions
    
    def solve_task(self, task_data: Dict[str, Any]) -> List[Dict[str, List[List[int]]]]:
        """Solve a single ARC task."""
        solutions = []
        
        for test_input in task_data['test']:
            input_grid = self.grid_to_numpy(test_input['input'])
            
            # Generate solutions using all transformation methods
            all_solutions = []
            
            for transform in self.transformations:
                try:
                    transform_solutions = transform(input_grid, task_data)
                    all_solutions.extend(transform_solutions)
                except Exception as e:
                    # Skip failed transformations
                    continue
            
            # Remove duplicates and select best 2 solutions
            unique_solutions = []
            seen = set()
            
            for sol in all_solutions:
                sol_tuple = tuple(sol.flatten())
                if sol_tuple not in seen:
                    seen.add(sol_tuple)
                    unique_solutions.append(sol)
            
            # Take first 2 unique solutions
            if len(unique_solutions) >= 2:
                attempt_1 = self.numpy_to_grid(unique_solutions[0])
                attempt_2 = self.numpy_to_grid(unique_solutions[1])
            elif len(unique_solutions) == 1:
                attempt_1 = self.numpy_to_grid(unique_solutions[0])
                attempt_2 = self.numpy_to_grid(unique_solutions[0])  # Same as first
            else:
                # Fallback: return input as-is
                attempt_1 = test_input['input']
                attempt_2 = test_input['input']
            
            solutions.append({
                'attempt_1': attempt_1,
                'attempt_2': attempt_2
            })
        
        return solutions
    
    def solve_all_tasks(self, test_data: Dict[str, Any]) -> Dict[str, List[Dict[str, List[List[int]]]]]:
        """Solve all tasks in the test data."""
        results = {}
        
        for task_id, task_data in test_data.items():
            print(f"Solving task {task_id}...")
            results[task_id] = self.solve_task(task_data)
        
        return results
    
    def save_submission(self, results: Dict[str, List[Dict[str, List[List[int]]]]], 
                       filename: str = 'submission.json'):
        """Save results in the required submission format."""
        with open(filename, 'w') as f:
            json.dump(results, f, separators=(',', ':'))
        print(f"Submission saved to {filename}")

def main():
    # Initialize solver
    solver = ARCTaskSolver()
    
    # Load test data
    print("Loading test data...")
    test_data = solver.load_data('arc-agi_test_challenges.json')
    
    print(f"Found {len(test_data)} tasks to solve")
    
    # Solve all tasks
    results = solver.solve_all_tasks(test_data)
    
    # Save submission
    solver.save_submission(results)
    
    print("ARC-AGI solving complete!")

if __name__ == "__main__":
    main()