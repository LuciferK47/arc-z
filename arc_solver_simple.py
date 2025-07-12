#!/usr/bin/env python3
"""
Simple ARC-AGI Challenge Solver
===============================

This script implements a simplified approach to solving ARC-AGI tasks
using only Python standard library modules.
"""

import json
import copy
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any, Optional

class SimpleARCSolver:
    def __init__(self):
        self.transformations = [
            self.tile_pattern,
            self.mirror_pattern,
            self.rotate_pattern,
            self.color_mapping,
            self.pattern_fill,
            self.size_transformation,
            self.symmetry_completion,
            self.direct_copy
        ]
    
    def load_data(self, filepath: str) -> Dict[str, Any]:
        """Load ARC challenge data from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_grid_dimensions(self, grid: List[List[int]]) -> Tuple[int, int]:
        """Get dimensions of a grid."""
        return len(grid), len(grid[0]) if grid else 0
    
    def grids_equal(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if two grids are equal."""
        if len(grid1) != len(grid2):
            return False
        for i in range(len(grid1)):
            if len(grid1[i]) != len(grid2[i]):
                return False
            for j in range(len(grid1[i])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        return True
    
    def mirror_horizontal(self, grid: List[List[int]]) -> List[List[int]]:
        """Mirror grid horizontally."""
        return [row[::-1] for row in grid]
    
    def mirror_vertical(self, grid: List[List[int]]) -> List[List[int]]:
        """Mirror grid vertically."""
        return grid[::-1]
    
    def rotate_90(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid 90 degrees clockwise."""
        rows, cols = len(grid), len(grid[0])
        rotated = [[0] * rows for _ in range(cols)]
        for i in range(rows):
            for j in range(cols):
                rotated[j][rows - 1 - i] = grid[i][j]
        return rotated
    
    def rotate_180(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid 180 degrees."""
        return [[grid[i][j] for j in range(len(grid[i]) - 1, -1, -1)] 
                for i in range(len(grid) - 1, -1, -1)]
    
    def rotate_270(self, grid: List[List[int]]) -> List[List[int]]:
        """Rotate grid 270 degrees clockwise."""
        return self.rotate_90(self.rotate_180(grid))
    
    def tile_grid(self, grid: List[List[int]], tiles_v: int, tiles_h: int) -> List[List[int]]:
        """Tile a grid multiple times."""
        rows, cols = len(grid), len(grid[0])
        result = []
        
        for tile_row in range(tiles_v):
            for grid_row in range(rows):
                new_row = []
                for tile_col in range(tiles_h):
                    new_row.extend(grid[grid_row])
                result.append(new_row)
        
        return result
    
    def find_color_mapping(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[int, int]:
        """Find color mapping between input and output grids."""
        if len(input_grid) != len(output_grid):
            return {}
        
        mapping = {}
        for i in range(len(input_grid)):
            if len(input_grid[i]) != len(output_grid[i]):
                return {}
            for j in range(len(input_grid[i])):
                in_color = input_grid[i][j]
                out_color = output_grid[i][j]
                
                if in_color in mapping:
                    if mapping[in_color] != out_color:
                        return {}  # Inconsistent mapping
                else:
                    mapping[in_color] = out_color
        
        return mapping
    
    def apply_color_mapping(self, grid: List[List[int]], mapping: Dict[int, int]) -> List[List[int]]:
        """Apply color mapping to a grid."""
        result = []
        for row in grid:
            new_row = []
            for cell in row:
                new_row.append(mapping.get(cell, cell))
            result.append(new_row)
        return result
    
    def tile_pattern(self, input_grid: List[List[int]], task_data: Dict[str, Any]) -> List[List[List[int]]]:
        """Apply tiling transformation."""
        solutions = []
        
        for example in task_data['train']:
            train_input = example['input']
            train_output = example['output']
            
            h_in, w_in = self.get_grid_dimensions(train_input)
            h_out, w_out = self.get_grid_dimensions(train_output)
            
            # Check if output is a multiple of input
            if h_out % h_in == 0 and w_out % w_in == 0:
                tiles_v = h_out // h_in
                tiles_h = w_out // w_in
                
                # Apply the same tiling pattern to input
                tiled = self.tile_grid(input_grid, tiles_v, tiles_h)
                solutions.append(tiled)
                
                # Try alternating pattern
                if tiles_v > 1 or tiles_h > 1:
                    alt_pattern = []
                    for tile_row in range(tiles_v):
                        for grid_row in range(h_in):
                            new_row = []
                            for tile_col in range(tiles_h):
                                if (tile_row + tile_col) % 2 == 0:
                                    new_row.extend(input_grid[grid_row])
                                else:
                                    new_row.extend(input_grid[grid_row][::-1])
                            alt_pattern.append(new_row)
                    solutions.append(alt_pattern)
        
        return solutions
    
    def mirror_pattern(self, input_grid: List[List[int]], task_data: Dict[str, Any]) -> List[List[List[int]]]:
        """Apply mirror transformation."""
        solutions = []
        
        # Try different mirror transformations
        solutions.append(self.mirror_horizontal(input_grid))
        solutions.append(self.mirror_vertical(input_grid))
        
        # Try combining original with mirrored
        h_mirrored = [row + row[::-1] for row in input_grid]
        v_mirrored = input_grid + input_grid[::-1]
        
        solutions.extend([h_mirrored, v_mirrored])
        
        return solutions
    
    def rotate_pattern(self, input_grid: List[List[int]], task_data: Dict[str, Any]) -> List[List[List[int]]]:
        """Apply rotation transformation."""
        solutions = []
        
        solutions.append(self.rotate_90(input_grid))
        solutions.append(self.rotate_180(input_grid))
        solutions.append(self.rotate_270(input_grid))
        
        return solutions
    
    def color_mapping(self, input_grid: List[List[int]], task_data: Dict[str, Any]) -> List[List[List[int]]]:
        """Apply color mapping transformation."""
        solutions = []
        
        for example in task_data['train']:
            train_input = example['input']
            train_output = example['output']
            
            mapping = self.find_color_mapping(train_input, train_output)
            if mapping:
                mapped = self.apply_color_mapping(input_grid, mapping)
                solutions.append(mapped)
        
        return solutions
    
    def pattern_fill(self, input_grid: List[List[int]], task_data: Dict[str, Any]) -> List[List[List[int]]]:
        """Fill empty spaces with patterns."""
        solutions = []
        
        result = copy.deepcopy(input_grid)
        rows, cols = len(result), len(result[0])
        
        # Fill zeros with most common non-zero neighbor
        for i in range(rows):
            for j in range(cols):
                if result[i][j] == 0:
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < rows and 0 <= nj < cols and result[ni][nj] != 0:
                                neighbors.append(result[ni][nj])
                    
                    if neighbors:
                        most_common = Counter(neighbors).most_common(1)[0][0]
                        result[i][j] = most_common
        
        solutions.append(result)
        return solutions
    
    def size_transformation(self, input_grid: List[List[int]], task_data: Dict[str, Any]) -> List[List[List[int]]]:
        """Handle size transformations."""
        solutions = []
        
        rows, cols = len(input_grid), len(input_grid[0])
        
        # Try extracting center
        if rows > 2 and cols > 2:
            center_row, center_col = rows // 2, cols // 2
            center_2x2 = [
                [input_grid[center_row-1][center_col-1], input_grid[center_row-1][center_col]],
                [input_grid[center_row][center_col-1], input_grid[center_row][center_col]]
            ]
            solutions.append(center_2x2)
        
        # Try extracting corners
        if rows >= 2 and cols >= 2:
            mid_row, mid_col = rows // 2, cols // 2
            
            corners = [
                [row[:mid_col] for row in input_grid[:mid_row]],  # Top-left
                [row[mid_col:] for row in input_grid[:mid_row]],  # Top-right
                [row[:mid_col] for row in input_grid[mid_row:]],  # Bottom-left
                [row[mid_col:] for row in input_grid[mid_row:]]   # Bottom-right
            ]
            
            solutions.extend([corner for corner in corners if corner and corner[0]])
        
        return solutions
    
    def symmetry_completion(self, input_grid: List[List[int]], task_data: Dict[str, Any]) -> List[List[List[int]]]:
        """Complete symmetry patterns."""
        solutions = []
        
        rows, cols = len(input_grid), len(input_grid[0])
        
        # Try horizontal symmetry completion
        if cols % 2 == 1:
            mid = cols // 2
            left_half = [row[:mid] for row in input_grid]
            right_half = [row[mid+1:] for row in input_grid]
            
            # Complete from left half
            completed = []
            for i, row in enumerate(input_grid):
                new_row = row[:mid] + [row[mid]] + row[:mid][::-1]
                completed.append(new_row)
            solutions.append(completed)
        
        # Try vertical symmetry completion
        if rows % 2 == 1:
            mid = rows // 2
            top_half = input_grid[:mid]
            
            # Complete from top half
            completed = top_half + [input_grid[mid]] + top_half[::-1]
            solutions.append(completed)
        
        return solutions
    
    def direct_copy(self, input_grid: List[List[int]], task_data: Dict[str, Any]) -> List[List[List[int]]]:
        """Direct copy as fallback."""
        return [copy.deepcopy(input_grid)]
    
    def solve_task(self, task_data: Dict[str, Any]) -> List[Dict[str, List[List[int]]]]:
        """Solve a single ARC task."""
        solutions = []
        
        for test_input in task_data['test']:
            input_grid = test_input['input']
            
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
                # Convert to tuple for hashing
                sol_tuple = tuple(tuple(row) for row in sol)
                if sol_tuple not in seen:
                    seen.add(sol_tuple)
                    unique_solutions.append(sol)
            
            # Take first 2 unique solutions
            if len(unique_solutions) >= 2:
                attempt_1 = unique_solutions[0]
                attempt_2 = unique_solutions[1]
            elif len(unique_solutions) == 1:
                attempt_1 = unique_solutions[0]
                attempt_2 = unique_solutions[0]  # Same as first
            else:
                # Fallback: return input as-is
                attempt_1 = input_grid
                attempt_2 = input_grid
            
            solutions.append({
                'attempt_1': attempt_1,
                'attempt_2': attempt_2
            })
        
        return solutions
    
    def solve_all_tasks(self, test_data: Dict[str, Any]) -> Dict[str, List[Dict[str, List[List[int]]]]]:
        """Solve all tasks in the test data."""
        results = {}
        
        for i, (task_id, task_data) in enumerate(test_data.items()):
            print(f"Solving task {i+1}/{len(test_data)}: {task_id}")
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
    solver = SimpleARCSolver()
    
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