from typing import List

import numpy as np
import pyvisgraph as vg # type: ignore


class VisibilityPathFinder:
    """
    Description:
        Generate the reference path via the visibility graph and A* algorithm.
    Attrs:
        env: The environment object of solving the visibility graph.
    Funcs:
        __prepare: Prepare the visibility graph including preprocess the map.
        get_ref_path: Get the (shortest) refenence path.
    """
    def __init__(self, boundary_coords, obstacle_list, verbose=False):
        self.boundary_coords = boundary_coords
        self.obstacle_list = obstacle_list
        self.vb = verbose
        self.__prepare()

    def __prepare(self):
        vg_obstacle_list = [
            [vg.Point(x, y) for (x, y) in pt]
            for pt in self.obstacle_list
        ]
        self.env = vg.VisGraph()
        self.env.build(vg_obstacle_list)

    def update_env(self):
        pass

    def get_ref_path(self, start_pos:tuple, end_pos:tuple) -> List[tuple]:
        """
        Description:
            Generate the initially guessed path based on obstacles and boundaries specified during preparation.
        Args:
            start_pos: The x,y coordinates.
            end_pos: - The x,y coordinates.
        Returns:
            path: List of coordinates of the inital path
            dist: The total distance of the path
        """
        if self.vb:
            print(f'[{self.__class__.__name__}] Reference path generated.')

        try:
            shortest_path = self.env.shortest_path(
                vg.Point(start_pos[0], start_pos[1]),
                vg.Point(end_pos[0], end_pos[1])
            )
            shortest_path_np = np.array([(p.x, p.y) for p in shortest_path])
            shortest_dist = np.sum(np.linalg.norm(shortest_path_np[1:] - shortest_path_np[:-1], axis=1))
        except Exception as e:
            print(f'[{self.__class__.__name__}] With start {start_pos} and goal {end_pos}.')
            print(f'[{self.__class__.__name__}] With boundary {self.boundary_coords}.')
            raise e
        return [(p.x, p.y) for p in shortest_path], shortest_dist

    
    