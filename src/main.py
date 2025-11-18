# ----- REQUIRED IMPORTS -----

import json
import random
import sys
import pygame
import numpy as np
import moderngl
import glm
from dataclasses import dataclass
from enum import Enum
from OpenGL.GL import *
from OpenGL.GLU import *
from noise import pnoise3
from pygame.locals import *

# ----- HELPER FUNCTIONS -----

class Camera:
    """
    camera class for smooth orbital camera controls
    """
    def __init__(self, target, distance, angle=45.0, pitch=30.0):
        self.target = glm.vec3(target[0], target[1], target[2])
        self.distance = distance
        self.angle = angle  # Azimuth angle in degrees
        self.pitch = pitch  # Elevation angle in degrees
        self.target_angle = angle
        self.target_pitch = pitch
        self.target_distance = distance
        self.position = self._calculate_position()
        self.up = glm.vec3(0, 1, 0)
        
        # Smoothing parameters with momentum
        self.rotation_speed = 5.0
        self.zoom_speed = 3.0
        self.pan_speed = 0.1
        
        # Velocity and acceleration for natural feel
        self.angle_velocity = 0.0
        self.pitch_velocity = 0.0
        self.zoom_velocity = 0.0
        self.damping = 0.85  # Friction/deceleration factor
        
        # Camera constraints
        self.min_pitch = -89.0
        self.max_pitch = 89.0
        self.min_distance = None  # Set dynamically
        self.max_distance = None  # Set dynamically
        
    def _calculate_position(self):
        """calculate camera position from spherical coordinates"""
        rad_angle = glm.radians(self.angle)
        rad_pitch = glm.radians(self.pitch)
        
        x = self.distance * glm.cos(rad_pitch) * glm.cos(rad_angle)
        y = self.distance * glm.sin(rad_pitch)
        z = self.distance * glm.cos(rad_pitch) * glm.sin(rad_angle)
        
        return self.target + glm.vec3(x, y, z)
    
    def update(self, dt):
        """smooth interpolation towards target values with momentum"""
        # Calculate deltas
        angle_delta = self.target_angle - self.angle
        pitch_delta = self.target_pitch - self.pitch
        zoom_delta = self.target_distance - self.distance
        
        # Apply velocities (momentum)
        self.angle_velocity += angle_delta * dt * self.rotation_speed
        self.pitch_velocity += pitch_delta * dt * self.rotation_speed
        self.zoom_velocity += zoom_delta * dt * self.zoom_speed
        
        # Apply damping (friction)
        self.angle_velocity *= self.damping
        self.pitch_velocity *= self.damping
        self.zoom_velocity *= self.damping
        
        # Update positions with velocity
        self.angle += self.angle_velocity * dt
        self.pitch += self.pitch_velocity * dt
        self.distance += self.zoom_velocity * dt
        
        # Apply constraints
        self.pitch = glm.clamp(self.pitch, self.min_pitch, self.max_pitch)
        if self.min_distance and self.max_distance:
            self.distance = glm.clamp(self.distance, self.min_distance, self.max_distance)
            self.target_distance = glm.clamp(self.target_distance, self.min_distance, self.max_distance)
        
        # Recalculate position
        self.position = self._calculate_position()
    
    def _lerp(self, current, target, t):
        """linear interpolation"""
        return current + (target - current) * min(t, 1.0)
    
    def get_view_matrix(self):
        """get view matrix for rendering"""
        return glm.lookAt(self.position, self.target, self.up)
    
    def rotate(self, delta_angle, delta_pitch=0.0):
        """set target rotation"""
        self.target_angle += delta_angle
        self.target_pitch += delta_pitch
    
    def zoom(self, delta):
        """set target zoom with improved range"""
        min_dist = max(self.target.x, self.target.y, self.target.z) * 0.3
        max_dist = max(self.target.x, self.target.y, self.target.z) * 4.0
        self.target_distance = glm.clamp(self.target_distance + delta, min_dist, max_dist)
    
    def pan(self, dx, dy):
        """pan camera target"""
        right = glm.normalize(glm.cross(self.target - self.position, self.up))
        forward = glm.normalize(glm.cross(self.up, right))
        self.target += right * dx * self.pan_speed + forward * dy * self.pan_speed
    
    def set_preset(self, preset_name):
        """set camera to preset position"""
        presets = {
            'top': (0.0, 89.0),         # Top-down view
            'front': (0.0, 0.0),        # Front view
            'side': (90.0, 0.0),        # Side view
            'perspective': (45.0, 30.0), # Default perspective
            'isometric': (45.0, 35.264), # True isometric angle
        }
        if preset_name in presets:
            angle, pitch = presets[preset_name]
            self.target_angle = angle
            self.target_pitch = pitch
            # Reset velocities for instant response
            self.angle_velocity = 0.0
            self.pitch_velocity = 0.0


class CellType(Enum):
    """
    cell types
    """
    EMPTY = 0
    VERTICAL = 1
    HORIZONTAL = 2
    BRIDGE = 3
    FACADE = 4
    STAIR = 5
    PIPE = 6
    ANTENNA = 7
    CABLE = 8
    VENT = 9
    ELEVATOR = 10

class MaterialType(Enum):
    """
    material types for rendering with PBR-like properties
    """
    CONCRETE = 0
    GLASS = 1
    METAL = 2
    NEON = 3
    RUST = 4
    STEEL = 5

class DistrictType(Enum):
    """
    district/biome types for varied architectural zones
    """
    INDUSTRIAL = 0
    RESIDENTIAL = 1
    COMMERCIAL = 2
    SLUM = 3
    ELITE = 4

# Mapping of CellTypes to their primary materials
CELL_TO_MATERIAL = {
    CellType.EMPTY: MaterialType.CONCRETE,
    CellType.VERTICAL: MaterialType.CONCRETE,
    CellType.HORIZONTAL: MaterialType.CONCRETE,
    CellType.BRIDGE: MaterialType.STEEL,
    CellType.FACADE: MaterialType.GLASS,
    CellType.STAIR: MaterialType.METAL,
    CellType.PIPE: MaterialType.RUST,
    CellType.ANTENNA: MaterialType.METAL,
    CellType.CABLE: MaterialType.STEEL,
    CellType.VENT: MaterialType.METAL,
    CellType.ELEVATOR: MaterialType.GLASS,
}

@dataclass
class Material:
    """
    material properties for PBR-inspired rendering
    """
    base_color: tuple  # RGB color (0-1 range)
    metallic: float    # 0.0 = dielectric, 1.0 = metal
    roughness: float   # 0.0 = smooth/glossy, 1.0 = rough/matte
    emission: float    # 0.0 = no glow, >0 = emissive strength
    alpha: float       # 0.0 = transparent, 1.0 = opaque

# Material definitions with properties
MATERIAL_PROPERTIES = {
    MaterialType.CONCRETE: Material(
        base_color=(0.5, 0.5, 0.6),  # Light gray-blue
        metallic=0.0,
        roughness=0.9,
        emission=0.0,
        alpha=1.0
    ),
    MaterialType.GLASS: Material(
        base_color=(0.4, 0.7, 0.9),  # Bright blue
        metallic=0.0,
        roughness=0.1,
        emission=0.0,
        alpha=0.3
    ),
    MaterialType.METAL: Material(
        base_color=(0.6, 0.6, 0.7),  # Silver
        metallic=0.8,
        roughness=0.4,
        emission=0.0,
        alpha=1.0
    ),
    MaterialType.NEON: Material(
        base_color=(0.1, 0.9, 0.9),  # Bright cyan
        metallic=0.0,
        roughness=0.2,
        emission=2.0,
        alpha=1.0
    ),
    MaterialType.RUST: Material(
        base_color=(0.8, 0.4, 0.2),  # Orange-brown
        metallic=0.5,
        roughness=0.8,
        emission=0.0,
        alpha=1.0
    ),
    MaterialType.STEEL: Material(
        base_color=(0.4, 0.5, 0.6),  # Blue-gray
        metallic=0.9,
        roughness=0.3,
        emission=0.0,
        alpha=1.0
    ),
}

# District properties defining generation parameters
DISTRICT_PROPERTIES = {
    DistrictType.INDUSTRIAL: {
        'color_palette': [(0.3, 0.3, 0.4), (0.4, 0.5, 0.5), (0.2, 0.3, 0.35)],
        'core_density': 1.2,  # More vertical cores
        'floor_thickness': 2,
        'vertical_variation': 0.3,
        'material_weights': {MaterialType.METAL: 0.4, MaterialType.STEEL: 0.3, MaterialType.CONCRETE: 0.3},
        'neon_probability': 0.1,
    },
    DistrictType.RESIDENTIAL: {
        'color_palette': [(0.6, 0.5, 0.4), (0.7, 0.6, 0.5), (0.5, 0.4, 0.3)],
        'core_density': 0.8,
        'floor_thickness': 1,
        'vertical_variation': 0.5,
        'material_weights': {MaterialType.CONCRETE: 0.5, MaterialType.GLASS: 0.3, MaterialType.RUST: 0.2},
        'neon_probability': 0.2,
    },
    DistrictType.COMMERCIAL: {
        'color_palette': [(0.2, 0.3, 0.4), (0.3, 0.4, 0.5), (0.1, 0.2, 0.3)],
        'core_density': 0.6,
        'floor_thickness': 3,
        'vertical_variation': 0.8,  # Tall towers
        'material_weights': {MaterialType.GLASS: 0.6, MaterialType.STEEL: 0.2, MaterialType.CONCRETE: 0.2},
        'neon_probability': 0.4,
    },
    DistrictType.SLUM: {
        'color_palette': [(0.4, 0.35, 0.3), (0.5, 0.4, 0.35), (0.45, 0.4, 0.35)],
        'core_density': 1.5,  # Dense, chaotic
        'floor_thickness': 1,
        'vertical_variation': 0.2,  # Low-rise
        'material_weights': {MaterialType.RUST: 0.4, MaterialType.CONCRETE: 0.4, MaterialType.METAL: 0.2},
        'neon_probability': 0.05,
    },
    DistrictType.ELITE: {
        'color_palette': [(0.8, 0.8, 0.85), (0.75, 0.75, 0.8), (0.7, 0.75, 0.8)],
        'core_density': 0.4,  # Spacious
        'floor_thickness': 3,
        'vertical_variation': 0.9,  # Very tall
        'material_weights': {MaterialType.GLASS: 0.5, MaterialType.STEEL: 0.3, MaterialType.CONCRETE: 0.2},
        'neon_probability': 0.3,
    },
}

class MegaStructureGenerator:

    def __init__(self, size=30, layers=15, seed=None):
        """
        initialize the mega structure generator with optional seed
        """
        self.size = size
        self.layers = layers
        self.seed = seed
        self.grid = np.full((size, size, layers), CellType.EMPTY, dtype=object)
        self.connections = []
        self.rooms = []
        self.support_map = np.zeros((size, size, layers), dtype=bool)
        
        # District/biome system
        self.district_map = np.zeros((size, size), dtype=int)
        self._generate_district_map()

    def _generate_district_map(self):
        """
        generate noise-based district zoning map
        """
        # Use multiple octaves of Perlin noise for organic district boundaries
        for x in range(self.size):
            for z in range(self.size):
                # Combine multiple noise frequencies
                noise_val = (
                    pnoise3(x * 0.05, z * 0.05, 0.0) * 1.0 +
                    pnoise3(x * 0.1, z * 0.1, 1.0) * 0.5 +
                    pnoise3(x * 0.2, z * 0.2, 2.0) * 0.25
                )
                
                # Map noise to district types
                if noise_val < -0.3:
                    self.district_map[x][z] = DistrictType.SLUM.value
                elif noise_val < -0.1:
                    self.district_map[x][z] = DistrictType.INDUSTRIAL.value
                elif noise_val < 0.1:
                    self.district_map[x][z] = DistrictType.RESIDENTIAL.value
                elif noise_val < 0.3:
                    self.district_map[x][z] = DistrictType.COMMERCIAL.value
                else:
                    self.district_map[x][z] = DistrictType.ELITE.value
    
    def _get_district(self, x, z):
        """
        get district type at given coordinates
        """
        if 0 <= x < self.size and 0 <= z < self.size:
            return DistrictType(self.district_map[x][z])
        return DistrictType.RESIDENTIAL  # Default

    def generate_mega(self):
        """
        generate a mega structure with optional seed for reproducibility
        """
        # Initialize random state with seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(hash(self.seed) % (2**32))
        
        self._create_vertical_cores()
        self._generate_floor_slabs()
        self._create_room_clusters()
        self._connect_vertical_cores()
        self._ensure_structural_integrity()
        self._add_support_pillars()
        self._add_secondary_structures()
        self._create_sky_bridges()
        self._add_infrastructure_details()  # Feature 7
        self._add_structural_variety()  # Feature 8

    def _create_vertical_cores(self):
        """
        generate vertical cores with district-specific density
        """
        for x in range(0, self.size):
            for z in range(0, self.size):
                district = self._get_district(x, z)
                district_props = DISTRICT_PROPERTIES[district]
                
                # Adjust core spacing based on district density
                base_probability = 0.15 * district_props['core_density']
                
                if random.random() < base_probability:
                    # Height varies by district
                    height_range = int(self.layers * district_props['vertical_variation'])
                    min_height = max(5, self.layers - height_range)
                    height = random.randint(min_height, self.layers - 2)
                    self._build_vertical_core(x, z, height, district)

    def _build_vertical_core(self, x, z, height, district=None):
        """
        build a vertical core with district-specific characteristics
        """
        if district is None:
            district = self._get_district(x, z)
        
        district_props = DISTRICT_PROPERTIES[district]
        base_width = random.randint(1, 2) if district == DistrictType.SLUM else random.randint(2, 3)
        
        for y in range(height):
            current_width = max(1, base_width - int(y/5))
            for dx in range(-current_width, current_width+1):
                for dz in range(-current_width, current_width+1):
                    nx, nz = x+dx, z+dz
                    if 0 <= nx < self.size and 0 <= nz < self.size:
                        self.grid[nx][nz][y] = CellType.VERTICAL
                        self.support_map[nx][nz][y] = True

    def _generate_floor_slabs(self):
        """
        generate floor slabs
        """
        for y in range(self.layers):
            noise_scale = 0.15
            floor_thickness = random.randint(1, 2)
            for x in range(self.size):
                for z in range(self.size):
                    if self.grid[x][z][y] == CellType.VERTICAL:
                        if random.random() < 0.7:
                            self._expand_floor(x, y, z, floor_thickness, noise_scale)

    def _expand_floor(self, x, y, z, thickness, noise_scale):
        """
        expand floor slab
        """
        queue = [(x, z)]
        visited = set()
        while queue:
            cx, cz = queue.pop(0)
            if (cx, cz) in visited:
                continue
            visited.add((cx, cz))
            if y > 0 and not self.support_map[cx][cz][y-1]:
                continue
            noise_val = pnoise3(cx*noise_scale, y*0.2, cz*noise_scale)
            if noise_val > -0.2:
                for dy in range(thickness):
                    if y+dy < self.layers:
                        self.grid[cx][cz][y+dy] = CellType.HORIZONTAL
                        self.support_map[cx][cz][y+dy] = True
                for dx, dz in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, nz = cx+dx, cz+dz
                    if 0 <= nx < self.size and 0 <= nz < self.size:
                        queue.append((nx, nz))

    def _create_room_clusters(self):
        """
        create room clusters
        """
        room_types = [
            {'size': (3,5), 'height': 2, 'type': CellType.HORIZONTAL},  
            {'size': (5,8), 'height': 3, 'type': CellType.HORIZONTAL}, 
            {'size': (2,4), 'height': 1, 'type': CellType.VERTICAL}     
        ]
        for _ in range(int(self.size**2 / 20)):
            room = random.choice(room_types)
            x = random.randint(0, self.size-1)
            z = random.randint(0, self.size-1)
            y = random.randint(0, self.layers-1)
            if self.grid[x][z][y] == CellType.HORIZONTAL:
                self._carve_room(x, y, z, room)

    def _carve_room(self, x, y, z, room):
        """
        carve a room
        """
        width, depth = random.randint(*room['size']), random.randint(*room['size'])
        height = room['height']
        for dx in range(width):
            for dz in range(depth):
                for dy in range(height):
                    nx = x + dx
                    nz = z + dz
                    ny = y + dy
                    if 0 <= nx < self.size and 0 <= nz < self.size and ny < self.layers:
                        if dy == 0:  
                            self.grid[nx][nz][ny] = CellType.HORIZONTAL
                        else:
                            if dx in [0, width-1] or dz in [0, depth-1]:
                                self.grid[nx][nz][ny] = CellType.FACADE
                        self.support_map[nx][nz][ny] = True

    def _ensure_structural_integrity(self):
        """
        ensure structural integrity
        """
        for y in range(1, self.layers):
            for x in range(self.size):
                for z in range(self.size):
                    if self.grid[x][z][y] in [CellType.HORIZONTAL, CellType.FACADE]:
                        if not self._has_support(x, y, z):
                            self.grid[x][z][y] = CellType.EMPTY

    def _has_support(self, x, y, z):
        """
        check if a cell has support
        """
        if self.support_map[x][z][y-1]:
            return True
        for dx, dz in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, nz = x+dx, z+dz
            if 0 <= nx < self.size and 0 <= nz < self.size:
                if self.grid[nx][nz][y] in [CellType.HORIZONTAL, CellType.BRIDGE]:
                    return True
        return False

    def _create_sky_bridges(self):
        """
        create sky bridges
        """
        for y in range(3, self.layers, 4):
            cores = [(x, z) for x in range(self.size) 
                    for z in range(self.size) 
                    if self.grid[x][z][y] == CellType.VERTICAL]
            if len(cores) > 1:
                start = random.choice(cores)
                end = random.choice([c for c in cores if c != start])
                self._build_bridge(start, end, y)

    def _add_secondary_structures(self):
        """
        add secondary structures
        """
        for _ in range(int(self.size * self.layers * 0.05)):  
            x = random.randint(0, self.size-1)
            z = random.randint(0, self.size-1)
            y = random.randint(1, self.layers-1)
            if self.grid[x][z][y] == CellType.EMPTY and self._has_support(x, y, z):
                structure_type = random.choice([CellType.HORIZONTAL, CellType.FACADE])
                self.grid[x][z][y] = structure_type
                for dy in range(1, random.randint(1, 3)):
                    if y + dy < self.layers and self.grid[x][z][y+dy] == CellType.EMPTY:
                        self.grid[x][z][y+dy] = structure_type

    def _add_support_pillars(self):
        """
        add support pillars
        """
        for x in range(self.size):
            for z in range(self.size):
                for y in range(1, self.layers):
                    if self.grid[x][z][y] == CellType.HORIZONTAL and not self._has_support(x, y, z):
                        for py in range(y-1, -1, -1):
                            if self.grid[x][z][py] == CellType.EMPTY:
                                self.grid[x][z][py] = CellType.VERTICAL
                                self.support_map[x][z][py] = True
                            else:
                                break  

    def _connect_vertical_cores(self):
        """
        connect vertical cores
        """
        cores = [(x, z) for x in range(self.size) for z in range(self.size) 
                if any(self.grid[x][z][y] == CellType.VERTICAL for y in range(self.layers))]
        for y in range(2, self.layers, 3):  
            for i in range(len(cores)):
                for j in range(i+1, len(cores)):
                    if random.random() < 0.3:  
                        start = cores[i]
                        end = cores[j]
                        self._build_bridge(start, end, y)
        for x, z in cores:
            for y in range(1, self.layers-1):
                if self.grid[x][z][y] == CellType.VERTICAL:
                    if random.random() < 0.2:  
                        self.grid[x][z][y] = CellType.STAIR

    def _build_bridge(self, start, end, y):
        """
        build a bridge
        """
        x1, z1 = start
        x2, z2 = end
        dx = abs(x2 - x1)
        dz = abs(z2 - z1)
        sx = 1 if x2 > x1 else -1
        sz = 1 if z2 > z1 else -1
        err = dx - dz
        while True:
            if self._is_valid_bridge_point(x1, z1, y):
                self.grid[x1][z1][y] = CellType.BRIDGE
                self.grid[x1][z1][y+1] = CellType.BRIDGE  
            if x1 == x2 and z1 == z2:
                break
            e2 = 2*err
            if e2 > -dz:
                err -= dz
                x1 += sx
            if e2 < dx:
                err += dx
                z1 += sz

    def _is_valid_bridge_point(self, x, z, y):
        """
        check if a bridge point is valid
        """
        if y > 0 and self.grid[x][z][y-1] in [CellType.VERTICAL, CellType.BRIDGE]:
            return True
        return False
    
    def _add_infrastructure_details(self):
        """
        Feature 7: Add visible infrastructure systems
        """
        self._add_pipe_networks()
        self._add_rooftop_details()
        self._add_external_elevators()
        self._add_cable_connections()
    
    def _add_pipe_networks(self):
        """
        Add pipe networks between buildings
        """
        for _ in range(int(self.size * self.layers * 0.03)):
            x, z = random.randint(0, self.size-1), random.randint(0, self.size-1)
            y = random.randint(1, self.layers-2)
            
            if self.grid[x][z][y] in [CellType.VERTICAL, CellType.HORIZONTAL]:
                # Run pipes horizontally along structures
                direction = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
                length = random.randint(3, 8)
                
                for i in range(length):
                    nx = x + direction[0] * i
                    nz = z + direction[1] * i
                    if 0 <= nx < self.size and 0 <= nz < self.size:
                        if self.grid[nx][nz][y] == CellType.EMPTY:
                            self.grid[nx][nz][y] = CellType.PIPE
    
    def _add_rooftop_details(self):
        """
        Add antennas, vents, and equipment to rooftops
        """
        for x in range(self.size):
            for z in range(self.size):
                # Find rooftop (highest non-empty cell)
                for y in range(self.layers-1, -1, -1):
                    if self.grid[x][z][y] != CellType.EMPTY:
                        # Add antenna or vent
                        if random.random() < 0.15 and y < self.layers - 1:
                            detail_type = random.choice([CellType.ANTENNA, CellType.VENT])
                            height = random.randint(1, 3)
                            for dy in range(1, height + 1):
                                if y + dy < self.layers:
                                    self.grid[x][z][y + dy] = detail_type
                        break
    
    def _add_external_elevators(self):
        """
        Add external elevator shafts on building faces
        """
        cores = [(x, z) for x in range(self.size) for z in range(self.size)
                if any(self.grid[x][z][y] == CellType.VERTICAL for y in range(self.layers))]
        
        for x, z in cores:
            if random.random() < 0.2:  # 20% of cores get elevators
                # Find a free adjacent position
                for dx, dz in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, nz = x + dx, z + dz
                    if 0 <= nx < self.size and 0 <= nz < self.size:
                        # Build elevator shaft upward
                        for y in range(self.layers):
                            if self.grid[nx][nz][y] == CellType.EMPTY:
                                if y == 0 or self.grid[nx][nz][y-1] in [CellType.ELEVATOR, CellType.HORIZONTAL, CellType.VERTICAL]:
                                    self.grid[nx][nz][y] = CellType.ELEVATOR
                        break
    
    def _add_cable_connections(self):
        """
        Add cable lines between buildings
        """
        for _ in range(int(self.size * 0.5)):
            # Find two random vertical structures
            cores = [(x, z, y) for x in range(self.size) for z in range(self.size) for y in range(self.layers)
                    if self.grid[x][z][y] == CellType.VERTICAL]
            
            if len(cores) >= 2:
                start_x, start_z, start_y = random.choice(cores)
                end_x, end_z, end_y = random.choice(cores)
                
                if abs(start_x - end_x) + abs(start_z - end_z) < 15:  # Not too far
                    # Simple line between points
                    cable_y = min(start_y, end_y) + random.randint(1, 3)
                    if cable_y < self.layers:
                        # Draw cable along path
                        steps = max(abs(end_x - start_x), abs(end_z - start_z))
                        if steps > 0:
                            for i in range(steps + 1):
                                t = i / steps
                                cx = int(start_x + (end_x - start_x) * t)
                                cz = int(start_z + (end_z - start_z) * t)
                                if 0 <= cx < self.size and 0 <= cz < self.size:
                                    if self.grid[cx][cz][cable_y] == CellType.EMPTY:
                                        self.grid[cx][cz][cable_y] = CellType.CABLE
    
    def _add_structural_variety(self):
        """
        Feature 8: Add organic chaos and variety
        """
        self._add_random_missing_sections()
        self._add_makeshift_additions()
        self._add_decay_and_overgrowth()
    
    def _add_random_missing_sections(self):
        """
        Remove random sections to simulate decay/damage
        """
        for _ in range(int(self.size * self.layers * 0.02)):
            x = random.randint(0, self.size-1)
            z = random.randint(0, self.size-1)
            y = random.randint(2, self.layers-2)
            
            # Create small gaps
            radius = random.randint(1, 2)
            for dx in range(-radius, radius+1):
                for dz in range(-radius, radius+1):
                    nx, nz = x + dx, z + dz
                    if 0 <= nx < self.size and 0 <= nz < self.size:
                        if self.grid[nx][nz][y] not in [CellType.VERTICAL, CellType.EMPTY]:
                            self.grid[nx][nz][y] = CellType.EMPTY
    
    def _add_makeshift_additions(self):
        """
        Add chaotic, shanty-style additions
        """
        slum_areas = [(x, z) for x in range(self.size) for z in range(self.size)
                     if self._get_district(x, z) == DistrictType.SLUM]
        
        for x, z in slum_areas:
            if random.random() < 0.1:
                # Find existing structure
                for y in range(self.layers):
                    if self.grid[x][z][y] in [CellType.HORIZONTAL, CellType.FACADE]:
                        # Add small chaotic extension
                        for dx, dz in [(1,0), (-1,0), (0,1), (0,-1)]:
                            nx, nz = x + dx, z + dz
                            if 0 <= nx < self.size and 0 <= nz < self.size:
                                if self.grid[nx][nz][y] == CellType.EMPTY and random.random() < 0.6:
                                    self.grid[nx][nz][y] = CellType.FACADE
                        break
    
    def _add_decay_and_overgrowth(self):
        """
        Add visual decay markers (change materials via noise)
        """
        # This will be reflected in material selection during rendering
        # For now, just mark certain cells for future material variation
        pass

    def save_structure(self, filename):
        """
        save the structure with seed for reproducibility
        """
        data = {
            'seed': self.seed,
            'size': self.size,
            'layers': self.layers,
            'grid': [[[cell.value for cell in col] for col in layer] for layer in self.grid],
            'connections': self.connections,
            'rooms': self.rooms
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_structure(self, filename):
        """
        load the structure with seed
        """
        with open(filename, 'r') as f:
            data = json.load(f)
            self.seed = data.get('seed', None)
            self.size = data.get('size', self.size)
            self.layers = data.get('layers', self.layers)
            self.grid = np.array([[[CellType(cell) for cell in col] for col in layer] 
                                for layer in data['grid']])
            self.connections = [tuple(map(tuple, c)) for c in data['connections']]
            self.rooms = data['rooms']

class IsometricVisualizer:

    def __init__(self, generator):
        """
        initialize the visualizer
        """
        self.generator = generator
        self.angle = 45  # Legacy, kept for compatibility
        self.zoom = 1.0
        self.seed = generator.seed  # Store seed for display
        self.init_pygame()
        self._init_font_system()
        self.debug_surface = pygame.Surface((200, 450), pygame.SRCALPHA).convert_alpha()
        self._create_shaders()
        self._create_projection_matrix()
        self._init_camera()
        
        # Mouse interaction state
        self.mouse_dragging = False
        self.last_mouse_pos = None
        
        # Feature 10: Interactive inspection mode
        self.selected_cell = None  # (x, y, z) tuple
        self.inspection_mode = False
        
        # UI state
        self.show_legend = True  # Show legend by default
    
    def _init_camera(self):
        """initialize camera with proper positioning and constraints"""
        center_x = self.generator.size / 2
        center_y = self.generator.layers / 2
        center_z = self.generator.size / 2
        distance = max(self.generator.size, self.generator.layers) * 1.5
        
        self.camera = Camera(
            target=(center_x, center_y, center_z),
            distance=distance,
            angle=45.0,
            pitch=30.0
        )
        
        # Set camera constraints based on scene size
        max_dim = max(self.generator.size, self.generator.layers, self.generator.size)
        self.camera.min_distance = max_dim * 0.5
        self.camera.max_distance = max_dim * 5.0
    
    def _create_projection_matrix(self):
        """
        create perspective projection matrix
        """
        aspect_ratio = self.display[0] / self.display[1]
        self.projection = glm.perspective(glm.radians(45.0), aspect_ratio, 0.1, 1000.0)
    
    def _calculate_view_matrix(self):
        """
        calculate view matrix from camera - now uses Camera class
        """
        return self.camera.get_view_matrix()
    
    def _create_shaders(self):
        """
        create vertex and fragment shaders with material support
        """
        vertex_shader = """
        #version 330 core
        
        uniform mat4 view;
        uniform mat4 projection;
        
        in vec3 in_position;
        in vec3 in_offset;         // Per-instance position
        in vec3 in_color;          // Per-instance base color
        in float in_metallic;      // Per-instance metallic value
        in float in_roughness;     // Per-instance roughness value
        in float in_emission;      // Per-instance emission strength
        in float in_alpha;         // Per-instance transparency
        
        out vec3 frag_pos;
        out vec3 frag_color;
        out float frag_metallic;
        out float frag_roughness;
        out float frag_emission;
        out float frag_alpha;
        
        void main() {
            vec3 world_pos = in_position + in_offset;
            gl_Position = projection * view * vec4(world_pos, 1.0);
            frag_pos = world_pos;
            frag_color = in_color;
            frag_metallic = in_metallic;
            frag_roughness = in_roughness;
            frag_emission = in_emission;
            frag_alpha = in_alpha;
        }
        """
        
        fragment_shader = """
        #version 330 core
        
        in vec3 frag_pos;
        in vec3 frag_color;
        in float frag_metallic;
        in float frag_roughness;
        in float frag_emission;
        in float frag_alpha;
        
        layout(location = 0) out vec4 out_color;
        layout(location = 1) out vec4 out_world_pos;  // For height-based fog
        
        void main() {
            // Calculate surface normal from derivatives for simple lighting
            vec3 normal = normalize(cross(dFdx(frag_pos), dFdy(frag_pos)));
            
            // Simple directional light for depth perception
            vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
            float light_intensity = max(dot(normal, light_dir), 0.0);
            
            // Simple ambient + diffuse lighting (no fancy PBR)
            vec3 ambient = frag_color * 0.5;  // Stronger ambient to see colors better
            vec3 diffuse = frag_color * light_intensity * 0.5;  // Diffuse light
            vec3 final_color = ambient + diffuse;
            
            // Add emission for glowing materials
            final_color += frag_color * frag_emission * 0.5;
            
            // Apply alpha/transparency
            out_color = vec4(final_color, frag_alpha);
            
            // Output world position for height-based fog
            out_world_pos = vec4(frag_pos, 1.0);
        }
        """
        
        self.shader_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        # Create cube geometry
        self._create_cube_geometry()
    
    def _create_cube_geometry(self):
        """
        create cube vertices and indices for shader rendering
        """
        # Cube vertices (position only, will expand later)
        vertices = np.array([
            # Front face
            -0.4, -0.4,  0.4,
             0.4, -0.4,  0.4,
             0.4,  0.4,  0.4,
            -0.4,  0.4,  0.4,
            # Back face
            -0.4, -0.4, -0.4,
             0.4, -0.4, -0.4,
             0.4,  0.4, -0.4,
            -0.4,  0.4, -0.4,
        ], dtype='f4')
        
        # Cube indices for triangles
        indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front
            1, 5, 6, 6, 2, 1,  # Right
            5, 4, 7, 7, 6, 5,  # Back
            4, 0, 3, 3, 7, 4,  # Left
            3, 2, 6, 6, 7, 3,  # Top
            4, 5, 1, 1, 0, 4,  # Bottom
        ], dtype='i4')
        
        self.cube_vertices = vertices
        self.cube_indices = indices
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        
        # Prepare instance data (positions and colors for all cubes)
        self._prepare_instance_data()
    
    def _prepare_instance_data(self):
        """
        prepare per-instance data with materials for all cubes in the structure
        """
        instance_positions = []
        instance_colors = []
        instance_metallic = []
        instance_roughness = []
        instance_emission = []
        instance_alpha = []
        
        for x in range(self.generator.size):
            for z in range(self.generator.size):
                for y in range(self.generator.layers):
                    cell_type = self.generator.grid[x][z][y]
                    if cell_type != CellType.EMPTY:
                        # Get material for this cell type
                        material_type = CELL_TO_MATERIAL.get(cell_type, MaterialType.CONCRETE)
                        material = MATERIAL_PROPERTIES[material_type]
                        
                        # Randomly add neon accents to some facades
                        if cell_type == CellType.FACADE and random.random() < 0.15:
                            material_type = MaterialType.NEON
                            material = MATERIAL_PROPERTIES[material_type]
                            # Randomize neon color
                            neon_colors = [
                                (0.1, 0.9, 0.9),  # Cyan
                                (0.9, 0.1, 0.9),  # Magenta
                                (0.9, 0.9, 0.1),  # Yellow
                            ]
                            material = Material(
                                base_color=random.choice(neon_colors),
                                metallic=material.metallic,
                                roughness=material.roughness,
                                emission=material.emission,
                                alpha=material.alpha
                            )
                        
                        # Add procedural color variation using 3D noise
                        noise_val = pnoise3(x * 0.1, y * 0.1, z * 0.1)
                        color_var = 0.1 * noise_val  # ±10% variation
                        varied_color = tuple(
                            max(0.0, min(1.0, c + color_var))
                            for c in material.base_color
                        )
                        
                        # Add slight roughness variation
                        roughness_var = 0.1 * pnoise3(x * 0.15, y * 0.15, z * 0.15)
                        varied_roughness = max(0.0, min(1.0, material.roughness + roughness_var))
                        
                        instance_positions.append([x, y, z])
                        instance_colors.append(varied_color)
                        instance_metallic.append(material.metallic)
                        instance_roughness.append(varied_roughness)
                        instance_emission.append(material.emission)
                        instance_alpha.append(material.alpha)
        
        self.instance_count = len(instance_positions)
        
        print(f"DEBUG: Prepared {self.instance_count} cube instances")
        
        if self.instance_count > 0:
            positions_array = np.array(instance_positions, dtype='f4')
            colors_array = np.array(instance_colors, dtype='f4')
            metallic_array = np.array(instance_metallic, dtype='f4')
            roughness_array = np.array(instance_roughness, dtype='f4')
            emission_array = np.array(instance_emission, dtype='f4')
            alpha_array = np.array(instance_alpha, dtype='f4')
            
            self.instance_vbo = self.ctx.buffer(positions_array.tobytes())
            self.color_vbo = self.ctx.buffer(colors_array.tobytes())
            self.metallic_vbo = self.ctx.buffer(metallic_array.tobytes())
            self.roughness_vbo = self.ctx.buffer(roughness_array.tobytes())
            self.emission_vbo = self.ctx.buffer(emission_array.tobytes())
            self.alpha_vbo = self.ctx.buffer(alpha_array.tobytes())
            
            # Create VAO with instancing and material data
            self.vao = self.ctx.vertex_array(
                self.shader_program,
                [
                    (self.vbo, '3f', 'in_position'),
                    (self.instance_vbo, '3f /i', 'in_offset'),
                    (self.color_vbo, '3f /i', 'in_color'),
                    (self.metallic_vbo, '1f /i', 'in_metallic'),
                    (self.roughness_vbo, '1f /i', 'in_roughness'),
                    (self.emission_vbo, '1f /i', 'in_emission'),
                    (self.alpha_vbo, '1f /i', 'in_alpha'),
                ],
                index_buffer=self.ibo
            )

    def _init_font_system(self):
        """
        initialize the font
        """
        pygame.font.init()
        try:
            self.font = pygame.font.Font(None, 24)
            test_surface = self.font.render("Test", True, (255, 255, 255))
            if test_surface.get_width() == 0:
                raise RuntimeError("Font rendering failed")
        except Exception as e:
            print(f"Font error: {e}")
            self.font = pygame.font.SysFont('Arial', 24)
    
    def _init_framebuffers(self):
        """
        initialize framebuffers for post-processing (offscreen rendering)
        """
        width, height = self.display
        
        # Main scene framebuffer - rendered scene will go here
        self.scene_color_texture = self.ctx.texture((width, height), 4, dtype='f4')  # RGBA float
        self.scene_world_pos_texture = self.ctx.texture((width, height), 4, dtype='f4')  # World position
        self.scene_depth_texture = self.ctx.depth_texture((width, height))
        self.scene_fbo = self.ctx.framebuffer(
            color_attachments=[self.scene_color_texture, self.scene_world_pos_texture],
            depth_attachment=self.scene_depth_texture
        )
        
        # Configure texture filtering
        self.scene_color_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_world_pos_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        # Bloom framebuffers (for bright pass extraction and blur)
        self.bloom_extract_texture = self.ctx.texture((width, height), 4, dtype='f4')
        self.bloom_extract_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bloom_extract_fbo = self.ctx.framebuffer(
            color_attachments=[self.bloom_extract_texture]
        )
        
        # Ping-pong framebuffers for Gaussian blur
        self.bloom_blur1_texture = self.ctx.texture((width, height), 4, dtype='f4')
        self.bloom_blur1_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bloom_blur1_fbo = self.ctx.framebuffer(
            color_attachments=[self.bloom_blur1_texture]
        )
        
        self.bloom_blur2_texture = self.ctx.texture((width, height), 4, dtype='f4')
        self.bloom_blur2_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.bloom_blur2_fbo = self.ctx.framebuffer(
            color_attachments=[self.bloom_blur2_texture]
        )
        
        print(f"Framebuffers initialized: {width}x{height} (scene + bloom)")
    
    def _init_postprocessing_quad(self):
        """
        create fullscreen quad for post-processing effects
        """
        # Fullscreen quad vertices (NDC coordinates: -1 to 1)
        quad_vertices = np.array([
            # positions  # texCoords
            -1.0,  1.0,  0.0, 1.0,
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
            
            -1.0,  1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0
        ], dtype='f4')
        
        # Simple pass-through shader for now (will add effects later)
        quad_vertex_shader = """
        #version 330 core
        
        in vec2 in_position;
        in vec2 in_texcoord;
        
        out vec2 uv;
        
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            uv = in_texcoord;
        }
        """
        
        quad_fragment_shader = """
        #version 330 core
        
        in vec2 uv;
        out vec4 fragColor;
        
        uniform sampler2D scene_texture;
        uniform sampler2D depth_texture;
        uniform sampler2D world_pos_texture;
        uniform sampler2D bloom_texture;
        uniform float fog_density;
        uniform vec3 fog_color;
        uniform float fog_height_falloff;
        uniform float bloom_intensity;
        uniform bool enable_postprocessing;
        
        // Convert depth buffer value to linear depth
        float linearizeDepth(float depth, float near, float far) {
            float z = depth * 2.0 - 1.0; // Back to NDC
            return (2.0 * near * far) / (far + near - z * (far - near));
        }
        
        void main() {
            vec3 color = texture(scene_texture, uv).rgb;
            
            // Early exit if post-processing disabled
            if (!enable_postprocessing) {
                fragColor = vec4(color, 1.0);
                return;
            }
            
            float depth = texture(depth_texture, uv).r;
            vec3 world_pos = texture(world_pos_texture, uv).rgb;
            vec3 bloom = texture(bloom_texture, uv).rgb;
            
            // Add bloom (additive blending)
            color += bloom * bloom_intensity;
            
            // Linearize depth (assuming near=0.1, far=500.0 from projection)
            float linear_depth = linearizeDepth(depth, 0.1, 500.0);
            
            // Height-based fog density modifier
            // Lower heights get denser fog (cyberpunk street-level smog)
            float height_factor = exp(-world_pos.y * fog_height_falloff);
            height_factor = clamp(height_factor, 0.0, 1.0);
            
            // Combined fog density (distance + height)
            float combined_density = fog_density * (1.0 + height_factor * 2.0);
            
            // Calculate fog factor (exponential fog)
            float fog_factor = exp(-combined_density * linear_depth * 0.01);
            fog_factor = clamp(fog_factor, 0.0, 1.0);
            
            // Mix scene color with fog
            vec3 final_color = mix(fog_color, color, fog_factor);
            
            fragColor = vec4(final_color, 1.0);
        }
        """
        
        self.quad_program = self.ctx.program(
            vertex_shader=quad_vertex_shader,
            fragment_shader=quad_fragment_shader
        )
        
        self.quad_vbo = self.ctx.buffer(quad_vertices.tobytes())
        self.quad_vao = self.ctx.vertex_array(
            self.quad_program,
            [(self.quad_vbo, '2f 2f', 'in_position', 'in_texcoord')]
        )
        
        # Set default fog parameters
        self.fog_density = 0.0  # Disabled by default
        self.fog_color = (0.1, 0.1, 0.12)  # Dark blue-gray
        self.fog_height_falloff = 0.05  # Lower values = more ground-level fog
        
        # Bloom parameters
        self.bloom_threshold = 1.5  # Higher threshold - only very bright things bloom
        self.bloom_intensity = 0.2  # Reduced bloom strength
        self.blur_iterations = 2  # Fewer blur passes
        
        # Post-processing toggle
        self.enable_postprocessing = False  # Disabled by default for clearer colors
        
        print("Post-processing quad initialized")
    
    def _init_bloom_shaders(self):
        """
        create shader programs for bloom effects
        """
        # Bloom extraction shader (extract bright pixels)
        bloom_extract_vertex = """
        #version 330 core
        
        in vec2 in_position;
        in vec2 in_texcoord;
        
        out vec2 uv;
        
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            uv = in_texcoord;
        }
        """
        
        bloom_extract_fragment = """
        #version 330 core
        
        in vec2 uv;
        out vec4 fragColor;
        
        uniform sampler2D scene_texture;
        uniform float bloom_threshold;
        
        void main() {
            vec3 color = texture(scene_texture, uv).rgb;
            
            // Calculate luminance
            float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
            
            // Extract bright pixels above threshold
            if (luminance > bloom_threshold) {
                fragColor = vec4(color, 1.0);
            } else {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
        """
        
        self.bloom_extract_program = self.ctx.program(
            vertex_shader=bloom_extract_vertex,
            fragment_shader=bloom_extract_fragment
        )
        
        # Gaussian blur shader (separable - horizontal and vertical passes)
        blur_vertex = """
        #version 330 core
        
        in vec2 in_position;
        in vec2 in_texcoord;
        
        out vec2 uv;
        
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            uv = in_texcoord;
        }
        """
        
        blur_fragment = """
        #version 330 core
        
        in vec2 uv;
        out vec4 fragColor;
        
        uniform sampler2D input_texture;
        uniform vec2 blur_direction;  // (1,0) for horizontal, (0,1) for vertical
        uniform vec2 texture_size;
        
        // 9-tap Gaussian blur kernel
        const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
        
        void main() {
            vec2 tex_offset = 1.0 / texture_size * blur_direction;
            vec3 result = texture(input_texture, uv).rgb * weights[0];
            
            for(int i = 1; i < 5; i++) {
                result += texture(input_texture, uv + tex_offset * float(i)).rgb * weights[i];
                result += texture(input_texture, uv - tex_offset * float(i)).rgb * weights[i];
            }
            
            fragColor = vec4(result, 1.0);
        }
        """
        
        self.blur_program = self.ctx.program(
            vertex_shader=blur_vertex,
            fragment_shader=blur_fragment
        )
        
        # UI overlay shader (simple texture rendering with alpha blending)
        ui_vertex = """
        #version 330 core
        
        in vec2 in_position;
        in vec2 in_texcoord;
        
        out vec2 uv;
        
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            uv = in_texcoord;
        }
        """
        
        ui_fragment = """
        #version 330 core
        
        in vec2 uv;
        out vec4 fragColor;
        
        uniform sampler2D ui_texture;
        
        void main() {
            fragColor = texture(ui_texture, uv);
        }
        """
        
        self.ui_overlay_program = self.ctx.program(
            vertex_shader=ui_vertex,
            fragment_shader=ui_fragment
        )
        
        print("Bloom shaders initialized")

    def init_pygame(self):
        """
        initialize pygame and moderngl context
        """
        pygame.init()
        
        # Request OpenGL 3.3 Core Profile
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
        
        self.screen = pygame.display.set_mode((0, 0), DOUBLEBUF|OPENGL|pygame.FULLSCREEN)
        
        # Set window title with seed
        seed_display = self.seed if self.seed else "NO-SEED"
        pygame.display.set_caption(f"Gibson v2.0 - Seed: {seed_display}")
        
        self.display = pygame.display.get_surface().get_size()
        
        # Initialize ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.depth_func = '<'  # Less than depth test
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = 'back'
        self.ctx.front_face = 'ccw'
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.clear_color = (0.05, 0.05, 0.08, 1.0)  # Darker background
        
        # Initialize framebuffers for post-processing
        self._init_framebuffers()
        
        # Initialize fullscreen quad for post-processing
        self._init_postprocessing_quad()
        
        # Initialize bloom shader programs
        self._init_bloom_shaders()

    def draw_cube(self, position, cell_type):
        """
        Legacy draw_cube method - no longer used with instanced rendering
        Kept for reference, will be removed in future cleanup
        """
        pass
    
    def _ray_cast_from_mouse(self, mouse_pos):
        """
        Feature 10: Cast ray from mouse position into 3D world
        Returns (x, y, z) of clicked cell or None
        """
        # Convert mouse to normalized device coordinates
        x_ndc = (2.0 * mouse_pos[0]) / self.display[0] - 1.0
        y_ndc = 1.0 - (2.0 * mouse_pos[1]) / self.display[1]
        
        # Ray in clip space
        ray_clip = glm.vec4(x_ndc, y_ndc, -1.0, 1.0)
        
        # Ray in eye space
        ray_eye = glm.inverse(self.projection) * ray_clip
        ray_eye = glm.vec4(ray_eye.x, ray_eye.y, -1.0, 0.0)
        
        # Ray in world space
        view = self.camera.get_view_matrix()
        ray_world = glm.inverse(view) * ray_eye
        ray_dir = glm.normalize(glm.vec3(ray_world.x, ray_world.y, ray_world.z))
        
        # Ray origin is camera position
        ray_origin = self.camera.position
        
        # March along ray and test voxel grid
        return self._march_ray(ray_origin, ray_dir)
    
    def _march_ray(self, origin, direction, max_distance=100.0):
        """
        March along ray to find first intersection with voxel grid
        """
        step_size = 0.5
        current_pos = glm.vec3(origin)
        
        for _ in range(int(max_distance / step_size)):
            current_pos += direction * step_size
            
            # Convert to grid coordinates
            x = int(round(current_pos.x))
            y = int(round(current_pos.y))
            z = int(round(current_pos.z))
            
            # Check if in bounds
            if (0 <= x < self.generator.size and 
                0 <= y < self.generator.layers and 
                0 <= z < self.generator.size):
                
                # Check if cell is occupied
                if self.generator.grid[x][z][y] != CellType.EMPTY:
                    return (x, y, z)
        
        return None
    
    def _get_cell_info(self, pos):
        """
        Get detailed information about a cell
        """
        if pos is None:
            return None
        
        x, y, z = pos
        if not (0 <= x < self.generator.size and 
                0 <= y < self.generator.layers and 
                0 <= z < self.generator.size):
            return None
        
        cell_type = self.generator.grid[x][z][y]
        district = self.generator._get_district(x, z)
        material = CELL_TO_MATERIAL.get(cell_type, MaterialType.CONCRETE)
        
        # Count connected cells
        connected = 0
        for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
            nx, ny, nz = x+dx, y+dy, z+dz
            if (0 <= nx < self.generator.size and 
                0 <= ny < self.generator.layers and 
                0 <= nz < self.generator.size):
                if self.generator.grid[nx][nz][ny] != CellType.EMPTY:
                    connected += 1
        
        return {
            'position': (x, y, z),
            'cell_type': cell_type,
            'material': material,
            'district': district,
            'connected_cells': connected,
            'has_support': self.generator._has_support(x, y, z) if y > 0 else True
        }

    def render_debug_panel(self):
        """
        render the debug panel
        """
        self.debug_surface.fill((50, 50, 50, 180))
        y_offset = 10
        legend_items = [
            ("MATERIALS", None),
            ("Concrete", (0.6, 0.6, 0.65)),
            ("Glass", (0.7, 0.85, 0.95)),
            ("Metal", (0.7, 0.7, 0.75)),
            ("Neon", (0.1, 0.9, 0.9)),
            ("Steel", (0.5, 0.5, 0.6)),
            ("", None),
            ("CAMERA", None),
            ("1-5: Views", (0.8, 0.8, 0.8)),
            ("", None),
            ("SEED", None),
            ("R: Regen", (0.8, 0.8, 0.8)),
            ("S: Save Seed", (0.8, 0.8, 0.8)),
            ("", None),
            ("POST-FX", None),
            ("P: Toggle", (0.8, 0.8, 0.8)),
            ("[/]: Fog", (0.8, 0.8, 0.8)),
            ("-/=: Bloom", (0.8, 0.8, 0.8)),
            ("", None),
            ("INSPECT", None),
            ("I: Toggle", (0.8, 0.8, 0.8)),
            ("Click: Select", (0.8, 0.8, 0.8)),
        ]
        for text, color in legend_items:
            if color is None:
                # Header text
                text_surface = self.font.render(text, True, (255, 255, 128))
                self.debug_surface.blit(text_surface, (10, y_offset))
            else:
                pygame_color = [int(c * 255) for c in color]
                text_surface = self.font.render(text, True, (255, 255, 255))
                if text and not text.startswith(("1-", "R:", "S:", "P:", "[", "-")):
                    pygame.draw.rect(self.debug_surface, pygame_color, (10, y_offset, 20, 20))
                    self.debug_surface.blit(text_surface, (40, y_offset))
                else:
                    self.debug_surface.blit(text_surface, (10, y_offset))
            y_offset += 25
        angle_text = self.font.render(f"Angle: {self.angle}°", True, (255, 255, 255))
        self.debug_surface.blit(angle_text, (10, y_offset))
        y_offset += 30
        zoom_text = self.font.render(f"Zoom: {self.zoom:.2f}x", True, (255, 255, 255))
        self.debug_surface.blit(zoom_text, (10, y_offset))
        y_offset += 30
        
        # Display seed
        if self.seed:
            seed_text = self.font.render(f"Seed: {self.seed}", True, (255, 255, 0))
            self.debug_surface.blit(seed_text, (10, y_offset))
            y_offset += 30
        
        # Feature 10: Display inspection info
        if self.inspection_mode and self.selected_cell:
            info = self._get_cell_info(self.selected_cell)
            if info:
                y_offset += 10
                header = self.font.render("INSPECT", True, (255, 255, 128))
                self.debug_surface.blit(header, (10, y_offset))
                y_offset += 25
                
                pos_text = self.font.render(f"Pos: {info['position']}", True, (200, 200, 200))
                self.debug_surface.blit(pos_text, (10, y_offset))
                y_offset += 20
                
                type_text = self.font.render(f"Type: {info['cell_type'].name}", True, (200, 200, 200))
                self.debug_surface.blit(type_text, (10, y_offset))
                y_offset += 20
                
                dist_text = self.font.render(f"Zone: {info['district'].name}", True, (200, 200, 200))
                self.debug_surface.blit(dist_text, (10, y_offset))
    
    def _take_screenshot(self):
        """
        take a screenshot and save it with timestamp
        """
        import datetime
        import os
        
        # Create screenshots directory if it doesn't exist
        screenshot_dir = "../screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        
        # Generate filename with timestamp and seed
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{screenshot_dir}/gibson_{self.seed}_{timestamp}.png"
        
        # Read the framebuffer
        width, height = self.display
        pixels = self.ctx.screen.read(components=3)  # RGB
        
        # Convert to pygame surface
        image = pygame.image.fromstring(pixels, (width, height), 'RGB')
        
        # Flip vertically (OpenGL to pygame coordinate conversion)
        image = pygame.transform.flip(image, False, True)
        
        # Save the image
        pygame.image.save(image, filename)
        print(f"Screenshot saved: {filename}")

                y_offset += 20
                
                conn_text = self.font.render(f"Links: {info['connected_cells']}", True, (200, 200, 200))
                self.debug_surface.blit(conn_text, (10, y_offset))

    def render(self):
        """
        render the scene using moderngl with smooth camera
        """
        # Update camera with delta time for smooth interpolation
        dt = 1.0 / 60.0  # Assuming 60 FPS target
        self.camera.update(dt)
        
        # PASS 1: Render scene to offscreen framebuffer
        self.scene_fbo.use()
        self.ctx.viewport = (0, 0, *self.display)
        self.ctx.clear(0.05, 0.05, 0.08, 1.0)  # Clear color
        self.scene_fbo.clear(depth=1.0)  # Explicitly clear depth buffer
        
        if hasattr(self, 'vao') and self.instance_count > 0:
            # Calculate view matrix from camera
            view = self._calculate_view_matrix()
            
            # Debug: Print once
            if not hasattr(self, '_debug_printed'):
                print(f"DEBUG: Rendering {self.instance_count} instances")
                print(f"DEBUG: Camera position: {self.camera.position}")
                print(f"DEBUG: Camera target: {self.camera.target}")
                self._debug_printed = True
            
            # Set uniforms efficiently (convert PyGLM matrices to bytes for moderngl)
            self.shader_program['view'].write(view.to_bytes())
            self.shader_program['projection'].write(self.projection.to_bytes())
            
            # Render all instances in one draw call
            self.vao.render(instances=self.instance_count)
        
        # PASS 2: Bloom extraction (extract bright pixels)
        if self.enable_postprocessing:
            self.bloom_extract_fbo.use()
            self.ctx.viewport = (0, 0, *self.display)
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            
            self.scene_color_texture.use(location=0)
            self.bloom_extract_program['scene_texture'].value = 0
            self.bloom_extract_program['bloom_threshold'].value = self.bloom_threshold
            
            self.quad_vao.render()
            
            # PASS 3: Gaussian blur (separable two-pass)
            width, height = self.display
            
            for iteration in range(self.blur_iterations):
                # Horizontal blur pass
                self.bloom_blur1_fbo.use()
                self.ctx.viewport = (0, 0, *self.display)
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                
                if iteration == 0:
                    self.bloom_extract_texture.use(location=0)
                else:
                    self.bloom_blur2_texture.use(location=0)
                
                self.blur_program['input_texture'].value = 0
                self.blur_program['blur_direction'].value = (1.0, 0.0)
                self.blur_program['texture_size'].value = (width, height)
                
                self.quad_vao.render()
                
                # Vertical blur pass
                self.bloom_blur2_fbo.use()
                self.ctx.viewport = (0, 0, *self.display)
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                
                self.bloom_blur1_texture.use(location=0)
                self.blur_program['input_texture'].value = 0
                self.blur_program['blur_direction'].value = (0.0, 1.0)
                self.blur_program['texture_size'].value = (width, height)
                
                self.quad_vao.render()
        
        # PASS 4: Final composite (scene + bloom + fog)
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, *self.display)
        self.ctx.clear(0.05, 0.05, 0.08, 1.0)
        
        # Debug once
        if not hasattr(self, '_composite_debug'):
            print(f"DEBUG: Composite pass - postprocessing={self.enable_postprocessing}, fog={self.fog_density}")
            self._composite_debug = True
        
        # Bind all textures
        self.scene_color_texture.use(location=0)
        self.scene_depth_texture.use(location=1)
        self.scene_world_pos_texture.use(location=2)
        self.bloom_blur2_texture.use(location=3)
        
        self.quad_program['scene_texture'].value = 0
        self.quad_program['depth_texture'].value = 1
        self.quad_program['world_pos_texture'].value = 2
        self.quad_program['bloom_texture'].value = 3
        
        # Set fog uniforms
        self.quad_program['fog_density'].value = self.fog_density
        self.quad_program['fog_color'].value = self.fog_color
        self.quad_program['fog_height_falloff'].value = self.fog_height_falloff
        
        # Set bloom uniforms
        self.quad_program['bloom_intensity'].value = self.bloom_intensity
        self.quad_program['enable_postprocessing'].value = self.enable_postprocessing
        
        # Render fullscreen quad
        self.quad_vao.render()
        
        # Legacy rendering for UI overlay
        self._render_ui_overlay()
        
        pygame.display.flip()
    
    def _render_ui_overlay(self):
        """
        render UI overlay with seed info and quit button using pygame overlay
        """
        # Create semi-transparent overlay surface
        overlay = pygame.Surface(self.display, pygame.SRCALPHA)
        
        # Render seed info at top-left
        seed_text = self.font.render(f"Seed: {self.seed}", True, (255, 255, 255, 220))
        seed_bg = pygame.Surface((seed_text.get_width() + 20, seed_text.get_height() + 10), pygame.SRCALPHA)
        seed_bg.fill((0, 0, 0, 180))
        overlay.blit(seed_bg, (10, 10))
        overlay.blit(seed_text, (20, 15))
        
        # Render controls at top-left
        controls = [
            "Controls:",
            "Mouse Drag: Rotate | Wheel: Zoom",
            "WASD: Pan | 1-5: Camera Presets",
            "R: Regenerate | S: Screenshot",
            "F/G: Fog -/+ | P: Bloom | L: Legend",
            "ESC/Q: Quit"
        ]
        y_offset = 50
        for line in controls:
            text = self.font.render(line, True, (200, 200, 200, 200))
            text_bg = pygame.Surface((text.get_width() + 20, text.get_height() + 5), pygame.SRCALPHA)
            text_bg.fill((0, 0, 0, 160))
            overlay.blit(text_bg, (10, y_offset))
            overlay.blit(text, (20, y_offset + 2))
            y_offset += 25
        
        # Render quit button at bottom-right
        quit_text = self.font.render("[Q] QUIT", True, (255, 100, 100, 240))
        quit_bg = pygame.Surface((quit_text.get_width() + 30, quit_text.get_height() + 15), pygame.SRCALPHA)
        quit_bg.fill((40, 0, 0, 200))
        quit_x = self.display[0] - quit_bg.get_width() - 20
        quit_y = self.display[1] - quit_bg.get_height() - 20
        overlay.blit(quit_bg, (quit_x, quit_y))
        overlay.blit(quit_text, (quit_x + 15, quit_y + 7))
        
        # Render material legend at bottom-left (if enabled)
        if hasattr(self, 'show_legend') and self.show_legend:
            legend_items = [
                ("Material Legend:", None),
                ("Concrete", (0.5, 0.5, 0.6)),
                ("Glass", (0.4, 0.7, 0.9)),
                ("Metal", (0.6, 0.6, 0.7)),
                ("Rust", (0.8, 0.4, 0.2)),
                ("Steel", (0.4, 0.5, 0.6)),
                ("Neon", (0.1, 0.9, 0.9)),
            ]
            
            legend_y = self.display[1] - 220
            for item_name, color in legend_items:
                if color is None:
                    # Title
                    text = self.font.render(item_name, True, (255, 255, 255, 240))
                    text_bg = pygame.Surface((text.get_width() + 20, text.get_height() + 5), pygame.SRCALPHA)
                    text_bg.fill((0, 0, 0, 180))
                else:
                    # Color swatch + label
                    text = self.font.render(item_name, True, (200, 200, 200, 220))
                    text_bg = pygame.Surface((text.get_width() + 45, text.get_height() + 5), pygame.SRCALPHA)
                    text_bg.fill((0, 0, 0, 160))
                    # Draw color swatch
                    swatch = pygame.Surface((20, 20), pygame.SRCALPHA)
                    swatch.fill(tuple(int(c * 255) for c in color) + (255,))
                    overlay.blit(swatch, (15, legend_y + 2))
                    overlay.blit(text_bg, (10, legend_y))
                    overlay.blit(text, (40, legend_y + 2))
                    legend_y += 25
                    continue
                    
                overlay.blit(text_bg, (10, legend_y))
                overlay.blit(text, (20, legend_y + 2))
                legend_y += 30
        
        # Convert pygame surface to OpenGL texture and render
        texture_data = pygame.image.tostring(overlay, 'RGBA', True)
        if not hasattr(self, 'ui_texture'):
            self.ui_texture = self.ctx.texture(self.display, 4, texture_data)
            self.ui_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            self.ui_texture.write(texture_data)
        
        # Disable depth test for UI overlay
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ui_texture.use(location=0)
        self.ui_overlay_program['ui_texture'].value = 0
        self.quad_vao.render()
        self.ctx.enable(moderngl.DEPTH_TEST)

    def run(self):
        """
        run the visualizer with smooth camera controls, mouse drag, and keyboard panning
        """
        clock = pygame.time.Clock()
        while True:
            # Handle keyboard panning (WASD keys)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.camera.pan(0, 1)
            if keys[pygame.K_s]:
                self.camera.pan(0, -1)
            if keys[pygame.K_a]:
                self.camera.pan(-1, 0)
            if keys[pygame.K_d]:
                self.camera.pan(1, 0)
            
            # Camera preset hotkeys (1-5)
            if keys[pygame.K_1]:
                self.camera.set_preset('top')
            if keys[pygame.K_2]:
                self.camera.set_preset('front')
            if keys[pygame.K_3]:
                self.camera.set_preset('side')
            if keys[pygame.K_4]:
                self.camera.set_preset('perspective')
            if keys[pygame.K_5]:
                self.camera.set_preset('isometric')
            
            # Press 'R' to regenerate with new seed
            if keys[pygame.K_r]:
                print("Regenerating with new seed...")
                new_seed = generate_seed()
                print(f"New seed: {new_seed}")
                # Regenerate structure
                self.generator = MegaStructureGenerator(seed=new_seed)
                self.generator.generate_mega()
                self.generator.save_structure('structure.json')
                self.seed = new_seed
                # Update window title
                pygame.display.set_caption(f"Gibson v2.0 - Seed: {new_seed}")
                # Recreate rendering data
                self._prepare_instance_data()
                # Wait for key release to avoid multiple triggers
                pygame.time.wait(500)
            
            # Press 'S' to save current seed to clipboard
            if keys[pygame.K_s]:
                print(f"Seed saved: {self.seed}")
                # Write seed to a file for easy sharing
                with open('current_seed.txt', 'w') as f:
                    f.write(self.seed)
                print("Seed written to current_seed.txt")
                pygame.time.wait(500)
            
            # Press 'P' to toggle post-processing effects
            if keys[pygame.K_p]:
                self.enable_postprocessing = not self.enable_postprocessing
                status = "enabled" if self.enable_postprocessing else "disabled"
                print(f"Post-processing {status}")
                pygame.time.wait(300)
            
            # Adjust fog density with '[' and ']'
            if keys[pygame.K_LEFTBRACKET]:
                self.fog_density = max(0.0, self.fog_density - 0.05)
                print(f"Fog density: {self.fog_density:.2f}")
            if keys[pygame.K_RIGHTBRACKET]:
                self.fog_density = min(2.0, self.fog_density + 0.05)
                print(f"Fog density: {self.fog_density:.2f}")
            
            # Adjust bloom intensity with '-' and '='
            if keys[pygame.K_MINUS]:
                self.bloom_intensity = max(0.0, self.bloom_intensity - 0.1)
                print(f"Bloom intensity: {self.bloom_intensity:.2f}")
            if keys[pygame.K_EQUALS]:
                self.bloom_intensity = min(2.0, self.bloom_intensity + 0.1)
                print(f"Bloom intensity: {self.bloom_intensity:.2f}")
            
            # Press 'I' to toggle inspection mode (Feature 10)
            if keys[pygame.K_i]:
                self.inspection_mode = not self.inspection_mode
                status = "enabled" if self.inspection_mode else "disabled"
                print(f"Inspection mode {status}")
                if not self.inspection_mode:
                    self.selected_cell = None
                pygame.time.wait(300)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_s:
                        # Take screenshot
                        self._take_screenshot()
                    elif event.key == pygame.K_l:
                        # Toggle legend
                        self.show_legend = not self.show_legend
                        status = "shown" if self.show_legend else "hidden"
                        print(f"Legend {status}")
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # LMB
                        if self.inspection_mode:
                            # Ray cast to select cell
                            mouse_pos = pygame.mouse.get_pos()
                            self.selected_cell = self._ray_cast_from_mouse(mouse_pos)
                            if self.selected_cell:
                                print(f"Selected cell: {self.selected_cell}")
                        else:
                            # Normal camera drag
                            self.mouse_dragging = True
                            self.last_mouse_pos = pygame.mouse.get_pos()
                            self.drag_start_time = pygame.time.get_ticks()
                    elif event.button == 3:  # RMB - smooth rotate right
                        self.camera.rotate(45.0)
                    elif event.button == 4:  # Mouse wheel up - zoom in smoothly
                        self.camera.zoom(-3.0)
                    elif event.button == 5:  # Mouse wheel down - zoom out smoothly
                        self.camera.zoom(3.0)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        # If click was quick (no drag), rotate left
                        if self.mouse_dragging and pygame.time.get_ticks() - self.drag_start_time < 200:
                            if self.last_mouse_pos == pygame.mouse.get_pos():
                                self.camera.rotate(-45.0)
                        self.mouse_dragging = False
                        self.last_mouse_pos = None
                elif event.type == pygame.MOUSEMOTION:
                    if self.mouse_dragging and self.last_mouse_pos:
                        # Free rotation via mouse drag
                        current_pos = pygame.mouse.get_pos()
                        dx = current_pos[0] - self.last_mouse_pos[0]
                        dy = current_pos[1] - self.last_mouse_pos[1]
                        
                        # Rotate camera based on drag distance
                        self.camera.rotate(-dx * 0.3, -dy * 0.3)
                        self.last_mouse_pos = current_pos
            
            self.render()
            clock.tick(60)

# ----- EXECUTION CODE -----

def generate_seed():
    """generate a random 8-character alphanumeric seed"""
    import string
    import time
    chars = string.ascii_uppercase + string.digits
    # Use timestamp for uniqueness
    random.seed(time.time())
    return ''.join(random.choice(chars) for _ in range(8))

def validate_seed(seed):
    """validate seed format (8-character alphanumeric)"""
    import string
    if not seed or len(seed) != 8:
        return False
    valid_chars = set(string.ascii_uppercase + string.digits)
    return all(c in valid_chars for c in seed.upper())

if __name__ == '__main__':
    # Check for command-line seed argument
    import sys
    if len(sys.argv) > 1:
        seed = sys.argv[1].upper()
        if not validate_seed(seed):
            print(f"Error: Invalid seed '{sys.argv[1]}'. Seed must be 8 alphanumeric characters.")
            sys.exit(1)
    else:
        seed = generate_seed()
    
    print(f"Gibson: generating structure with seed: {seed}")
    generator = MegaStructureGenerator(seed=seed)
    generator.generate_mega()
    generator.save_structure('structure.json')
    print("Gibson: visualizing structure...")
    visualizer = IsometricVisualizer(generator)
    visualizer.run()