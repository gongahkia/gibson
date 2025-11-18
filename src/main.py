# ----- REQUIRED IMPORTS -----

import json
import random
import sys
import pygame
import numpy as np
import moderngl
import glm
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
        
        # Smoothing parameters
        self.rotation_speed = 5.0
        self.zoom_speed = 3.0
        self.pan_speed = 0.1
        
    def _calculate_position(self):
        """calculate camera position from spherical coordinates"""
        rad_angle = glm.radians(self.angle)
        rad_pitch = glm.radians(self.pitch)
        
        x = self.distance * glm.cos(rad_pitch) * glm.cos(rad_angle)
        y = self.distance * glm.sin(rad_pitch)
        z = self.distance * glm.cos(rad_pitch) * glm.sin(rad_angle)
        
        return self.target + glm.vec3(x, y, z)
    
    def update(self, dt):
        """smooth interpolation towards target values"""
        # Interpolate angles
        self.angle = self._lerp(self.angle, self.target_angle, dt * self.rotation_speed)
        self.pitch = self._lerp(self.pitch, self.target_pitch, dt * self.rotation_speed)
        self.distance = self._lerp(self.distance, self.target_distance, dt * self.zoom_speed)
        
        # Clamp pitch to avoid gimbal lock
        self.pitch = glm.clamp(self.pitch, -89.0, 89.0)
        
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
        """set target zoom"""
        self.target_distance = glm.clamp(self.target_distance + delta, 5.0, 200.0)
    
    def pan(self, dx, dy):
        """pan camera target"""
        right = glm.normalize(glm.cross(self.target - self.position, self.up))
        forward = glm.normalize(glm.cross(self.up, right))
        self.target += right * dx * self.pan_speed + forward * dy * self.pan_speed


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

class MegaStructureGenerator:

    def __init__(self, size=30, layers=15):
        """
        initialize the mega structure generator
        """
        self.size = size
        self.layers = layers
        self.grid = np.full((size, size, layers), CellType.EMPTY, dtype=object)
        self.connections = []
        self.rooms = []
        self.support_map = np.zeros((size, size, layers), dtype=bool)

    def generate_mega(self):
        """
        generate a mega structure
        """
        self._create_vertical_cores()
        self._generate_floor_slabs()
        self._create_room_clusters()
        self._connect_vertical_cores()
        self._ensure_structural_integrity()
        self._add_support_pillars()
        self._add_secondary_structures()
        self._create_sky_bridges()

    def _create_vertical_cores(self):
        """
        generate vertical cores
        """
        core_spacing = random.randint(4, 6)
        for x in range(0, self.size, core_spacing):
            for z in range(0, self.size, core_spacing):
                if random.random() < 0.8:
                    height = min(random.randint(8, self.layers-2), self.layers)
                    self._build_vertical_core(x, z, height)

    def _build_vertical_core(self, x, z, height):
        """
        build a vertical core
        """
        base_width = random.randint(2, 3)
        for y in range(height):
            current_width = max(1, base_width - int(y/4))
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

    def save_structure(self, filename):
        """
        save the structure
        """
        data = {
            'grid': [[[cell.value for cell in col] for col in layer] for layer in self.grid],
            'connections': self.connections,
            'rooms': self.rooms
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_structure(self, filename):
        """
        load the structure
        """
        with open(filename, 'r') as f:
            data = json.load(f)
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
        self.init_pygame()
        self._init_font_system()
        self.debug_surface = pygame.Surface((200, 180), pygame.SRCALPHA).convert_alpha()
        self._create_shaders()
        self._create_projection_matrix()
        self._init_camera()
    
    def _init_camera(self):
        """initialize camera with proper positioning"""
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
        create vertex and fragment shaders for modern rendering with instancing
        """
        vertex_shader = """
        #version 330 core
        
        uniform mat4 view;
        uniform mat4 projection;
        
        in vec3 in_position;
        in vec3 in_offset;      // Per-instance position
        in vec3 in_color;       // Per-instance color
        
        out vec3 frag_color;
        
        void main() {
            vec3 world_pos = in_position + in_offset;
            gl_Position = projection * view * vec4(world_pos, 1.0);
            frag_color = in_color;
        }
        """
        
        fragment_shader = """
        #version 330 core
        
        in vec3 frag_color;
        out vec4 out_color;
        
        void main() {
            // Calculate simple lighting based on position
            vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
            float ambient = 0.4;
            float diffuse = 0.6;
            
            // Simple fake normal based on color variation
            vec3 normal = normalize(vec3(
                frag_color.r - 0.5,
                frag_color.g - 0.5,
                frag_color.b - 0.5
            ));
            
            float light = ambient + diffuse * max(dot(normal, light_dir), 0.0);
            vec3 lit_color = frag_color * light;
            
            out_color = vec4(lit_color, 1.0);
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
        prepare per-instance data for all cubes in the structure
        """
        colors_map = {
            CellType.EMPTY: (0.1, 0.1, 0.1),
            CellType.VERTICAL: (0.2, 0.6, 0.8),
            CellType.HORIZONTAL: (0.8, 0.4, 0.2),
            CellType.BRIDGE: (0.6, 0.8, 0.2),
            CellType.FACADE: (0.9, 0.9, 0.7),
            CellType.STAIR: (0.7, 0.3, 0.7)
        }
        
        instance_positions = []
        instance_colors = []
        
        for x in range(self.generator.size):
            for z in range(self.generator.size):
                for y in range(self.generator.layers):
                    cell_type = self.generator.grid[x][z][y]
                    if cell_type != CellType.EMPTY:
                        instance_positions.append([x, y, z])
                        instance_colors.append(colors_map.get(cell_type, (1.0, 1.0, 1.0)))
        
        self.instance_count = len(instance_positions)
        
        if self.instance_count > 0:
            positions_array = np.array(instance_positions, dtype='f4')
            colors_array = np.array(instance_colors, dtype='f4')
            
            self.instance_vbo = self.ctx.buffer(positions_array.tobytes())
            self.color_vbo = self.ctx.buffer(colors_array.tobytes())
            
            # Create VAO with instancing
            self.vao = self.ctx.vertex_array(
                self.shader_program,
                [
                    (self.vbo, '3f', 'in_position'),
                    (self.instance_vbo, '3f /i', 'in_offset'),
                    (self.color_vbo, '3f /i', 'in_color'),
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

    def init_pygame(self):
        """
        initialize pygame and moderngl context
        """
        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), DOUBLEBUF|OPENGL|pygame.FULLSCREEN)
        self.display = pygame.display.get_surface().get_size()
        
        # Initialize ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = '<'  # Less than depth test
        self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.cull_face = 'back'
        self.ctx.front_face = 'ccw'
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.clear_color = (0.1, 0.1, 0.1, 1.0)
        
        # Legacy OpenGL for UI overlay only (2D rendering)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)

    def draw_cube(self, position, cell_type):
        """
        Legacy draw_cube method - no longer used with instanced rendering
        Kept for reference, will be removed in future cleanup
        """
        pass

    def render_debug_panel(self):
        """
        render the debug panel
        """
        self.debug_surface.fill((50, 50, 50, 180))
        y_offset = 10
        legend_items = [
            ("Vertical", (0.2, 0.6, 0.8)),
            ("Horizontal", (0.8, 0.4, 0.2)),
            ("Bridge", (0.6, 0.8, 0.2)),
            ("Facade", (0.9, 0.9, 0.7)),
            ("Stair", (0.7, 0.3, 0.7))
        ]
        for text, color in legend_items:
            pygame_color = [int(c * 255) for c in color]
            text_surface = self.font.render(text, True, (255, 255, 255))
            pygame.draw.rect(self.debug_surface, pygame_color, (10, y_offset, 20, 20))
            self.debug_surface.blit(text_surface, (40, y_offset))
            y_offset += 30
        angle_text = self.font.render(f"Angle: {self.angle}°", True, (255, 255, 255))
        self.debug_surface.blit(angle_text, (10, y_offset))
        y_offset += 30
        zoom_text = self.font.render(f"Zoom: {self.zoom:.2f}x", True, (255, 255, 255))
        self.debug_surface.blit(zoom_text, (10, y_offset))

    def render(self):
        """
        render the scene using moderngl with smooth camera
        """
        # Update camera with delta time for smooth interpolation
        dt = 1.0 / 60.0  # Assuming 60 FPS target
        self.camera.update(dt)
        
        # Clear buffers
        self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        if hasattr(self, 'vao') and self.instance_count > 0:
            # Calculate view matrix from camera
            view = self._calculate_view_matrix()
            
            # Set uniforms efficiently
            self.shader_program['view'].write(glm.value_ptr(view))
            self.shader_program['projection'].write(glm.value_ptr(self.projection))
            
            # Render all instances in one draw call
            self.vao.render(instances=self.instance_count)
        
        # Legacy rendering for UI overlay
        self._render_ui_overlay()
        
        pygame.display.flip()
    
    def _render_ui_overlay(self):
        """
        render UI overlay using legacy OpenGL for 2D elements
        """
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.display[0], self.display[1], 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.render_debug_panel()
        tex_data = pygame.image.tostring(self.debug_surface, "RGBA", True)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 200, 180, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_data)
        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(self.display[0]-200, 0)
        glTexCoord2f(1, 1); glVertex2f(self.display[0], 0)
        glTexCoord2f(1, 0); glVertex2f(self.display[0], 180)
        glTexCoord2f(0, 0); glVertex2f(self.display[0]-200, 180)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([texture])
        
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

    def run(self):
        """
        run the visualizer
        """
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.angle = (self.angle - 45) % 360
                    elif event.button == 3:
                        self.angle = (self.angle + 45) % 360
                    elif event.button == 4:  
                        self.zoom = min(self.zoom * 1.1, 3.0)
                    elif event.button == 5:  
                        self.zoom = max(self.zoom / 1.1, 0.5)
            self.render()
            clock.tick(30)

# ----- EXECUTION CODE -----

if __name__ == '__main__':
    print("Gibson: generating structure...")
    generator = MegaStructureGenerator()
    generator.generate_mega()
    generator.save_structure('structure.json')
    print("Gibson: visualizing structure...")
    visualizer = IsometricVisualizer(generator)
    visualizer.run()