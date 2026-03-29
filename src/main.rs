use macroquad::models::{draw_cube_wires, draw_mesh, Mesh, Vertex};
use macroquad::prelude::*;
use serde::Serialize;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

const GRID_SIZE: usize = 30;
const GRID_LAYERS: usize = 15;
const CHUNK_SIZE_X: usize = 8;
const CHUNK_SIZE_Z: usize = 8;
const CHUNK_SIZE_Y: usize = 4;
const CAMERA_FOV_DEGREES: f32 = 45.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[repr(u8)]
enum CellType {
    Empty = 0,
    Vertical = 1,
    Horizontal = 2,
    Bridge = 3,
    Facade = 4,
    Stair = 5,
    Pipe = 6,
    Antenna = 7,
    Cable = 8,
    Vent = 9,
    Elevator = 10,
}

impl CellType {
    fn name(self) -> &'static str {
        match self {
            Self::Empty => "EMPTY",
            Self::Vertical => "VERTICAL",
            Self::Horizontal => "HORIZONTAL",
            Self::Bridge => "BRIDGE",
            Self::Facade => "FACADE",
            Self::Stair => "STAIR",
            Self::Pipe => "PIPE",
            Self::Antenna => "ANTENNA",
            Self::Cable => "CABLE",
            Self::Vent => "VENT",
            Self::Elevator => "ELEVATOR",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum MaterialType {
    Concrete = 0,
    Glass = 1,
    Metal = 2,
    Neon = 3,
    Rust = 4,
    Steel = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[repr(u8)]
enum DistrictType {
    Industrial = 0,
    Residential = 1,
    Commercial = 2,
    Slum = 3,
    Elite = 4,
}

impl DistrictType {
    fn name(self) -> &'static str {
        match self {
            Self::Industrial => "INDUSTRIAL",
            Self::Residential => "RESIDENTIAL",
            Self::Commercial => "COMMERCIAL",
            Self::Slum => "SLUM",
            Self::Elite => "ELITE",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum BiomeStratum {
    Underground = 0,
    Surface = 1,
    Midrise = 2,
    Skyline = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum WFCTile {
    Empty = 0,
    FloorSolid = 1,
    FloorHalfN = 2,
    FloorHalfE = 3,
    WallN = 4,
    WallE = 5,
    WallCornerNE = 6,
    WallCornerNW = 7,
    CorridorNS = 8,
    CorridorEW = 9,
    RoomCenter = 10,
    DoorN = 11,
    DoorE = 12,
    Stairwell = 13,
    ElevatorShaft = 14,
}

const WFC_TILE_COUNT: usize = 15;

#[derive(Clone, Copy)]
struct MaterialStyle {
    base_color: (f32, f32, f32),
    alpha: f32,
}

#[derive(Clone, Copy)]
struct DistrictProps {
    color_palette: [(f32, f32, f32); 3],
    core_density: f32,
    floor_thickness: i32,
    vertical_variation: f32,
    neon_probability: f32,
    pipe_probability: f32,
    elevator_probability: f32,
}

#[derive(Clone, Copy)]
struct BiomeParams {
    y_min: usize,
    y_max: usize,
    rust_mult: f32,
}

const MATERIALS: [MaterialStyle; 6] = [
    MaterialStyle {
        base_color: (0.50, 0.50, 0.60),
        alpha: 1.0,
    },
    MaterialStyle {
        base_color: (0.40, 0.70, 0.90),
        alpha: 0.42,
    },
    MaterialStyle {
        base_color: (0.60, 0.60, 0.70),
        alpha: 1.0,
    },
    MaterialStyle {
        base_color: (0.08, 0.92, 0.96),
        alpha: 1.0,
    },
    MaterialStyle {
        base_color: (0.80, 0.40, 0.20),
        alpha: 1.0,
    },
    MaterialStyle {
        base_color: (0.40, 0.50, 0.60),
        alpha: 1.0,
    },
];

const DISTRICTS: [DistrictProps; 5] = [
    DistrictProps {
        color_palette: [(0.30, 0.30, 0.40), (0.40, 0.50, 0.50), (0.20, 0.30, 0.35)],
        core_density: 1.2,
        floor_thickness: 2,
        vertical_variation: 0.30,
        neon_probability: 0.10,
        pipe_probability: 0.40,
        elevator_probability: 0.10,
    },
    DistrictProps {
        color_palette: [(0.60, 0.50, 0.40), (0.70, 0.60, 0.50), (0.50, 0.40, 0.30)],
        core_density: 0.8,
        floor_thickness: 1,
        vertical_variation: 0.50,
        neon_probability: 0.20,
        pipe_probability: 0.20,
        elevator_probability: 0.12,
    },
    DistrictProps {
        color_palette: [(0.20, 0.30, 0.40), (0.30, 0.40, 0.50), (0.10, 0.20, 0.30)],
        core_density: 0.6,
        floor_thickness: 3,
        vertical_variation: 0.80,
        neon_probability: 0.40,
        pipe_probability: 0.18,
        elevator_probability: 0.20,
    },
    DistrictProps {
        color_palette: [(0.40, 0.35, 0.30), (0.50, 0.40, 0.35), (0.45, 0.40, 0.35)],
        core_density: 1.5,
        floor_thickness: 1,
        vertical_variation: 0.20,
        neon_probability: 0.05,
        pipe_probability: 0.50,
        elevator_probability: 0.08,
    },
    DistrictProps {
        color_palette: [(0.80, 0.80, 0.85), (0.75, 0.75, 0.80), (0.70, 0.75, 0.80)],
        core_density: 0.4,
        floor_thickness: 3,
        vertical_variation: 0.90,
        neon_probability: 0.30,
        pipe_probability: 0.12,
        elevator_probability: 0.26,
    },
];

const BIOME_TABLE: [BiomeParams; 4] = [
    BiomeParams {
        y_min: 0,
        y_max: 2,
        rust_mult: 1.5,
    },
    BiomeParams {
        y_min: 3,
        y_max: 6,
        rust_mult: 1.0,
    },
    BiomeParams {
        y_min: 7,
        y_max: 11,
        rust_mult: 0.8,
    },
    BiomeParams {
        y_min: 12,
        y_max: 14,
        rust_mult: 0.5,
    },
];

#[derive(Clone, Copy)]
struct WfcCell {
    possible: u16,
    collapsed_tile: Option<usize>,
    entropy: f32,
}

#[derive(Clone, Serialize)]
struct ConnectionRecord {
    kind: String,
    start: [usize; 3],
    end: [usize; 3],
}

#[derive(Clone, Serialize)]
struct RoomRecord {
    position: [usize; 3],
    district: String,
    label: String,
}

#[derive(Serialize)]
struct SavedStructure {
    seed: String,
    size: usize,
    layers: usize,
    grid: Vec<Vec<Vec<u8>>>,
    connections: Vec<ConnectionRecord>,
    rooms: Vec<RoomRecord>,
}

struct SpatialChunk {
    mesh: Mesh,
    center: Vec3,
}

struct RenderWorld {
    opaque_chunks: Vec<SpatialChunk>,
    translucent_chunks: Vec<SpatialChunk>,
}

#[derive(Clone, Copy)]
struct Rng32 {
    state: u64,
}

impl Rng32 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    fn range_usize(&mut self, min_inclusive: usize, max_inclusive: usize) -> usize {
        if max_inclusive <= min_inclusive {
            return min_inclusive;
        }
        let span = (max_inclusive - min_inclusive + 1) as u32;
        min_inclusive + (self.next_u32() % span) as usize
    }

    fn choose_index(&mut self, len: usize) -> usize {
        self.range_usize(0, len.saturating_sub(1))
    }
}

fn seed_hash(seed: &str) -> u64 {
    let mut h = 0u64;
    for byte in seed.bytes() {
        h = h.wrapping_mul(31).wrapping_add(byte as u64);
    }
    h
}

fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

fn lerpf(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn mix_color(a: (f32, f32, f32), b: (f32, f32, f32), t: f32) -> (f32, f32, f32) {
    (lerpf(a.0, b.0, t), lerpf(a.1, b.1, t), lerpf(a.2, b.2, t))
}

fn biome_for_y(y: usize) -> BiomeStratum {
    for (index, entry) in BIOME_TABLE.iter().enumerate() {
        if y >= entry.y_min && y <= entry.y_max {
            return match index {
                0 => BiomeStratum::Underground,
                1 => BiomeStratum::Surface,
                2 => BiomeStratum::Midrise,
                _ => BiomeStratum::Skyline,
            };
        }
    }
    BiomeStratum::Surface
}

fn biome_rust_at(y: usize) -> f32 {
    match biome_for_y(y) {
        BiomeStratum::Underground => BIOME_TABLE[0].rust_mult,
        BiomeStratum::Surface => BIOME_TABLE[1].rust_mult,
        BiomeStratum::Midrise => BIOME_TABLE[2].rust_mult,
        BiomeStratum::Skyline => BIOME_TABLE[3].rust_mult,
    }
}

fn cell_to_material(cell: CellType) -> MaterialType {
    match cell {
        CellType::Empty | CellType::Vertical | CellType::Horizontal => MaterialType::Concrete,
        CellType::Bridge | CellType::Cable => MaterialType::Steel,
        CellType::Facade | CellType::Elevator => MaterialType::Glass,
        CellType::Stair | CellType::Antenna | CellType::Vent => MaterialType::Metal,
        CellType::Pipe => MaterialType::Rust,
    }
}

fn is_walkable_floor_cell(cell: CellType) -> bool {
    matches!(cell, CellType::Horizontal | CellType::Bridge)
}

fn is_traversal_carveable_cell(cell: CellType) -> bool {
    matches!(
        cell,
        CellType::Horizontal
            | CellType::Bridge
            | CellType::Facade
            | CellType::Stair
            | CellType::Pipe
            | CellType::Antenna
            | CellType::Cable
            | CellType::Vent
            | CellType::Elevator
    )
}

fn timestamp_token() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

fn validate_seed(seed: &str) -> bool {
    seed.len() == 8
        && seed
            .bytes()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
}

fn generate_seed() -> String {
    let chars = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let time_seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let mut rng = Rng32::new(time_seed ^ 0x9E37_79B9_7F4A_7C15);
    let mut out = String::new();
    for _ in 0..8 {
        out.push(chars[rng.choose_index(chars.len())] as char);
    }
    out
}

mod simplex {
    const PERM: [u8; 256] = [
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30,
        69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94,
        252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171,
        168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60,
        211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1,
        216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86,
        164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118,
        126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
        213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39,
        253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34,
        242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49,
        192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
        138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
    ];

    const GRAD3: [[f32; 3]; 12] = [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, -1.0],
    ];

    fn fastfloor(x: f32) -> i32 {
        let xi = x as i32;
        if x < xi as f32 {
            xi - 1
        } else {
            xi
        }
    }

    fn gdot3(g: [f32; 3], x: f32, y: f32, z: f32) -> f32 {
        g[0] * x + g[1] * y + g[2] * z
    }

    pub fn noise3(xin: f32, yin: f32, zin: f32) -> f32 {
        const F3: f32 = 1.0 / 3.0;
        const G3: f32 = 1.0 / 6.0;
        let s = (xin + yin + zin) * F3;
        let i = fastfloor(xin + s);
        let j = fastfloor(yin + s);
        let k = fastfloor(zin + s);
        let t = (i + j + k) as f32 * G3;
        let x0 = xin - (i as f32 - t);
        let y0 = yin - (j as f32 - t);
        let z0 = zin - (k as f32 - t);

        let (i1, j1, k1, i2, j2, k2) = if x0 >= y0 {
            if y0 >= z0 {
                (1, 0, 0, 1, 1, 0)
            } else if x0 >= z0 {
                (1, 0, 0, 1, 0, 1)
            } else {
                (0, 0, 1, 1, 0, 1)
            }
        } else if y0 < z0 {
            (0, 0, 1, 0, 1, 1)
        } else if x0 < z0 {
            (0, 1, 0, 0, 1, 1)
        } else {
            (0, 1, 0, 1, 1, 0)
        };

        let x1 = x0 - i1 as f32 + G3;
        let y1 = y0 - j1 as f32 + G3;
        let z1 = z0 - k1 as f32 + G3;
        let x2 = x0 - i2 as f32 + 2.0 * G3;
        let y2 = y0 - j2 as f32 + 2.0 * G3;
        let z2 = z0 - k2 as f32 + 2.0 * G3;
        let x3 = x0 - 1.0 + 3.0 * G3;
        let y3 = y0 - 1.0 + 3.0 * G3;
        let z3 = z0 - 1.0 + 3.0 * G3;

        let ii = (i & 255) as usize;
        let jj = (j & 255) as usize;
        let kk = (k & 255) as usize;

        let gi0 = PERM[(ii + PERM[(jj + PERM[kk] as usize) & 255] as usize) & 255] as usize % 12;
        let gi1 = PERM[(ii
            + i1 as usize
            + PERM[(jj + j1 as usize + PERM[(kk + k1 as usize) & 255] as usize) & 255] as usize)
            & 255] as usize
            % 12;
        let gi2 = PERM[(ii
            + i2 as usize
            + PERM[(jj + j2 as usize + PERM[(kk + k2 as usize) & 255] as usize) & 255] as usize)
            & 255] as usize
            % 12;
        let gi3 = PERM
            [(ii + 1 + PERM[(jj + 1 + PERM[(kk + 1) & 255] as usize) & 255] as usize) & 255]
            as usize
            % 12;

        let mut n0 = 0.0;
        let mut n1 = 0.0;
        let mut n2 = 0.0;
        let mut n3 = 0.0;

        let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
        if t0 >= 0.0 {
            let t0sq = t0 * t0;
            n0 = t0sq * t0sq * gdot3(GRAD3[gi0], x0, y0, z0);
        }
        let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
        if t1 >= 0.0 {
            let t1sq = t1 * t1;
            n1 = t1sq * t1sq * gdot3(GRAD3[gi1], x1, y1, z1);
        }
        let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
        if t2 >= 0.0 {
            let t2sq = t2 * t2;
            n2 = t2sq * t2sq * gdot3(GRAD3[gi2], x2, y2, z2);
        }
        let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
        if t3 >= 0.0 {
            let t3sq = t3 * t3;
            n3 = t3sq * t3sq * gdot3(GRAD3[gi3], x3, y3, z3);
        }

        32.0 * (n0 + n1 + n2 + n3)
    }
}

fn catmull_rom_point(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    p0 * (-0.5 * t3 + t2 - 0.5 * t)
        + p1 * (1.5 * t3 - 2.5 * t2 + 1.0)
        + p2 * (-1.5 * t3 + 2.0 * t2 + 0.5 * t)
        + p3 * (0.5 * t3 - 0.5 * t2)
}

fn rasterize_spline(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    steps: usize,
) -> Vec<(usize, usize, usize)> {
    let mut points = Vec::new();
    let mut last = None;
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let p = catmull_rom_point(p0, p1, p2, p3, t);
        let point = (
            p.x.round().max(0.0) as usize,
            p.z.round().max(0.0) as usize,
            p.y.round().max(0.0) as usize,
        );
        if Some(point) != last {
            points.push(point);
            last = Some(point);
        }
    }
    points
}

fn wfc_init_tables() -> ([[u16; 4]; WFC_TILE_COUNT], [f32; WFC_TILE_COUNT]) {
    let mut adjacency = [[0xFFFFu16; 4]; WFC_TILE_COUNT];
    let mut weights = [1.0f32; WFC_TILE_COUNT];
    weights[WFCTile::Empty as usize] = 3.0;
    weights[WFCTile::FloorSolid as usize] = 5.0;
    weights[WFCTile::CorridorNS as usize] = 2.0;
    weights[WFCTile::CorridorEW as usize] = 2.0;
    weights[WFCTile::RoomCenter as usize] = 3.0;
    weights[WFCTile::WallN as usize] = 1.5;
    weights[WFCTile::WallE as usize] = 1.5;
    weights[WFCTile::WallCornerNE as usize] = 0.8;
    weights[WFCTile::WallCornerNW as usize] = 0.8;
    weights[WFCTile::DoorN as usize] = 0.5;
    weights[WFCTile::DoorE as usize] = 0.5;
    weights[WFCTile::Stairwell as usize] = 0.3;
    weights[WFCTile::ElevatorShaft as usize] = 0.2;
    weights[WFCTile::FloorHalfN as usize] = 1.0;
    weights[WFCTile::FloorHalfE as usize] = 1.0;

    let no_wall_n = 0xFFFF & !(1u16 << WFCTile::WallN as usize);
    let no_wall_e = 0xFFFF & !(1u16 << WFCTile::WallE as usize);
    adjacency[WFCTile::WallN as usize][0] = no_wall_n;
    adjacency[WFCTile::WallN as usize][2] = no_wall_n;
    adjacency[WFCTile::WallE as usize][1] = no_wall_e;
    adjacency[WFCTile::WallE as usize][3] = no_wall_e;

    let corridor_ns_ew = (1u16 << WFCTile::WallN as usize)
        | (1u16 << WFCTile::WallE as usize)
        | (1u16 << WFCTile::WallCornerNE as usize)
        | (1u16 << WFCTile::WallCornerNW as usize)
        | (1u16 << WFCTile::Empty as usize);
    adjacency[WFCTile::CorridorNS as usize][1] = corridor_ns_ew;
    adjacency[WFCTile::CorridorNS as usize][3] = corridor_ns_ew;
    adjacency[WFCTile::CorridorEW as usize][0] = corridor_ns_ew;
    adjacency[WFCTile::CorridorEW as usize][2] = corridor_ns_ew;

    let struct_mask = (1u16 << WFCTile::FloorSolid as usize)
        | (1u16 << WFCTile::CorridorNS as usize)
        | (1u16 << WFCTile::CorridorEW as usize)
        | (1u16 << WFCTile::RoomCenter as usize)
        | (1u16 << WFCTile::DoorN as usize)
        | (1u16 << WFCTile::DoorE as usize)
        | (1u16 << WFCTile::Stairwell as usize)
        | (1u16 << WFCTile::ElevatorShaft as usize);
    for direction in 0..4 {
        adjacency[WFCTile::Stairwell as usize][direction] =
            struct_mask | (1u16 << WFCTile::Empty as usize);
        adjacency[WFCTile::ElevatorShaft as usize][direction] =
            struct_mask | (1u16 << WFCTile::Empty as usize);
    }

    (adjacency, weights)
}

fn wfc_calc_entropy(possible: u16, weights: &[f32; WFC_TILE_COUNT]) -> f32 {
    let mut sum = 0.0;
    let mut sum_log = 0.0;
    for (index, weight) in weights.iter().enumerate() {
        if possible & (1u16 << index) != 0 {
            sum += *weight;
            sum_log += *weight * (*weight + 1e-6).ln();
        }
    }
    if sum < 1e-6 {
        0.0
    } else {
        sum.ln() - sum_log / sum
    }
}

fn wfc_count_options(possible: u16) -> usize {
    (0..WFC_TILE_COUNT)
        .filter(|index| possible & (1u16 << index) != 0)
        .count()
}

struct WfcSolver {
    cells: Vec<WfcCell>,
    weights: [f32; WFC_TILE_COUNT],
    adjacency: [[u16; 4]; WFC_TILE_COUNT],
    rng: Rng32,
    backtrack_depth: usize,
}

impl WfcSolver {
    fn new(seed: u64, district: DistrictType, stratum: BiomeStratum) -> Self {
        let (adjacency, mut weights) = wfc_init_tables();
        match district {
            DistrictType::Industrial => {
                weights[WFCTile::CorridorNS as usize] *= 1.5;
                weights[WFCTile::CorridorEW as usize] *= 1.5;
            }
            DistrictType::Elite => {
                weights[WFCTile::RoomCenter as usize] *= 2.0;
                weights[WFCTile::FloorSolid as usize] *= 1.5;
            }
            DistrictType::Slum => {
                weights[WFCTile::Empty as usize] *= 2.0;
                weights[WFCTile::FloorHalfN as usize] *= 2.0;
            }
            _ => {}
        }
        match stratum {
            BiomeStratum::Underground => {
                weights[WFCTile::CorridorNS as usize] *= 1.5;
                weights[WFCTile::CorridorEW as usize] *= 1.5;
            }
            BiomeStratum::Skyline => {
                weights[WFCTile::Empty as usize] *= 3.0;
            }
            _ => {}
        }
        let mut cells = vec![
            WfcCell {
                possible: (1u16 << WFC_TILE_COUNT) - 1,
                collapsed_tile: None,
                entropy: 0.0,
            };
            GRID_SIZE * GRID_SIZE
        ];
        for cell in &mut cells {
            cell.entropy = wfc_calc_entropy(cell.possible, &weights);
        }
        Self {
            cells,
            weights,
            adjacency,
            rng: Rng32::new(seed),
            backtrack_depth: 0,
        }
    }

    fn idx(x: usize, z: usize) -> usize {
        x * GRID_SIZE + z
    }

    fn constrain(&mut self, x: usize, z: usize, tile: WFCTile) {
        let cell = &mut self.cells[Self::idx(x, z)];
        cell.possible = 1u16 << tile as usize;
        cell.collapsed_tile = Some(tile as usize);
        cell.entropy = 0.0;
    }

    fn propagate(&mut self, start_x: usize, start_z: usize) -> bool {
        let mut queue = vec![(start_x, start_z)];
        let directions = [(0isize, 1isize), (1, 0), (0, -1), (-1, 0)];
        let mut iterations = 0usize;
        while let Some((x, z)) = queue.pop() {
            if iterations > 1000 {
                break;
            }
            let current_tile = match self.cells[Self::idx(x, z)].collapsed_tile {
                Some(tile) => tile,
                None => {
                    iterations += 1;
                    continue;
                }
            };
            for (direction, (dx, dz)) in directions.iter().enumerate() {
                let nx = x as isize + dx;
                let nz = z as isize + dz;
                if nx < 0 || nz < 0 || nx >= GRID_SIZE as isize || nz >= GRID_SIZE as isize {
                    continue;
                }
                let nx = nx as usize;
                let nz = nz as usize;
                let index = Self::idx(nx, nz);
                if self.cells[index].collapsed_tile.is_some() {
                    continue;
                }
                let allowed = self.adjacency[current_tile][direction];
                let previous = self.cells[index].possible;
                self.cells[index].possible &= allowed;
                if self.cells[index].possible == 0 {
                    return false;
                }
                if self.cells[index].possible != previous {
                    self.cells[index].entropy =
                        wfc_calc_entropy(self.cells[index].possible, &self.weights);
                    if wfc_count_options(self.cells[index].possible) == 1 {
                        let tile = (0..WFC_TILE_COUNT)
                            .find(|candidate| self.cells[index].possible & (1u16 << candidate) != 0)
                            .unwrap_or(WFCTile::Empty as usize);
                        self.cells[index].collapsed_tile = Some(tile);
                        queue.push((nx, nz));
                    }
                }
            }
            iterations += 1;
        }
        true
    }

    fn collapse_one(&mut self) -> bool {
        let mut best = None;
        let mut best_entropy = f32::MAX;
        for x in 0..GRID_SIZE {
            for z in 0..GRID_SIZE {
                let cell = self.cells[Self::idx(x, z)];
                if cell.collapsed_tile.is_some() {
                    continue;
                }
                if cell.entropy < best_entropy {
                    best_entropy = cell.entropy;
                    best = Some((x, z));
                }
            }
        }
        let (bx, bz) = match best {
            Some(value) => value,
            None => return false,
        };
        let possible = self.cells[Self::idx(bx, bz)].possible;
        let mut total = 0.0;
        for index in 0..WFC_TILE_COUNT {
            if possible & (1u16 << index) != 0 {
                total += self.weights[index];
            }
        }
        let mut pick = self.rng.next_f32() * total.max(0.0001);
        let mut chosen = WFCTile::Empty as usize;
        for index in 0..WFC_TILE_COUNT {
            if possible & (1u16 << index) == 0 {
                continue;
            }
            pick -= self.weights[index];
            if pick <= 0.0 {
                chosen = index;
                break;
            }
        }
        let index = Self::idx(bx, bz);
        self.cells[index].collapsed_tile = Some(chosen);
        self.cells[index].possible = 1u16 << chosen;
        self.cells[index].entropy = 0.0;
        if !self.propagate(bx, bz) {
            self.backtrack_depth += 1;
            if self.backtrack_depth > 50 {
                return false;
            }
            self.cells[index].possible = ((1u16 << WFC_TILE_COUNT) - 1) & !(1u16 << chosen);
            self.cells[index].collapsed_tile = None;
            self.cells[index].entropy = wfc_calc_entropy(self.cells[index].possible, &self.weights);
            if self.cells[index].possible == 0 {
                self.cells[index].possible = 1u16;
                self.cells[index].collapsed_tile = Some(WFCTile::Empty as usize);
                self.cells[index].entropy = 0.0;
            }
        }
        true
    }

    fn solve(&mut self) {
        for _ in 0..1000 {
            if !self.collapse_one() {
                break;
            }
        }
        for cell in &mut self.cells {
            if cell.collapsed_tile.is_none() {
                let mut best_weight = -1.0f32;
                let mut best_tile = WFCTile::Empty as usize;
                for tile in 0..WFC_TILE_COUNT {
                    if cell.possible & (1u16 << tile) != 0 && self.weights[tile] > best_weight {
                        best_weight = self.weights[tile];
                        best_tile = tile;
                    }
                }
                cell.collapsed_tile = Some(best_tile);
            }
        }
    }

    fn tile_to_cell(tile: usize) -> CellType {
        match tile {
            x if x == WFCTile::FloorSolid as usize
                || x == WFCTile::FloorHalfN as usize
                || x == WFCTile::FloorHalfE as usize
                || x == WFCTile::RoomCenter as usize
                || x == WFCTile::CorridorNS as usize
                || x == WFCTile::CorridorEW as usize =>
            {
                CellType::Horizontal
            }
            x if x == WFCTile::WallN as usize
                || x == WFCTile::WallE as usize
                || x == WFCTile::WallCornerNE as usize
                || x == WFCTile::WallCornerNW as usize
                || x == WFCTile::DoorN as usize
                || x == WFCTile::DoorE as usize =>
            {
                CellType::Facade
            }
            x if x == WFCTile::Stairwell as usize => CellType::Stair,
            x if x == WFCTile::ElevatorShaft as usize => CellType::Elevator,
            _ => CellType::Empty,
        }
    }
}

struct LSystem {
    rng: Rng32,
    pipe_probability: f32,
    elevator_probability: f32,
    termination_probability: f32,
}

impl LSystem {
    fn new(seed: u64, district: DistrictType) -> Self {
        let props = DISTRICTS[district as usize];
        Self {
            rng: Rng32::new(seed),
            pipe_probability: props.pipe_probability,
            elevator_probability: props.elevator_probability,
            termination_probability: if district == DistrictType::Slum {
                0.30
            } else {
                0.05
            },
        }
    }

    fn produce(&mut self, input: &[char], iterations: usize) -> Vec<char> {
        let mut current = input.to_vec();
        for _ in 0..iterations {
            let mut next = Vec::new();
            for symbol in &current {
                if *symbol != 'C' {
                    next.push(*symbol);
                    continue;
                }
                let roll = self.rng.next_f32();
                if roll < 0.60 {
                    next.extend(['C', 'U', 'C']);
                } else if roll < 0.60 + self.pipe_probability {
                    next.extend(['C', '[', '+', 'P', ']']);
                } else if roll < 0.60 + self.pipe_probability * 2.0 {
                    next.extend(['C', '[', '-', 'P', ']']);
                } else if roll < 0.60 + self.pipe_probability * 2.0 + self.elevator_probability {
                    next.extend(['C', '[', '+', 'E', ']']);
                } else if roll
                    < 0.60 + self.pipe_probability * 2.0 + self.elevator_probability + 0.15
                {
                    next.extend(['C', 'S']);
                } else if roll < 1.0 - self.termination_probability {
                    next.push('C');
                }
            }
            current = next;
            if current.len() > 500 {
                break;
            }
        }
        current
    }
}

#[derive(Clone, Copy)]
struct TurtleState {
    x: isize,
    y: isize,
    z: isize,
    dx: isize,
    dz: isize,
}

struct MegaStructureGenerator {
    size: usize,
    layers: usize,
    seed: String,
    seed_hash: u64,
    rng: Rng32,
    grid: Vec<CellType>,
    support_map: Vec<bool>,
    district_map: Vec<DistrictType>,
    connections: Vec<ConnectionRecord>,
    rooms: Vec<RoomRecord>,
}

impl MegaStructureGenerator {
    fn new(seed: String) -> Self {
        let hash = seed_hash(&seed);
        let mut generator = Self {
            size: GRID_SIZE,
            layers: GRID_LAYERS,
            seed,
            seed_hash: hash,
            rng: Rng32::new(hash),
            grid: vec![CellType::Empty; GRID_SIZE * GRID_SIZE * GRID_LAYERS],
            support_map: vec![false; GRID_SIZE * GRID_SIZE * GRID_LAYERS],
            district_map: vec![DistrictType::Residential; GRID_SIZE * GRID_SIZE],
            connections: Vec::new(),
            rooms: Vec::new(),
        };
        generator.generate_district_map();
        generator
    }

    fn idx(&self, x: usize, z: usize, y: usize) -> usize {
        x * self.size * self.layers + z * self.layers + y
    }

    fn district_idx(&self, x: usize, z: usize) -> usize {
        x * self.size + z
    }

    fn get(&self, x: usize, z: usize, y: usize) -> CellType {
        self.grid[self.idx(x, z, y)]
    }

    fn set(&mut self, x: usize, z: usize, y: usize, cell: CellType, supported: bool) {
        let index = self.idx(x, z, y);
        self.grid[index] = cell;
        if supported {
            self.support_map[index] = true;
        }
    }

    fn support_at(&self, x: usize, z: usize, y: usize) -> bool {
        self.support_map[self.idx(x, z, y)]
    }

    fn district_at(&self, x: usize, z: usize) -> DistrictType {
        self.district_map[self.district_idx(x, z)]
    }

    fn generate_district_map(&mut self) {
        for x in 0..self.size {
            for z in 0..self.size {
                let noise = simplex::noise3(x as f32 * 0.05, z as f32 * 0.05, 0.0)
                    + simplex::noise3(x as f32 * 0.10, z as f32 * 0.10, 1.0) * 0.5
                    + simplex::noise3(x as f32 * 0.20, z as f32 * 0.20, 2.0) * 0.25;
                let district = if noise < -0.3 {
                    DistrictType::Slum
                } else if noise < -0.1 {
                    DistrictType::Industrial
                } else if noise < 0.1 {
                    DistrictType::Residential
                } else if noise < 0.3 {
                    DistrictType::Commercial
                } else {
                    DistrictType::Elite
                };
                let index = self.district_idx(x, z);
                self.district_map[index] = district;
            }
        }
    }

    fn generate(&mut self) {
        self.phase1_skeleton();
        self.phase2_floorplans();
        self.apply_floor_thickness();
        self.phase3_infrastructure();
        self.phase4_erosion();
        self.ensure_structural_integrity();
        self.add_support_pillars();
        self.carve_traversal_space();
    }

    fn phase1_skeleton(&mut self) {
        for x in 0..self.size {
            for z in 0..self.size {
                let district = self.district_at(x, z);
                let props = DISTRICTS[district as usize];
                let base_probability = 0.15 * props.core_density;
                let noise_mod = simplex::noise3(x as f32 * 0.1, z as f32 * 0.1, 3.0) * 0.1;
                if self.rng.next_f32() >= (base_probability + noise_mod).max(0.02) {
                    continue;
                }

                let height_range = (self.layers as f32 * props.vertical_variation) as usize;
                let min_height = self.layers.saturating_sub(height_range).max(5);
                let max_height = self.layers.saturating_sub(2).max(min_height);
                let height = self.rng.range_usize(min_height, max_height);

                let mut axiom = Vec::new();
                for _ in 0..height {
                    axiom.extend(['C', 'U']);
                }
                let mut lsystem =
                    LSystem::new(self.seed_hash ^ ((x as u64) << 16) ^ z as u64, district);
                let result = lsystem.produce(&axiom, 3);
                self.interpret_lsystem(&result, x, z);

                let base_width: usize = if district == DistrictType::Slum { 1 } else { 2 };
                for y in 0..height {
                    let current_width = base_width.saturating_sub(y / 5).max(1);
                    for dx in -(current_width as isize)..=(current_width as isize) {
                        for dz in -(current_width as isize)..=(current_width as isize) {
                            let nx = x as isize + dx;
                            let nz = z as isize + dz;
                            if nx < 0
                                || nz < 0
                                || nx >= self.size as isize
                                || nz >= self.size as isize
                            {
                                continue;
                            }
                            self.set(nx as usize, nz as usize, y, CellType::Vertical, true);
                        }
                    }
                }
            }
        }
    }

    fn interpret_lsystem(&mut self, symbols: &[char], start_x: usize, start_z: usize) {
        let mut state = TurtleState {
            x: start_x as isize,
            y: 0,
            z: start_z as isize,
            dx: 1,
            dz: 0,
        };
        let mut stack = Vec::new();
        for symbol in symbols {
            match *symbol {
                'C' => {
                    if state.x >= 0
                        && state.y >= 0
                        && state.z >= 0
                        && state.x < self.size as isize
                        && state.z < self.size as isize
                        && state.y < self.layers as isize
                    {
                        self.set(
                            state.x as usize,
                            state.z as usize,
                            state.y as usize,
                            CellType::Vertical,
                            true,
                        );
                    }
                }
                'U' => state.y += 1,
                'P' => {
                    for step in 0..3 {
                        let nx = state.x + state.dx * step;
                        let nz = state.z + state.dz * step;
                        if nx < 0
                            || nz < 0
                            || nx >= self.size as isize
                            || nz >= self.size as isize
                            || state.y < 0
                            || state.y >= self.layers as isize
                        {
                            continue;
                        }
                        let current = self.get(nx as usize, nz as usize, state.y as usize);
                        if current == CellType::Empty {
                            self.set(
                                nx as usize,
                                nz as usize,
                                state.y as usize,
                                CellType::Pipe,
                                false,
                            );
                        }
                    }
                }
                'E' => {
                    for dy in 0..3 {
                        let nx = state.x + state.dx;
                        let nz = state.z + state.dz;
                        let ny = state.y + dy;
                        if nx < 0
                            || nz < 0
                            || ny < 0
                            || nx >= self.size as isize
                            || nz >= self.size as isize
                            || ny >= self.layers as isize
                        {
                            continue;
                        }
                        if self.get(nx as usize, nz as usize, ny as usize) == CellType::Empty {
                            self.set(
                                nx as usize,
                                nz as usize,
                                ny as usize,
                                CellType::Elevator,
                                false,
                            );
                        }
                    }
                }
                'S' => {
                    if state.x >= 0
                        && state.z >= 0
                        && state.y >= 0
                        && state.x < self.size as isize
                        && state.z < self.size as isize
                        && state.y < self.layers as isize
                        && self.get(state.x as usize, state.z as usize, state.y as usize)
                            == CellType::Vertical
                    {
                        self.set(
                            state.x as usize,
                            state.z as usize,
                            state.y as usize,
                            CellType::Stair,
                            true,
                        );
                    }
                }
                '+' => {
                    let previous_dx = state.dx;
                    state.dx = -state.dz;
                    state.dz = previous_dx;
                }
                '-' => {
                    let previous_dx = state.dx;
                    state.dx = state.dz;
                    state.dz = -previous_dx;
                }
                '[' => stack.push(state),
                ']' => {
                    if let Some(previous) = stack.pop() {
                        state = previous;
                    }
                }
                _ => {}
            }
        }
    }

    fn phase2_floorplans(&mut self) {
        for y in 0..self.layers {
            let district = self.district_at(self.size / 2, self.size / 2);
            let stratum = biome_for_y(y);
            let mut solver = WfcSolver::new(self.seed_hash ^ (y as u64 * 12345), district, stratum);
            for x in 0..self.size {
                for z in 0..self.size {
                    match self.get(x, z, y) {
                        CellType::Vertical | CellType::Stair => {
                            solver.constrain(x, z, WFCTile::Stairwell)
                        }
                        CellType::Elevator => solver.constrain(x, z, WFCTile::ElevatorShaft),
                        _ => {}
                    }
                }
            }
            solver.solve();
            for x in 0..self.size {
                for z in 0..self.size {
                    let existing = self.get(x, z, y);
                    if existing != CellType::Empty && existing != CellType::Horizontal {
                        continue;
                    }
                    let tile = solver.cells[WfcSolver::idx(x, z)]
                        .collapsed_tile
                        .unwrap_or(WFCTile::Empty as usize);
                    let cell = WfcSolver::tile_to_cell(tile);
                    if cell == CellType::Empty {
                        continue;
                    }
                    let mut adjacent = y > 0 && self.support_at(x, z, y - 1);
                    if !adjacent {
                        for (dx, dz) in [(0isize, 1isize), (1, 0), (0, -1), (-1, 0)] {
                            let nx = x as isize + dx;
                            let nz = z as isize + dz;
                            if nx < 0
                                || nz < 0
                                || nx >= self.size as isize
                                || nz >= self.size as isize
                            {
                                continue;
                            }
                            if self.get(nx as usize, nz as usize, y) != CellType::Empty {
                                adjacent = true;
                                break;
                            }
                        }
                    }
                    if adjacent {
                        self.set(x, z, y, cell, true);
                        if tile == WFCTile::RoomCenter as usize {
                            self.rooms.push(RoomRecord {
                                position: [x, y, z],
                                district: district.name().to_owned(),
                                label: "ROOM_CENTER".to_owned(),
                            });
                        }
                    }
                }
            }
        }
    }

    fn phase3_infrastructure(&mut self) {
        self.add_spline_bridges();
        self.add_spline_cables();
        self.add_spline_pipes();
        self.add_rooftop_details();
        self.add_external_elevators();
    }

    fn apply_floor_thickness(&mut self) {
        for x in 0..self.size {
            for z in 0..self.size {
                let thickness = DISTRICTS[self.district_at(x, z) as usize]
                    .floor_thickness
                    .max(1) as usize;
                for y in 0..self.layers {
                    if self.get(x, z, y) != CellType::Horizontal {
                        continue;
                    }
                    for extra in 1..thickness {
                        if y + extra >= self.layers {
                            break;
                        }
                        if self.get(x, z, y + extra) == CellType::Empty {
                            self.set(x, z, y + extra, CellType::Horizontal, true);
                        }
                    }
                }
            }
        }
    }

    fn add_spline_bridges(&mut self) {
        for _ in 0..((self.size * self.layers) as f32 * 0.02) as usize {
            let y = self
                .rng
                .range_usize(3, self.layers.saturating_sub(2).max(3));
            let mut cores = Vec::new();
            for x in 0..self.size {
                for z in 0..self.size {
                    if self.get(x, z, y) == CellType::Vertical {
                        cores.push((x, z));
                    }
                }
            }
            if cores.len() < 2 {
                continue;
            }
            let start = cores[self.rng.choose_index(cores.len())];
            let mut end = cores[self.rng.choose_index(cores.len())];
            if start == end {
                end = cores[(self.rng.choose_index(cores.len()) + 1) % cores.len()];
            }
            if start == end {
                continue;
            }
            let arch = 1.2
                + simplex::noise3(start.0 as f32 * 0.1, y as f32 * 0.1, start.1 as f32 * 0.1).abs()
                    * 1.4;
            let p0 = vec3(start.0 as f32, y as f32, start.1 as f32);
            let p1 = vec3(start.0 as f32, y as f32, start.1 as f32);
            let p2 = vec3(end.0 as f32, y as f32 + arch, end.1 as f32);
            let p3 = vec3(end.0 as f32, y as f32, end.1 as f32);
            for (x, z, yv) in rasterize_spline(p0, p1, p2, p3, 30) {
                if x >= self.size || z >= self.size || yv >= self.layers {
                    continue;
                }
                if self.get(x, z, yv) == CellType::Empty {
                    self.set(x, z, yv, CellType::Bridge, true);
                }
            }
            self.connections.push(ConnectionRecord {
                kind: "bridge".to_owned(),
                start: [start.0, y, start.1],
                end: [end.0, y, end.1],
            });
        }
    }

    fn add_spline_cables(&mut self) {
        for _ in 0..((self.size as f32) * 0.5) as usize {
            let mut cores = Vec::new();
            for x in 0..self.size {
                for z in 0..self.size {
                    for y in 0..self.layers {
                        if self.get(x, z, y) == CellType::Vertical {
                            cores.push((x, z, y));
                        }
                    }
                }
            }
            if cores.len() < 2 {
                continue;
            }
            let start = cores[self.rng.choose_index(cores.len())];
            let end = cores[self.rng.choose_index(cores.len())];
            let manhattan = start.0.abs_diff(end.0) + start.1.abs_diff(end.1);
            if manhattan < 3 || manhattan > 15 {
                continue;
            }
            let droop = start.2.min(end.2) as f32 - 2.0;
            let p0 = vec3(start.0 as f32, start.2 as f32, start.1 as f32);
            let p1 = vec3(
                (start.0 + end.0) as f32 * 0.5,
                droop,
                (start.1 + end.1) as f32 * 0.5,
            );
            let p2 = vec3(
                (start.0 + end.0) as f32 * 0.5,
                droop + 0.5,
                (start.1 + end.1) as f32 * 0.5,
            );
            let p3 = vec3(end.0 as f32, end.2 as f32, end.1 as f32);
            for (x, z, y) in rasterize_spline(p0, p1, p2, p3, 25) {
                if x >= self.size || z >= self.size || y >= self.layers {
                    continue;
                }
                if self.get(x, z, y) == CellType::Empty {
                    self.set(x, z, y, CellType::Cable, false);
                }
            }
            self.connections.push(ConnectionRecord {
                kind: "cable".to_owned(),
                start: [start.0, start.2, start.1],
                end: [end.0, end.2, end.1],
            });
        }
    }

    fn add_spline_pipes(&mut self) {
        for _ in 0..((self.size * self.layers) as f32 * 0.03) as usize {
            let x = self.rng.range_usize(0, self.size.saturating_sub(1));
            let z = self.rng.range_usize(0, self.size.saturating_sub(1));
            let y = self.rng.range_usize(1, self.layers.saturating_sub(2));
            let base = self.get(x, z, y);
            if base != CellType::Vertical && base != CellType::Horizontal {
                continue;
            }
            let direction = self.rng.choose_index(4);
            let ddx = [1isize, -1, 0, 0];
            let ddz = [0isize, 0, 1, -1];
            let mut cx = x as isize;
            let mut cz = z as isize;
            let mut waypoints = vec![vec3(x as f32, y as f32, z as f32)];
            for _ in 0..3 {
                cx += ddx[direction] * (3 + self.rng.range_usize(0, 2) as isize);
                cz += ddz[direction] * (self.rng.range_usize(0, 1) as isize - 1);
                cx = cx.clamp(0, self.size as isize - 1);
                cz = cz.clamp(0, self.size as isize - 1);
                waypoints.push(vec3(cx as f32, y as f32, cz as f32));
            }
            if waypoints.len() >= 4 {
                for (px, pz, py) in
                    rasterize_spline(waypoints[0], waypoints[1], waypoints[2], waypoints[3], 20)
                {
                    if px >= self.size || pz >= self.size || py >= self.layers {
                        continue;
                    }
                    if self.get(px, pz, py) == CellType::Empty {
                        self.set(px, pz, py, CellType::Pipe, false);
                    }
                }
                self.connections.push(ConnectionRecord {
                    kind: "pipe".to_owned(),
                    start: [x, y, z],
                    end: [cx as usize, y, cz as usize],
                });
            }
        }
    }

    fn add_rooftop_details(&mut self) {
        for x in 0..self.size {
            for z in 0..self.size {
                for y in (0..self.layers).rev() {
                    if self.get(x, z, y) == CellType::Empty {
                        continue;
                    }
                    if self.rng.next_f32() < 0.15 && y < self.layers - 1 {
                        let detail = if self.rng.next_f32() < 0.5 {
                            CellType::Antenna
                        } else {
                            CellType::Vent
                        };
                        let height = self.rng.range_usize(1, 3);
                        for dy in 1..=height {
                            if y + dy < self.layers {
                                self.set(x, z, y + dy, detail, false);
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    fn add_external_elevators(&mut self) {
        let directions = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
        for x in 0..self.size {
            for z in 0..self.size {
                let is_core = (0..self.layers).any(|y| self.get(x, z, y) == CellType::Vertical);
                if !is_core || self.rng.next_f32() > 0.2 {
                    continue;
                }
                for (dx, dz) in directions {
                    let nx = x as isize + dx;
                    let nz = z as isize + dz;
                    if nx < 0 || nz < 0 || nx >= self.size as isize || nz >= self.size as isize {
                        continue;
                    }
                    for y in 0..self.layers {
                        let nx = nx as usize;
                        let nz = nz as usize;
                        if self.get(nx, nz, y) == CellType::Empty
                            && (y == 0
                                || matches!(
                                    self.get(nx, nz, y - 1),
                                    CellType::Elevator | CellType::Horizontal | CellType::Vertical
                                ))
                        {
                            self.set(nx, nz, y, CellType::Elevator, false);
                        }
                    }
                    break;
                }
            }
        }
    }

    fn phase4_erosion(&mut self) {
        self.erosion_structural_weakening();
        self.erosion_collapse_propagation();
    }

    fn count_empty_neighbors(&self, x: usize, z: usize, y: usize) -> usize {
        let directions = [
            (-1isize, 0isize, 0isize),
            (1, 0, 0),
            (0, 0, -1),
            (0, 0, 1),
            (0, -1, 0),
            (0, 1, 0),
        ];
        let mut count = 0;
        for (dx, dy, dz) in directions {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            let nz = z as isize + dz;
            if nx < 0
                || ny < 0
                || nz < 0
                || nx >= self.size as isize
                || ny >= self.layers as isize
                || nz >= self.size as isize
            {
                count += 1;
                continue;
            }
            if self.get(nx as usize, nz as usize, ny as usize) == CellType::Empty {
                count += 1;
            }
        }
        count
    }

    fn erosion_structural_weakening(&mut self) {
        for x in 0..self.size {
            for z in 0..self.size {
                for y in 0..self.layers {
                    if self.get(x, z, y) == CellType::Empty {
                        continue;
                    }
                    let exposure = self.count_empty_neighbors(x, z, y);
                    let noise =
                        simplex::noise3(x as f32 * 0.2, y as f32 * 0.2, z as f32 * 0.2) * 0.5 + 0.5;
                    let threshold = if biome_for_y(y) == BiomeStratum::Skyline {
                        0.3
                    } else {
                        0.6
                    };
                    if exposure >= 4 && noise > threshold {
                        let index = self.idx(x, z, y);
                        self.grid[index] = CellType::Empty;
                        self.support_map[index] = false;
                    }
                }
            }
        }
    }

    fn has_support(&self, x: usize, z: usize, y: usize) -> bool {
        if y == 0 {
            return true;
        }
        if self.support_at(x, z, y - 1) {
            return true;
        }
        for (dx, dz) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
            let nx = x as isize + dx;
            let nz = z as isize + dz;
            if nx < 0 || nz < 0 || nx >= self.size as isize || nz >= self.size as isize {
                continue;
            }
            let neighbor = self.get(nx as usize, nz as usize, y);
            if matches!(neighbor, CellType::Horizontal | CellType::Bridge) {
                return true;
            }
        }
        false
    }

    fn erosion_collapse_propagation(&mut self) {
        for _ in 0..3 {
            for x in 0..self.size {
                for z in 0..self.size {
                    for y in 1..self.layers {
                        let cell = self.get(x, z, y);
                        if cell == CellType::Empty || cell == CellType::Vertical {
                            continue;
                        }
                        if !self.has_support(x, z, y) {
                            let index = self.idx(x, z, y);
                            self.grid[index] = CellType::Empty;
                            self.support_map[index] = false;
                        }
                    }
                }
            }
        }
    }

    fn ensure_structural_integrity(&mut self) {
        for y in 1..self.layers {
            for x in 0..self.size {
                for z in 0..self.size {
                    let cell = self.get(x, z, y);
                    if matches!(cell, CellType::Horizontal | CellType::Facade)
                        && !self.has_support(x, z, y)
                    {
                        let index = self.idx(x, z, y);
                        self.grid[index] = CellType::Empty;
                        self.support_map[index] = false;
                    }
                }
            }
        }
    }

    fn add_support_pillars(&mut self) {
        for x in 0..self.size {
            for z in 0..self.size {
                for y in 1..self.layers {
                    if self.get(x, z, y) == CellType::Horizontal && !self.has_support(x, z, y) {
                        for py in (0..y).rev() {
                            if self.get(x, z, py) == CellType::Empty {
                                self.set(x, z, py, CellType::Vertical, true);
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    fn carve_traversal_space(&mut self) {
        for y in 0..self.layers.saturating_sub(2) {
            for x in 0..self.size {
                for z in 0..self.size {
                    if !is_walkable_floor_cell(self.get(x, z, y)) {
                        continue;
                    }
                    for dy in 1..=2 {
                        if y + dy >= self.layers {
                            continue;
                        }
                        let above = self.get(x, z, y + dy);
                        if is_traversal_carveable_cell(above) {
                            let index = self.idx(x, z, y + dy);
                            self.grid[index] = CellType::Empty;
                            self.support_map[index] = false;
                        }
                    }
                }
            }
        }
    }

    fn serialize(&self) -> String {
        let mut grid = vec![vec![vec![0u8; self.layers]; self.size]; self.size];
        for x in 0..self.size {
            for z in 0..self.size {
                for y in 0..self.layers {
                    grid[x][z][y] = self.get(x, z, y) as u8;
                }
            }
        }
        serde_json::to_string_pretty(&SavedStructure {
            seed: self.seed.clone(),
            size: self.size,
            layers: self.layers,
            grid,
            connections: self.connections.clone(),
            rooms: self.rooms.clone(),
        })
        .unwrap_or_else(|_| "{}".to_owned())
    }

    fn save_outputs(&self) {
        let _ = fs::write("current_seed.txt", &self.seed);
        let _ = fs::write("structure.json", self.serialize());
    }
}

fn hash_noise(seed_hash: u64, x: usize, z: usize, y: usize) -> f32 {
    let mut value = seed_hash
        ^ ((x as u64).wrapping_mul(0x9E37_79B1))
        ^ ((z as u64).wrapping_mul(0x85EB_CA77))
        ^ ((y as u64).wrapping_mul(0xC2B2_AE3D));
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51afd7ed558ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ceb9fe1a85ec53);
    value ^= value >> 33;
    (value as u32) as f32 / u32::MAX as f32
}

fn cell_style(
    generator: &MegaStructureGenerator,
    x: usize,
    z: usize,
    y: usize,
    cell: CellType,
) -> (Color, bool) {
    let district = generator.district_at(x, z);
    let district_props = DISTRICTS[district as usize];
    let mut style = MATERIALS[cell_to_material(cell) as usize];
    let base_tint = district_props.color_palette[(x + z + y) % district_props.color_palette.len()];
    let noise = simplex::noise3(x as f32 * 0.12, y as f32 * 0.12, z as f32 * 0.12) * 0.5 + 0.5;
    let patina = hash_noise(generator.seed_hash, x, z, y);
    let mut color = mix_color(style.base_color, base_tint, 0.16 + noise * 0.10);

    if cell == CellType::Facade && patina > 1.0 - district_props.neon_probability {
        style = MATERIALS[MaterialType::Neon as usize];
        color = match ((x + y + z) % 3) as i32 {
            0 => (0.10, 0.92, 0.96),
            1 => (0.92, 0.20, 0.84),
            _ => (0.95, 0.92, 0.20),
        };
    } else {
        let decay = biome_rust_at(y) * (0.06 + patina * 0.08);
        color = (
            clampf(color.0 * (1.0 - decay), 0.0, 1.0),
            clampf(color.1 * (1.0 - decay * 0.9), 0.0, 1.0),
            clampf(color.2 * (1.0 - decay * 0.7), 0.0, 1.0),
        );
    }

    if matches!(cell, CellType::Pipe | CellType::Cable) {
        color = mix_color(
            color,
            MATERIALS[MaterialType::Rust as usize].base_color,
            0.30,
        );
    }

    let is_translucent = style.alpha < 0.99;
    (
        Color::new(
            clampf(color.0, 0.0, 1.0),
            clampf(color.1, 0.0, 1.0),
            clampf(color.2, 0.0, 1.0),
            style.alpha,
        ),
        is_translucent,
    )
}

fn face_vertex(position: Vec3, uv: Vec2, color: Color) -> Vertex {
    Vertex::new2(position, uv, color)
}

fn push_face(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u16>,
    quad: [Vec3; 4],
    color: Color,
    brightness: f32,
) {
    let base = vertices.len() as u16;
    let shaded = Color::new(
        clampf(color.r * brightness, 0.0, 1.0),
        clampf(color.g * brightness, 0.0, 1.0),
        clampf(color.b * brightness, 0.0, 1.0),
        color.a,
    );
    vertices.push(face_vertex(quad[0], vec2(0.0, 0.0), shaded));
    vertices.push(face_vertex(quad[1], vec2(1.0, 0.0), shaded));
    vertices.push(face_vertex(quad[2], vec2(1.0, 1.0), shaded));
    vertices.push(face_vertex(quad[3], vec2(0.0, 1.0), shaded));
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn face_brightness(normal: Vec3) -> f32 {
    let sun = vec3(0.5, 1.0, 0.3).normalize();
    let fill = vec3(-0.3, 0.5, -0.7).normalize();
    let direct = normal.dot(sun).max(0.0);
    let secondary = normal.dot(fill).max(0.0) * 0.3;
    0.32 + direct * 0.55 + secondary
}

fn build_mesh_chunk(
    generator: &MegaStructureGenerator,
    ox: usize,
    oz: usize,
    oy: usize,
    translucent: bool,
) -> Option<SpatialChunk> {
    let max_x = (ox + CHUNK_SIZE_X).min(generator.size);
    let max_z = (oz + CHUNK_SIZE_Z).min(generator.size);
    let max_y = (oy + CHUNK_SIZE_Y).min(generator.layers);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let directions = [
        (1isize, 0isize, 0isize, vec3(1.0, 0.0, 0.0)),
        (-1, 0, 0, vec3(-1.0, 0.0, 0.0)),
        (0, 1, 0, vec3(0.0, 1.0, 0.0)),
        (0, -1, 0, vec3(0.0, -1.0, 0.0)),
        (0, 0, 1, vec3(0.0, 0.0, 1.0)),
        (0, 0, -1, vec3(0.0, 0.0, -1.0)),
    ];

    for x in ox..max_x {
        for z in oz..max_z {
            for y in oy..max_y {
                let cell = generator.get(x, z, y);
                if cell == CellType::Empty {
                    continue;
                }
                let (color, is_translucent) = cell_style(generator, x, z, y, cell);
                if is_translucent != translucent {
                    continue;
                }
                let center = vec3(x as f32, y as f32, z as f32);
                let half = 0.46;
                for (dx, dy, dz, normal) in directions {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    let nz = z as isize + dz;
                    let neighbor = if nx < 0
                        || ny < 0
                        || nz < 0
                        || nx >= generator.size as isize
                        || ny >= generator.layers as isize
                        || nz >= generator.size as isize
                    {
                        CellType::Empty
                    } else {
                        generator.get(nx as usize, nz as usize, ny as usize)
                    };
                    if neighbor != CellType::Empty {
                        let (_, neighbor_translucent) = cell_style(
                            generator,
                            nx.max(0) as usize,
                            nz.max(0) as usize,
                            ny.max(0) as usize,
                            neighbor,
                        );
                        if !neighbor_translucent || is_translucent == neighbor_translucent {
                            continue;
                        }
                    }
                    let quad = match (dx, dy, dz) {
                        (1, 0, 0) => [
                            center + vec3(half, -half, -half),
                            center + vec3(half, -half, half),
                            center + vec3(half, half, half),
                            center + vec3(half, half, -half),
                        ],
                        (-1, 0, 0) => [
                            center + vec3(-half, -half, half),
                            center + vec3(-half, -half, -half),
                            center + vec3(-half, half, -half),
                            center + vec3(-half, half, half),
                        ],
                        (0, 1, 0) => [
                            center + vec3(-half, half, -half),
                            center + vec3(half, half, -half),
                            center + vec3(half, half, half),
                            center + vec3(-half, half, half),
                        ],
                        (0, -1, 0) => [
                            center + vec3(-half, -half, half),
                            center + vec3(half, -half, half),
                            center + vec3(half, -half, -half),
                            center + vec3(-half, -half, -half),
                        ],
                        (0, 0, 1) => [
                            center + vec3(-half, -half, half),
                            center + vec3(-half, half, half),
                            center + vec3(half, half, half),
                            center + vec3(half, -half, half),
                        ],
                        _ => [
                            center + vec3(half, -half, -half),
                            center + vec3(half, half, -half),
                            center + vec3(-half, half, -half),
                            center + vec3(-half, -half, -half),
                        ],
                    };
                    push_face(
                        &mut vertices,
                        &mut indices,
                        quad,
                        color,
                        face_brightness(normal),
                    );
                }
            }
        }
    }

    if vertices.is_empty() {
        return None;
    }

    Some(SpatialChunk {
        mesh: Mesh {
            vertices,
            indices,
            texture: None,
        },
        center: vec3(
            (ox + max_x) as f32 * 0.5,
            (oy + max_y) as f32 * 0.5,
            (oz + max_z) as f32 * 0.5,
        ),
    })
}

fn build_render_world(generator: &MegaStructureGenerator) -> RenderWorld {
    let mut opaque_chunks = Vec::new();
    let mut translucent_chunks = Vec::new();
    for ox in (0..generator.size).step_by(CHUNK_SIZE_X) {
        for oz in (0..generator.size).step_by(CHUNK_SIZE_Z) {
            for oy in (0..generator.layers).step_by(CHUNK_SIZE_Y) {
                if let Some(chunk) = build_mesh_chunk(generator, ox, oz, oy, false) {
                    opaque_chunks.push(chunk);
                }
                if let Some(chunk) = build_mesh_chunk(generator, ox, oz, oy, true) {
                    translucent_chunks.push(chunk);
                }
            }
        }
    }
    RenderWorld {
        opaque_chunks,
        translucent_chunks,
    }
}

struct OrbitalCamera {
    target: Vec3,
    distance: f32,
    angle: f32,
    pitch: f32,
    target_angle: f32,
    target_pitch: f32,
    target_distance: f32,
    angle_velocity: f32,
    pitch_velocity: f32,
    zoom_velocity: f32,
    damping: f32,
    min_distance: f32,
    max_distance: f32,
    position: Vec3,
}

impl OrbitalCamera {
    fn new(target: Vec3, distance: f32) -> Self {
        let mut camera = Self {
            target,
            distance,
            angle: 45.0,
            pitch: 30.0,
            target_angle: 45.0,
            target_pitch: 30.0,
            target_distance: distance,
            angle_velocity: 0.0,
            pitch_velocity: 0.0,
            zoom_velocity: 0.0,
            damping: 0.85,
            min_distance: distance * 0.3,
            max_distance: distance * 5.0,
            position: vec3(0.0, 0.0, 0.0),
        };
        camera.position = camera.calc_position();
        camera
    }

    fn calc_position(&self) -> Vec3 {
        let rad_angle = self.angle.to_radians();
        let rad_pitch = self.pitch.to_radians();
        let x = self.distance * rad_pitch.cos() * rad_angle.cos();
        let y = self.distance * rad_pitch.sin();
        let z = self.distance * rad_pitch.cos() * rad_angle.sin();
        self.target + vec3(x, y, z)
    }

    fn update(&mut self, dt: f32) {
        let ad = self.target_angle - self.angle;
        let pd = self.target_pitch - self.pitch;
        let zd = self.target_distance - self.distance;
        self.angle_velocity += ad * dt * 5.0;
        self.pitch_velocity += pd * dt * 5.0;
        self.zoom_velocity += zd * dt * 3.0;
        self.angle_velocity *= self.damping;
        self.pitch_velocity *= self.damping;
        self.zoom_velocity *= self.damping;
        self.angle += self.angle_velocity * dt;
        self.pitch += self.pitch_velocity * dt;
        self.distance += self.zoom_velocity * dt;
        self.pitch = clampf(self.pitch, -89.0, 89.0);
        self.distance = clampf(self.distance, self.min_distance, self.max_distance);
        self.target_distance = clampf(self.target_distance, self.min_distance, self.max_distance);
        self.position = self.calc_position();
    }

    fn rotate(&mut self, da: f32, dp: f32) {
        self.target_angle += da;
        self.target_pitch += dp;
    }

    fn zoom(&mut self, delta: f32) {
        let bound = self.target.max_element().max(10.0);
        self.min_distance = bound * 0.3;
        self.max_distance = bound * 4.0;
        self.target_distance = clampf(
            self.target_distance + delta,
            self.min_distance,
            self.max_distance,
        );
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(vec3(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(forward).normalize();
        self.target += right * (dx * 0.12) + up * (dy * 0.12);
    }

    fn set_preset(&mut self, preset: usize) {
        const PRESETS: [(f32, f32); 5] = [
            (0.0, 89.0),
            (0.0, 0.0),
            (90.0, 0.0),
            (45.0, 30.0),
            (45.0, 35.264),
        ];
        let index = preset.min(PRESETS.len() - 1);
        self.target_angle = PRESETS[index].0;
        self.target_pitch = PRESETS[index].1;
        self.angle_velocity = 0.0;
        self.pitch_velocity = 0.0;
    }

    fn view_camera(&self, render_target: Option<RenderTarget>) -> Camera3D {
        Camera3D {
            position: self.position,
            target: self.target,
            up: vec3(0.0, 1.0, 0.0),
            fovy: CAMERA_FOV_DEGREES.to_radians(),
            projection: Projection::Perspective,
            render_target,
            z_near: 0.1,
            z_far: 500.0,
            ..Default::default()
        }
    }
}

struct FpsCamera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    speed: f32,
    sensitivity: f32,
    velocity_y: f32,
    on_ground: bool,
}

impl FpsCamera {
    const EYE_HEIGHT: f32 = 1.6;
    const RADIUS: f32 = 0.3;
    const GRAVITY: f32 = 9.8;
    const JUMP_VELOCITY: f32 = 5.0;
    const MAX_DELTA: f32 = 0.5;

    fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: -90.0,
            pitch: 0.0,
            speed: 5.0,
            sensitivity: 0.1,
            velocity_y: 0.0,
            on_ground: false,
        }
    }

    fn front(&self) -> Vec3 {
        let yaw = self.yaw.to_radians();
        let pitch = self.pitch.to_radians();
        vec3(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        )
        .normalize()
    }

    fn look_delta(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * self.sensitivity;
        self.pitch -= dy * self.sensitivity;
        self.pitch = clampf(self.pitch, -89.0, 89.0);
    }

    fn jump(&mut self) {
        if self.on_ground {
            self.velocity_y = Self::JUMP_VELOCITY;
            self.on_ground = false;
        }
    }

    fn collides_at(&self, position: Vec3, generator: &MegaStructureGenerator) -> bool {
        let offsets = [
            vec2(-Self::RADIUS, -Self::RADIUS),
            vec2(Self::RADIUS, -Self::RADIUS),
            vec2(-Self::RADIUS, Self::RADIUS),
            vec2(Self::RADIUS, Self::RADIUS),
        ];
        for offset in offsets {
            for height in [0.0, Self::EYE_HEIGHT] {
                let gx = (position.x + offset.x).floor() as isize;
                let gy = (position.y + height).floor() as isize;
                let gz = (position.z + offset.y).floor() as isize;
                if gx < 0
                    || gy < 0
                    || gz < 0
                    || gx >= generator.size as isize
                    || gy >= generator.layers as isize
                    || gz >= generator.size as isize
                {
                    continue;
                }
                if generator.get(gx as usize, gz as usize, gy as usize) != CellType::Empty {
                    return true;
                }
            }
        }
        false
    }

    fn update(
        &mut self,
        dt: f32,
        move_forward: f32,
        move_right: f32,
        generator: &MegaStructureGenerator,
    ) {
        let front = self.front();
        let right = front.cross(vec3(0.0, 1.0, 0.0)).normalize();
        let mut movement = (vec3(front.x, 0.0, front.z) * move_forward
            + vec3(right.x, 0.0, right.z) * move_right)
            * self.speed
            * dt;
        let length = movement.length();
        if length > Self::MAX_DELTA {
            movement *= Self::MAX_DELTA / length;
        }

        let mut next = self.position;
        next.x += movement.x;
        if self.collides_at(next, generator) {
            next.x = self.position.x;
        }
        next.z += movement.z;
        if self.collides_at(next, generator) {
            next.z = self.position.z;
        }

        self.velocity_y -= Self::GRAVITY * dt;
        let mut delta_y = self.velocity_y * dt;
        delta_y = clampf(delta_y, -Self::MAX_DELTA, Self::MAX_DELTA);
        next.y += delta_y;
        if self.collides_at(next, generator) {
            if self.velocity_y < 0.0 {
                self.on_ground = true;
            }
            self.velocity_y = 0.0;
            next.y = self.position.y;
        } else {
            self.on_ground = false;
        }
        if next.y < 0.0 {
            next.y = 0.0;
            self.velocity_y = 0.0;
            self.on_ground = true;
        }
        self.position = next;
    }

    fn view_camera(&self, render_target: Option<RenderTarget>) -> Camera3D {
        let eye = self.position + vec3(0.0, Self::EYE_HEIGHT, 0.0);
        Camera3D {
            position: eye,
            target: eye + self.front(),
            up: vec3(0.0, 1.0, 0.0),
            fovy: CAMERA_FOV_DEGREES.to_radians(),
            projection: Projection::Perspective,
            render_target,
            z_near: 0.1,
            z_far: 500.0,
            ..Default::default()
        }
    }
}

fn is_valid_spawn_cell(generator: &MegaStructureGenerator, x: usize, z: usize, y: usize) -> bool {
    if x == 0 || z == 0 || x >= generator.size - 1 || z >= generator.size - 1 {
        return false;
    }
    if y == 0 || y + 1 >= generator.layers {
        return false;
    }
    is_walkable_floor_cell(generator.get(x, z, y - 1))
        && generator.get(x, z, y) == CellType::Empty
        && generator.get(x, z, y + 1) == CellType::Empty
}

fn find_fps_spawn(generator: &MegaStructureGenerator) -> Vec3 {
    let center_x = generator.size as isize / 2;
    let center_z = generator.size as isize / 2;
    let center_y = generator.layers as isize / 3;
    let directions = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
    let mut best_score = i32::MIN;
    let mut best_position = vec3(
        generator.size as f32 / 2.0,
        1.0,
        generator.size as f32 / 2.0,
    );
    for y in 1..generator.layers - 1 {
        for x in 1..generator.size - 1 {
            for z in 1..generator.size - 1 {
                if !is_valid_spawn_cell(generator, x, z, y) {
                    continue;
                }
                let mut floor_links = 0;
                let mut open_links = 0;
                let mut sheltered_links = 0;
                for (dx, dz) in directions {
                    let nx = (x as isize + dx) as usize;
                    let nz = (z as isize + dz) as usize;
                    if is_walkable_floor_cell(generator.get(nx, nz, y - 1)) {
                        floor_links += 1;
                    }
                    if generator.get(nx, nz, y) == CellType::Empty
                        && generator.get(nx, nz, y + 1) == CellType::Empty
                    {
                        open_links += 1;
                    }
                    if generator.get(nx, nz, y - 1) != CellType::Empty
                        || generator.get(nx, nz, y + 1) != CellType::Empty
                    {
                        sheltered_links += 1;
                    }
                }
                let center_penalty = (x as isize - center_x).abs() as i32
                    + (z as isize - center_z).abs() as i32
                    + ((y as isize - center_y).abs() as i32 * 3);
                let score =
                    floor_links * 24 + open_links * 10 + sheltered_links * 4 - center_penalty;
                if score > best_score {
                    best_score = score;
                    best_position = vec3(x as f32 + 0.5, y as f32, z as f32 + 0.5);
                }
            }
        }
    }
    best_position
}

struct PostFxResources {
    scene_target: RenderTarget,
    material: Material,
    width: u32,
    height: u32,
}

impl PostFxResources {
    async fn new(width: u32, height: u32) -> Self {
        let scene_target = render_target_ex(
            width,
            height,
            RenderTargetParams {
                sample_count: 1,
                depth: true,
            },
        );
        scene_target.texture.set_filter(FilterMode::Linear);
        let material = load_material(
            ShaderSource::Glsl {
                vertex: POST_VERTEX,
                fragment: POST_FRAGMENT,
            },
            MaterialParams {
                uniforms: vec![
                    UniformDesc::new("FogDensity", UniformType::Float1),
                    UniformDesc::new("BloomIntensity", UniformType::Float1),
                    UniformDesc::new("Time", UniformType::Float1),
                    UniformDesc::new("ScreenSize", UniformType::Float2),
                ],
                ..Default::default()
            },
        )
        .expect("postfx material");
        Self {
            scene_target,
            material,
            width,
            height,
        }
    }

    async fn ensure_size(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        *self = Self::new(width, height).await;
    }
}

struct AppState {
    generator: MegaStructureGenerator,
    render_world: RenderWorld,
    orbital: OrbitalCamera,
    fps: FpsCamera,
    fps_mode: bool,
    postfx: PostFxResources,
    fog_density: f32,
    bloom_intensity: f32,
    enable_postfx: bool,
    inspection_mode: bool,
    show_legend: bool,
    selected_cell: Option<(usize, usize, usize)>,
    mouse_dragging: bool,
    last_mouse: Option<Vec2>,
    last_fps_mouse: Option<Vec2>,
    screenshot_requested: bool,
}

impl AppState {
    async fn new(seed: String) -> Self {
        let mut generator = MegaStructureGenerator::new(seed);
        generator.generate();
        generator.save_outputs();

        let render_world = build_render_world(&generator);
        let center = vec3(
            generator.size as f32 / 2.0,
            generator.layers as f32 / 2.0,
            generator.size as f32 / 2.0,
        );
        let distance = generator.size.max(generator.layers) as f32 * 1.5;
        let orbital = OrbitalCamera::new(center, distance);
        let fps = FpsCamera::new(find_fps_spawn(&generator));
        let postfx = PostFxResources::new(
            screen_width().max(1.0) as u32,
            screen_height().max(1.0) as u32,
        )
        .await;

        Self {
            generator,
            render_world,
            orbital,
            fps,
            fps_mode: false,
            postfx,
            fog_density: 0.0,
            bloom_intensity: 0.2,
            enable_postfx: false,
            inspection_mode: false,
            show_legend: true,
            selected_cell: None,
            mouse_dragging: false,
            last_mouse: None,
            last_fps_mouse: None,
            screenshot_requested: false,
        }
    }

    async fn regenerate(&mut self) {
        let seed = generate_seed();
        self.generator = MegaStructureGenerator::new(seed);
        self.generator.generate();
        self.generator.save_outputs();
        self.render_world = build_render_world(&self.generator);
        let center = vec3(
            self.generator.size as f32 / 2.0,
            self.generator.layers as f32 / 2.0,
            self.generator.size as f32 / 2.0,
        );
        let distance = self.generator.size.max(self.generator.layers) as f32 * 1.5;
        self.orbital = OrbitalCamera::new(center, distance);
        self.fps = FpsCamera::new(find_fps_spawn(&self.generator));
        self.selected_cell = None;
        self.mouse_dragging = false;
        self.last_mouse = None;
        self.last_fps_mouse = None;
        self.screenshot_requested = false;
    }

    fn current_seed(&self) -> &str {
        &self.generator.seed
    }
}

fn camera_ray(camera: &Camera3D, mouse_x: f32, mouse_y: f32) -> Vec3 {
    let aspect = (screen_width() / screen_height().max(1.0)).max(0.001);
    let tan_half = (CAMERA_FOV_DEGREES.to_radians() * 0.5).tan();
    let x = (2.0 * mouse_x / screen_width().max(1.0) - 1.0) * tan_half * aspect;
    let y = (1.0 - 2.0 * mouse_y / screen_height().max(1.0)) * tan_half;
    let forward = (camera.target - camera.position).normalize();
    let right = forward.cross(camera.up).normalize();
    let up = right.cross(forward).normalize();
    (forward + right * x + up * y).normalize()
}

fn ray_cast(
    generator: &MegaStructureGenerator,
    camera: &Camera3D,
    mouse_x: f32,
    mouse_y: f32,
) -> Option<(usize, usize, usize)> {
    let ray = camera_ray(camera, mouse_x, mouse_y);
    let mut position = camera.position;
    for _ in 0..220 {
        position += ray * 0.5;
        let gx = position.x.round() as isize;
        let gy = position.y.round() as isize;
        let gz = position.z.round() as isize;
        if gx < 0
            || gy < 0
            || gz < 0
            || gx >= generator.size as isize
            || gy >= generator.layers as isize
            || gz >= generator.size as isize
        {
            continue;
        }
        if generator.get(gx as usize, gz as usize, gy as usize) != CellType::Empty {
            return Some((gx as usize, gy as usize, gz as usize));
        }
    }
    None
}

fn draw_world(app: &AppState, camera_position: Vec3) {
    for chunk in &app.render_world.opaque_chunks {
        draw_mesh(&chunk.mesh);
    }
    let mut translucent: Vec<&SpatialChunk> = app.render_world.translucent_chunks.iter().collect();
    translucent.sort_by(|a, b| {
        b.center
            .distance_squared(camera_position)
            .partial_cmp(&a.center.distance_squared(camera_position))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for chunk in translucent {
        draw_mesh(&chunk.mesh);
    }
    if let Some((x, y, z)) = app.selected_cell {
        draw_cube_wires(
            vec3(x as f32, y as f32, z as f32),
            vec3(0.96, 0.96, 0.96),
            YELLOW,
        );
    }
}

fn draw_overlay(app: &AppState) {
    let padding = 12.0;
    let line_height = 22.0;
    draw_rectangle(8.0, 8.0, 340.0, 28.0, Color::new(0.0, 0.0, 0.0, 0.65));
    draw_text(
        &format!("Seed: {}", app.current_seed()),
        padding,
        28.0,
        24.0,
        WHITE,
    );

    let controls = [
        "Drag: Rotate | Wheel: Zoom | WASD: Pan",
        "1-5: Presets | TAB: FPS | Space: Jump",
        "R: Regenerate | S: Screenshot | I: Inspect",
        "P: PostFX | [ ]: Fog | - =: Bloom",
        "L: Legend | Q/Esc: Quit",
    ];
    let mut y = 48.0;
    for line in controls {
        draw_rectangle(8.0, y - 18.0, 420.0, 22.0, Color::new(0.0, 0.0, 0.0, 0.55));
        draw_text(line, padding, y, 20.0, Color::new(0.80, 0.80, 0.80, 1.0));
        y += line_height;
    }

    draw_rectangle(8.0, y - 18.0, 180.0, 22.0, Color::new(0.0, 0.0, 0.0, 0.65));
    draw_text(
        if app.fps_mode {
            "Mode: FPS"
        } else {
            "Mode: Orbital"
        },
        padding,
        y,
        20.0,
        Color::new(1.0, 1.0, 0.55, 1.0),
    );
    y += line_height;

    if app.enable_postfx {
        draw_rectangle(8.0, y - 18.0, 250.0, 22.0, Color::new(0.0, 0.0, 0.0, 0.55));
        draw_text(
            &format!(
                "Fog: {:.2} | Bloom: {:.2}",
                app.fog_density, app.bloom_intensity
            ),
            padding,
            y,
            20.0,
            Color::new(0.78, 0.78, 0.78, 1.0),
        );
        y += line_height;
    }

    if app.inspection_mode {
        draw_rectangle(8.0, y - 18.0, 180.0, 22.0, Color::new(0.0, 0.0, 0.0, 0.65));
        draw_text(
            "Inspection Enabled",
            padding,
            y,
            20.0,
            Color::new(0.85, 0.85, 0.45, 1.0),
        );
        y += line_height;
    }

    if let Some((x, yv, z)) = app.selected_cell {
        let district = app.generator.district_at(x, z);
        let cell = app.generator.get(x, z, yv);
        draw_rectangle(8.0, y + 2.0, 320.0, 90.0, Color::new(0.0, 0.0, 0.0, 0.72));
        draw_text(
            "INSPECT",
            padding,
            y + 24.0,
            24.0,
            Color::new(1.0, 1.0, 0.55, 1.0),
        );
        draw_text(
            &format!("Pos: ({}, {}, {})", x, yv, z),
            padding,
            y + 46.0,
            20.0,
            LIGHTGRAY,
        );
        draw_text(
            &format!("Type: {}", cell.name()),
            padding,
            y + 66.0,
            20.0,
            LIGHTGRAY,
        );
        draw_text(
            &format!("Zone: {}", district.name()),
            padding,
            y + 86.0,
            20.0,
            LIGHTGRAY,
        );
    }

    if app.show_legend {
        let legend_y = screen_height() - 182.0;
        draw_rectangle(8.0, legend_y, 200.0, 170.0, Color::new(0.0, 0.0, 0.0, 0.65));
        draw_text("Materials", padding, legend_y + 24.0, 24.0, WHITE);
        let items = [
            (
                "Concrete",
                MATERIALS[MaterialType::Concrete as usize].base_color,
            ),
            ("Glass", MATERIALS[MaterialType::Glass as usize].base_color),
            ("Metal", MATERIALS[MaterialType::Metal as usize].base_color),
            ("Neon", MATERIALS[MaterialType::Neon as usize].base_color),
            ("Rust", MATERIALS[MaterialType::Rust as usize].base_color),
            ("Steel", MATERIALS[MaterialType::Steel as usize].base_color),
        ];
        let mut ly = legend_y + 38.0;
        for (label, rgb) in items {
            draw_rectangle(
                16.0,
                ly - 12.0,
                16.0,
                16.0,
                Color::new(rgb.0, rgb.1, rgb.2, 1.0),
            );
            draw_text(label, 42.0, ly, 20.0, LIGHTGRAY);
            ly += 22.0;
        }
    }
}

fn take_screenshot(seed: &str) {
    let _ = fs::create_dir_all("screenshots");
    let image = get_screen_data();
    let path = format!("screenshots/gibson_{}_{}.png", seed, timestamp_token());
    image.export_png(&path);
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Gibson Rust".to_owned(),
        window_width: 1600,
        window_height: 900,
        high_dpi: true,
        window_resizable: true,
        sample_count: 4,
        ..Default::default()
    }
}

const POST_VERTEX: &str = r#"#version 100
attribute vec3 position;
attribute vec2 texcoord;
attribute vec4 color0;

varying lowp vec2 uv;
varying lowp vec4 color;

uniform mat4 Model;
uniform mat4 Projection;

void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
    color = color0 / 255.0;
    uv = texcoord;
}
"#;

const POST_FRAGMENT: &str = r#"#version 100
precision lowp float;

varying vec2 uv;
varying vec4 color;

uniform sampler2D Texture;
uniform float FogDensity;
uniform float BloomIntensity;
uniform float Time;
uniform vec2 ScreenSize;

void main() {
    vec3 scene = texture2D(Texture, uv).rgb * color.rgb;
    vec2 texel = 1.0 / max(ScreenSize, vec2(1.0, 1.0));
    vec3 blur =
        texture2D(Texture, uv + vec2(texel.x, 0.0)).rgb +
        texture2D(Texture, uv - vec2(texel.x, 0.0)).rgb +
        texture2D(Texture, uv + vec2(0.0, texel.y)).rgb +
        texture2D(Texture, uv - vec2(0.0, texel.y)).rgb;
    blur *= 0.25;

    vec3 composed = scene + blur * BloomIntensity;
    float fog = clamp(FogDensity * (1.1 - uv.y), 0.0, 1.0);
    vec3 fog_color = vec3(0.10, 0.10, 0.12);
    composed = mix(composed, fog_color, fog);

    vec2 vignette_uv = uv * (1.0 - uv.yx);
    float vignette = clamp(pow(16.0 * vignette_uv.x * vignette_uv.y, 0.22), 0.0, 1.0);
    composed *= vignette;

    float grain = fract(sin(dot(uv * (Time + 1.0), vec2(12.9898, 78.233))) * 43758.5453);
    composed += vec3((grain - 0.5) * 0.025);
    gl_FragColor = vec4(composed, 1.0);
}
"#;

#[macroquad::main(window_conf)]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed = if args.len() > 1 {
        let candidate = args[1].to_uppercase();
        if !validate_seed(&candidate) {
            eprintln!("Invalid seed '{}'. Must be 8 alphanumeric chars.", args[1]);
            std::process::exit(1);
        }
        candidate
    } else {
        generate_seed()
    };

    let mut app = AppState::new(seed).await;

    loop {
        app.postfx
            .ensure_size(
                screen_width().max(1.0) as u32,
                screen_height().max(1.0) as u32,
            )
            .await;

        let dt = get_frame_time().max(1.0 / 120.0);

        if is_key_pressed(KeyCode::Escape) || is_key_pressed(KeyCode::Q) {
            break;
        }
        if is_key_pressed(KeyCode::Tab) {
            app.fps_mode = !app.fps_mode;
            set_cursor_grab(app.fps_mode);
            show_mouse(!app.fps_mode);
            app.mouse_dragging = false;
            app.last_mouse = None;
            app.last_fps_mouse = None;
            if app.fps_mode {
                app.fps = FpsCamera::new(find_fps_spawn(&app.generator));
            }
        }
        if is_key_pressed(KeyCode::R) {
            app.regenerate().await;
        }
        if is_key_pressed(KeyCode::S) {
            app.screenshot_requested = true;
        }
        if is_key_pressed(KeyCode::P) {
            app.enable_postfx = !app.enable_postfx;
        }
        if is_key_pressed(KeyCode::I) {
            app.inspection_mode = !app.inspection_mode;
            app.mouse_dragging = false;
            app.last_mouse = None;
            if !app.inspection_mode {
                app.selected_cell = None;
            }
        }
        if is_key_pressed(KeyCode::L) {
            app.show_legend = !app.show_legend;
        }
        if is_key_down(KeyCode::LeftBracket) {
            app.fog_density = (app.fog_density - 0.01).max(0.0);
        }
        if is_key_down(KeyCode::RightBracket) {
            app.fog_density = (app.fog_density + 0.01).min(2.0);
        }
        if is_key_down(KeyCode::Minus) {
            app.bloom_intensity = (app.bloom_intensity - 0.01).max(0.0);
        }
        if is_key_down(KeyCode::Equal) {
            app.bloom_intensity = (app.bloom_intensity + 0.01).min(2.0);
        }

        if !app.fps_mode {
            if is_key_down(KeyCode::W) {
                app.orbital.pan(0.0, 1.0);
            }
            if is_key_down(KeyCode::S) {
                app.orbital.pan(0.0, -1.0);
            }
            if is_key_down(KeyCode::A) {
                app.orbital.pan(-1.0, 0.0);
            }
            if is_key_down(KeyCode::D) {
                app.orbital.pan(1.0, 0.0);
            }

            if is_key_pressed(KeyCode::Key1) {
                app.orbital.set_preset(0);
            }
            if is_key_pressed(KeyCode::Key2) {
                app.orbital.set_preset(1);
            }
            if is_key_pressed(KeyCode::Key3) {
                app.orbital.set_preset(2);
            }
            if is_key_pressed(KeyCode::Key4) {
                app.orbital.set_preset(3);
            }
            if is_key_pressed(KeyCode::Key5) {
                app.orbital.set_preset(4);
            }

            let wheel = mouse_wheel().1;
            if wheel.abs() > 0.01 {
                app.orbital.zoom(-wheel * 3.0);
            }

            if app.inspection_mode {
                if is_mouse_button_pressed(MouseButton::Left) {
                    let camera = app.orbital.view_camera(None);
                    let (mx, my) = mouse_position();
                    app.selected_cell = ray_cast(&app.generator, &camera, mx, my);
                }
            } else {
                if is_mouse_button_pressed(MouseButton::Left) {
                    app.mouse_dragging = true;
                    let (mx, my) = mouse_position();
                    app.last_mouse = Some(vec2(mx, my));
                }
                if is_mouse_button_released(MouseButton::Left) {
                    app.mouse_dragging = false;
                    app.last_mouse = None;
                }
                if app.mouse_dragging {
                    let (mx, my) = mouse_position();
                    let current = vec2(mx, my);
                    if let Some(previous) = app.last_mouse {
                        let delta = current - previous;
                        app.orbital.rotate(-delta.x * 0.3, -delta.y * 0.3);
                    }
                    app.last_mouse = Some(current);
                }
            }
            app.orbital.update(dt);
        } else {
            let mut move_forward = 0.0;
            let mut move_right = 0.0;
            if is_key_down(KeyCode::W) {
                move_forward += 1.0;
            }
            if is_key_down(KeyCode::S) {
                move_forward -= 1.0;
            }
            if is_key_down(KeyCode::D) {
                move_right += 1.0;
            }
            if is_key_down(KeyCode::A) {
                move_right -= 1.0;
            }
            if is_key_pressed(KeyCode::Space) {
                app.fps.jump();
            }

            let (mx, my) = mouse_position();
            let mouse = vec2(mx, my);
            if let Some(previous) = app.last_fps_mouse {
                let delta = mouse - previous;
                app.fps.look_delta(delta.x, delta.y);
            }
            app.last_fps_mouse = Some(mouse);
            app.fps.update(dt, move_forward, move_right, &app.generator);
        }

        let render_camera = if app.fps_mode {
            app.fps.view_camera(Some(app.postfx.scene_target.clone()))
        } else {
            app.orbital
                .view_camera(Some(app.postfx.scene_target.clone()))
        };
        set_camera(&render_camera);
        clear_background(Color::new(0.05, 0.05, 0.08, 1.0));
        let camera_position = render_camera.position;
        draw_world(&app, camera_position);

        set_default_camera();
        clear_background(Color::new(0.05, 0.05, 0.08, 1.0));

        if app.enable_postfx {
            app.postfx
                .material
                .set_uniform("FogDensity", app.fog_density);
            app.postfx
                .material
                .set_uniform("BloomIntensity", app.bloom_intensity);
            app.postfx.material.set_uniform("Time", get_time() as f32);
            app.postfx
                .material
                .set_uniform("ScreenSize", vec2(screen_width(), screen_height()));
            gl_use_material(&app.postfx.material);
            draw_texture_ex(
                &app.postfx.scene_target.texture,
                0.0,
                0.0,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(screen_width(), screen_height())),
                    flip_y: true,
                    ..Default::default()
                },
            );
            gl_use_default_material();
        } else {
            draw_texture_ex(
                &app.postfx.scene_target.texture,
                0.0,
                0.0,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(screen_width(), screen_height())),
                    flip_y: true,
                    ..Default::default()
                },
            );
        }

        draw_overlay(&app);
        if app.screenshot_requested {
            take_screenshot(app.current_seed());
            app.screenshot_requested = false;
        }
        next_frame().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_seed_shape() {
        assert!(validate_seed("ABCD1234"));
        assert!(!validate_seed("abc123"));
        assert!(!validate_seed("ABCD-123"));
    }

    #[test]
    fn generation_is_deterministic_for_same_seed() {
        let seed = "ABCD1234".to_owned();
        let mut a = MegaStructureGenerator::new(seed.clone());
        a.generate();
        let mut b = MegaStructureGenerator::new(seed);
        b.generate();
        assert_eq!(a.serialize(), b.serialize());
    }
}
