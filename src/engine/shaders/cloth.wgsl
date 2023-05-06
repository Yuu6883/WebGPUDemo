struct ClothPoint {
    position: vec4<f32>, // using w component to store if this point is fixed (padded anyways)
    normal:   vec4<f32>,
    force:    vec4<f32>,
    velocity: vec4<f32>,
};

struct ClothPointShared {
    p: vec4<f32>,
    v: vec4<f32>,
};

struct ClothPoints {
    data: array<ClothPoint>,
};

struct SimulationConstants {
    mass:             f32,
    rest_length:      f32,
    spring_constant:  f32,
    damping_constant: f32,
    floor_y:          f32,
};

struct Dimension {
    size: vec2<u32>,
};

struct Vectors {
    wind: vec3<f32>,
    gravity: vec3<f32>,
};

struct Indices {
    data: array<array<u32, 3>>,
};

// Group 1 is per cloth
@group(1) @binding(0) var<uniform> constants: SimulationConstants;
@group(1) @binding(1) var<uniform> dimension: Dimension;
@group(1) @binding(2) var<uniform> vectors: Vectors;

const TILE_SIZE = 16;
const TILE_SIZE_U = 16u;
const INNER_TILE = 14u;

@group(0) @binding(0) var<storage, read_write> indices: Indices;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn init_indices(
  @builtin(workgroup_id)        blockIdx :  vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
    let tx = threadIdx.x;
    let ty = threadIdx.y;

    let row = blockIdx.y * TILE_SIZE_U + ty;
    let col = blockIdx.x * TILE_SIZE_U + tx;

    let out_w = dimension.size.x;
    let out_h = dimension.size.y;

    if (row >= out_h - 1u || col >= out_w - 1u) {
        return;
    }

    let offset = (row * (out_w - 1u) + col) * 2u;

    // Triangle 1
    indices.data[offset][0] = row * out_w + col;
    indices.data[offset][1] = (row + 1u) * out_w + col;
    indices.data[offset][2] = row * out_w + col + 1u;
    
    // Triangle 2
    indices.data[offset + 1u][0] = row * out_w + col + 1u;
    indices.data[offset + 1u][1] = (row + 1u) * out_w + col;
    indices.data[offset + 1u][2] = (row + 1u) * out_w + col + 1u;
}

@group(0) @binding(0) var<storage, read_write> points: ClothPoints;

var<private> force: vec3<f32>;
var<private> p1: vec3<f32>;
var<private> v1: vec3<f32>;

const SQRT2 = 1.4142135623730951;
const EPSIL = 0.0001;

fn spring_damper(p2: vec4<f32>, v2: vec4<f32>, rest_length: f32) {
    // Empty padded point
    if (v2.w < 0.0) {
        return;
    }

    let delta = p2.xyz - p1;
    let len = length(delta);

    if (len < EPSIL) {
        return;
    }

    let dir = normalize(delta);

    // Spring force
    force = force + constants.spring_constant * (len - rest_length) * dir;

    // Damper force
    let v_close = dot(v1 - v2.xyz, dir);
    force = force - constants.damping_constant * v_close * dir;
}

const AIR_DENSITY = 1.225;
const DRAG_COEFFI = 1.5;

fn aerodynamic(p2: ClothPointShared, p3: ClothPointShared) {
    // Empty padded points
    if (p2.v.w < 0.0 || p3.v.w < 0.0) {
        return;
    }

    let surf_v = (v1 + p2.v.xyz + p3.v.xyz) / 3.0;
    let delta_v = surf_v - vectors.wind;
    let len = length(delta_v);

    if (len < EPSIL) {
        return;
    }

    let dir = normalize(delta_v);

    let prod = cross(p2.p.xyz - p1, p3.p.xyz - p1);

    if (length(prod) < EPSIL) {
        return;
    }

    let norm = normalize(prod);
    let area = length(prod) / 2.0 * dot(norm, dir);

    force = force + -0.5 * AIR_DENSITY * len * len * DRAG_COEFFI * area * norm / 3.9;
}

var<workgroup> tile : array<array<ClothPointShared, 16>, 16>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn calc_forces(
  @builtin(workgroup_id)        blockIdx :  vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
    let tx = threadIdx.x;
    let ty = threadIdx.y;

    let row_o = i32(blockIdx.y * INNER_TILE + ty);
    let col_o = i32(blockIdx.x * INNER_TILE + tx);

    // Could be -1 so it's all casted to signed
    let row_i = i32(row_o) - 1;
    let col_i = i32(col_o) - 1;

    let out_w = i32(dimension.size.x);
    let out_h = i32(dimension.size.y);
    
    // Load tile
    if (row_i >= 0 && row_i < out_h && 
        col_i >= 0 && col_i < out_w) {
        tile[ty][tx].p = points.data[row_i * out_w + col_i].position;
        tile[ty][tx].v = points.data[row_i * out_w + col_i].velocity;
    } else {
        tile[ty][tx].p = vec4<f32>(0.0, 0.0, 0.0, -1.0);
        tile[ty][tx].v = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    workgroupBarrier();

    let cx = tx + 1u;
    let cy = ty + 1u;

    // Out of grid || out of tile || fixed point
    if (row_o >= out_h || col_o >= out_w || 
        tx >= INNER_TILE || ty >= INNER_TILE ||
        tile[cy][cx].p.w < 0.0) {
        return;
    }

    force = vectors.gravity * constants.mass;

    p1 = tile[cy][cx].p.xyz;
    v1 = tile[cy][cx].v.xyz;

    let rest_len = constants.rest_length;
    let diag_len = rest_len * SQRT2;

    // 8x spring damper force accumulation 
    spring_damper(tile[cy - 1u][cx - 1u].p, tile[cy - 1u][cx - 1u].v, diag_len);
    spring_damper(tile[cy - 1u][cx - 0u].p, tile[cy - 1u][cx - 0u].v, rest_len);
    spring_damper(tile[cy - 1u][cx + 1u].p, tile[cy - 1u][cx + 1u].v, diag_len);
    
    spring_damper(tile[cy][cx - 1u].p, tile[cy][cx - 1u].v, rest_len);
    spring_damper(tile[cy][cx + 1u].p, tile[cy][cx + 1u].v, rest_len);
    
    spring_damper(tile[cy + 1u][cx - 1u].p, tile[cy + 1u][cx - 1u].v, diag_len);
    spring_damper(tile[cy + 1u][cx - 0u].p, tile[cy + 1u][cx - 0u].v, rest_len);
    spring_damper(tile[cy + 1u][cx + 1u].p, tile[cy + 1u][cx + 1u].v, diag_len);

    // 8 Triangles aerodynamic force accumulation
    aerodynamic(tile[cy - 1u][cx - 1u], tile[cy - 1u][cx - 0u]);
    aerodynamic(tile[cy - 1u][cx - 0u], tile[cy - 1u][cx + 1u]);
    aerodynamic(tile[cy - 1u][cx + 1u], tile[cy - 0u][cx + 1u]);
    aerodynamic(tile[cy - 0u][cx + 1u], tile[cy + 1u][cx + 1u]);
    aerodynamic(tile[cy + 1u][cx + 1u], tile[cy + 1u][cx + 0u]);
    aerodynamic(tile[cy + 1u][cx + 0u], tile[cy + 1u][cx - 1u]);
    aerodynamic(tile[cy + 1u][cx - 1u], tile[cy + 0u][cx - 1u]);
    aerodynamic(tile[cy + 0u][cx - 1u], tile[cy - 1u][cx - 1u]);
    
    points.data[row_o * out_w + col_o].force = vec4<f32>(force, 0.0);
}

var<private> accum_norm: vec3<f32>;

fn triangle_normal(p2: vec4<f32>, p3: vec4<f32>) {
    if (p2.w < 0.0 || p3.w < 0.0) {
        return;
    }

    let prod = cross(p2.xyz - p1, p3.xyz - p1);

    if (length(prod) < EPSIL) {
        return;
    }

    let norm = normalize(prod);

    accum_norm = accum_norm + norm;
}

var<workgroup> p_tile : array<array<vec4<f32>, 16>, 16>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn calc_normal(
    @builtin(workgroup_id)        blockIdx :  vec3<u32>,
    @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
    let tx = threadIdx.x;
    let ty = threadIdx.y;

    let row_o = i32(blockIdx.y * INNER_TILE + ty);
    let col_o = i32(blockIdx.x * INNER_TILE + tx);

    // Could be -1 so it's all casted to signed
    let row_i = i32(row_o) - 1;
    let col_i = i32(col_o) - 1;

    let out_w = i32(dimension.size.x);
    let out_h = i32(dimension.size.y);
    
    // Load position tile
    if (row_i >= 0 && row_i < out_h && 
        col_i >= 0 && col_i < out_w) {
        p_tile[ty][tx] = points.data[row_i * out_w + col_i].position;
    } else {
        p_tile[ty][tx] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    workgroupBarrier();

    let cx = tx + 1u;
    let cy = ty + 1u;

    // Out of grid || out of tile
    if (row_o >= out_h || col_o >= out_w || 
        tx >= INNER_TILE || ty >= INNER_TILE) {
        return;
    }
    
    p1 = p_tile[cy][cx].xyz;

    accum_norm = vec3<f32>(0.0);

    // 8 Triangles normal accumulation
    triangle_normal(p_tile[cy - 1u][cx - 1u], p_tile[cy - 1u][cx - 0u]);
    triangle_normal(p_tile[cy - 1u][cx - 0u], p_tile[cy - 1u][cx + 1u]);
    triangle_normal(p_tile[cy - 1u][cx + 1u], p_tile[cy - 0u][cx + 1u]);
    triangle_normal(p_tile[cy - 0u][cx + 1u], p_tile[cy + 1u][cx + 1u]);
    triangle_normal(p_tile[cy + 1u][cx + 1u], p_tile[cy + 1u][cx + 0u]);
    triangle_normal(p_tile[cy + 1u][cx + 0u], p_tile[cy + 1u][cx - 1u]);
    triangle_normal(p_tile[cy + 1u][cx - 1u], p_tile[cy + 0u][cx - 1u]);
    triangle_normal(p_tile[cy + 0u][cx - 1u], p_tile[cy - 1u][cx - 1u]);

    // 4 Triangle normal accumulation
    // triangle_normal(p_tile[cy - 1u][cx], p_tile[cy][cx - 1u]);
    // triangle_normal(p_tile[cy][cx - 1u], p_tile[cy + 1u][cx]);
    // triangle_normal(p_tile[cy + 1u][cx], p_tile[cy][cx + 1u]);
    // triangle_normal(p_tile[cy][cx + 1u], p_tile[cy - 1u][cx]);

    accum_norm = accum_norm;

    if (length(accum_norm) < EPSIL) {
        return;
    }

    let norm = normalize(accum_norm);
    points.data[row_o * out_w + col_o].normal = vec4<f32>(norm, 0.0);
    points.data[row_o * out_w + col_o].force = points.data[row_o * out_w + col_o].position + vec4<f32>(norm, 0.0);
}

struct DT {
    value: f32,
};

@group(2) @binding(0) var<uniform> dt: DT;

@compute @workgroup_size(256)
fn update(
  @builtin(workgroup_id)        blockIdx :  vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
    let offset = i32(blockIdx.x * 256u + threadIdx.x);
    
    let out_w = i32(dimension.size.x);
    let out_h = i32(dimension.size.y);

    if (offset >= out_w * out_w) {
        return;
    }

    let p = points.data[offset];

    if (p.position.w < 0.0) {
        points.data[offset].velocity = vec4<f32>(0.0);
        // Not necessary but...
        points.data[offset].force = vec4<f32>(0.0);
        return;
    }

    let a = p.force / constants.mass;
    points.data[offset].velocity = p.velocity + a * dt.value;
    let pos = p.position + p.velocity * dt.value;
    points.data[offset].position = vec4<f32>(pos.x, max(pos.y, constants.floor_y), pos.zw);
}