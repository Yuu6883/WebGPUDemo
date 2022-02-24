struct ClothPoint {
    position: vec4<f32>; // using w component to store if this point is fixed (padded anyways)
    normal:   vec4<f32>;
    velocity: vec4<f32>;
    force:    vec4<f32>;
};

struct ClothPointShared {
    p: vec4<f32>;
    v: vec4<f32>;
};

[[block]] struct ClothPoints {
    data: array<ClothPoint>;
};

[[block]] struct SimulationConstants {
    mass:             f32;
    rest_length:      f32;
    spring_constant:  f32;
    damping_constant: f32;
};

[[block]] struct Dimension {
    size: vec2<u32>;
};

[[block]] struct Vectors {
    wind: vec3<f32>;
    gravity: vec3<f32>;
};

[[block]] struct Indices {
    data: [[stride(12)]] array<array<u32, 3>>;
};

// Group 1 is per cloth
[[group(1), binding(0)]] var<uniform> constants: SimulationConstants;
[[group(1), binding(1)]] var<uniform> dimension: Dimension;
[[group(1), binding(2)]] var<uniform> vectors: Vectors;

var<workgroup> tile : array<array<ClothPointShared, 16>, 16>;

var<private> force: vec3<f32>;
var<private> p1: vec3<f32>;
var<private> v1: vec3<f32>;

let SQRT2 = 1.4142135623730951;

fn spring_damper(p2: vec4<f32>, v2: vec4<f32>, rest_length: f32) {
    let delta = p2.xyz - p1;
    let dir = normalize(delta);

    // Spring force
    force = force + constants.spring_constant * (length(delta) - rest_length) * dir;

    // Damper force
    let v_close = dot(v1 - v2.xyz, dir);
    force = force - constants.damping_constant * v_close * dir;
}

let TILE_SIZE = 16;
let TILE_SIZE_U = 16u;
let INNER_TILE = 14u;

[[group(0), binding(0)]] var<storage, read_write> indices: Indices;

[[stage(compute), workgroup_size(TILE_SIZE, TILE_SIZE, 1)]]
fn init_indices(
  [[builtin(workgroup_id)]]        blockIdx :  vec3<u32>,
  [[builtin(local_invocation_id)]] threadIdx : vec3<u32>
) {
    let tx = threadIdx.x;
    let ty = threadIdx.y;

    let row = blockIdx.y * TILE_SIZE_U + ty;
    let col = blockIdx.x * TILE_SIZE_U + tx;

    let out_w = dimension.size.x;
    let out_h = dimension.size.y;

    if (row >= out_h - 1u && col >= out_w - 1u) {
        return;
    }

    let offset = (row * out_w + col) * 2u;

    // Triangle 1
    indices.data[offset][0] = row * out_w + col;
    indices.data[offset][1] = (row + 1u) * out_w + col;
    indices.data[offset][2] = row * out_w + col + 1u;
    
    // Triangle 2
    indices.data[offset + 1u][0] = row * out_w + col + 1u;
    indices.data[offset + 1u][1] = (row + 1u) * out_w + col;
    indices.data[offset + 1u][2] = (row + 1u) * out_w + col + 1u;
}

[[group(0), binding(0)]] var<storage, read_write> points: ClothPoints;

[[stage(compute), workgroup_size(TILE_SIZE, TILE_SIZE, 1)]]
fn calc_forces(
  [[builtin(workgroup_id)]]        blockIdx :  vec3<u32>,
  [[builtin(local_invocation_id)]] threadIdx : vec3<u32>
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

    // Out of grid || out of tile || fixed point
    if (row_o >= out_h || col_o >= out_w || 
        tx >= INNER_TILE || ty >= INNER_TILE ||
        tile[ty + 1u][tx + 1u].p.w < 0.0) {
        return;
    }

    let cx = tx + 1u;
    let cy = ty + 1u;

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
    
    points.data[row_o * out_w + col_o].force = vec4<f32>(force, 0.0);
}

[[block]] struct DT {
    value: f32;
};

[[group(2), binding(0)]] var<uniform> dt: DT;

[[stage(compute), workgroup_size(256)]]
fn update(
  [[builtin(workgroup_id)]]        blockIdx :  vec3<u32>,
  [[builtin(local_invocation_id)]] threadIdx : vec3<u32>
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
    points.data[offset].position = p.position + p.velocity * dt.value;
}