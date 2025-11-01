// sprite.vert.wgsl
@group(0) @binding(0) var<storage, read> pos_x: array<i32>;
@group(0) @binding(1) var<storage, read> pos_y: array<i32>;
@group(0) @binding(2) var<storage, read> state: array<u32>;

@group(1) @binding(3) var<storage, read> sprite_params: array<SpriteInfo>;
@group(1) @binding(4) var<uniform> camera: Camera;
@group(1) @binding(5) var<uniform> tick: u32;

struct SpriteInfo {
    size: vec2<f32>,  // width, height in world units
    uv_scale: vec2<f32>,  // uv scale
    variant_offset: u32,
    variant_count: u32,
    normal_strength: f32,
    light_width: f32,
    brightness: f32,
    specular_strength: f32,
    specular_power: f32,
    specular_purity: f32,
    sss_contrast: f32,
    sss_amount: f32,
    light_offset: u32,
    light_count: u32,
    ambient_light: vec3<f32>,
    flags: u32,
};

struct Camera {
    viewProjectionMatrix : mat4x4<f32>,
};

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) state: u32,
    @location(3) @interpolate(flat) rotation: f32
};

fn rng_next(state: ptr<function, u32>) -> f32 {
    // Simple xorshift32
    var x = *state;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    *state = x;
    return f32(x & 0x00FFFFFFu) / f32(0x01000000u);
}

@vertex
fn main(
    @builtin(vertex_index) VertexIndex : u32, 
    @builtin(instance_index) InstanceIndex: u32) -> VSOut {

    var out: VSOut;

    // Lookup sprite data
    let sx = f32(pos_x[InstanceIndex]) / f32(1 << 11);
    let sy = f32(pos_y[InstanceIndex]) / f32(1 << 11);
    let pack = state[InstanceIndex];

    let id = pack >> 16u;
    let info = sprite_params[id];

    // 0–5 vertices → two triangles
    let quad_verts = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0)
    );

    let uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0)
    );

    var local = quad_verts[VertexIndex] - 0.5;

    var seed = pack & 0xFFFFu;
    let init_rot = rng_next(&seed) * 6.283185307179586;
    let curr_rot = (rng_next(&seed) - 0.5) * f32(tick % 16777216u) * 0.01;

    let rot = init_rot + curr_rot;
    local = vec2<f32>(
        local.x * cos(rot) - local.y * sin(rot),
        local.x * sin(rot) + local.y * cos(rot)
    );

    let world = vec2<f32>(sx, sy) + local * info.size * f32(1u - ((pack >> 15u) & 1u));

    out.position = camera.viewProjectionMatrix * vec4<f32>(world, 0.0, 1.0);
    out.uv = uvs[VertexIndex];
    out.state = pack;
    out.rotation = rot;

    return out;
}
