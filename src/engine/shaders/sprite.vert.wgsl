// sprite.vert.wgsl
@group(0) @binding(0) var<storage, read> pos_x: array<i32>;
@group(0) @binding(1) var<storage, read> pos_y: array<i32>;
@group(0) @binding(2) var<storage, read> state: array<u32>;
@group(0) @binding(3) var<storage, read> sprite_params: array<SpriteInfo>;
@group(0) @binding(4) var<uniform> camera: Camera;
@group(0) @binding(5) var<uniform> tick: u32;


struct SpriteInfo {
    size: vec2<f32>,  // width, height in world units
    uv_scale: vec2<f32>  // uv scale
};

struct Camera {
    viewProjectionMatrix : mat4x4<f32>,
};

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv_scale: vec2<f32>,
    @location(1) @interpolate(flat) state: u32,
};

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

    let local = quad_verts[VertexIndex];
    let world = vec2<f32>(sx, sy) + (local - 0.5) * info.size;

    out.position = camera.viewProjectionMatrix * vec4<f32>(world, 0.0, 1.0);
    out.uv_scale = info.uv_scale;
    out.state = pack;

    return out;
}
