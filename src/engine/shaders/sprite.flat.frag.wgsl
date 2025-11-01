// unlit.frag.wgsl

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

@group(1) @binding(0) var u_texture: texture_2d<f32>;
@group(1) @binding(1) var u_sampler: sampler;
@group(1) @binding(3) var<storage, read> sprite_params: array<SpriteInfo>;

struct FSInput {
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) state: u32,
};

struct FSOutput {
    @location(0) color: vec4<f32>,
};

@fragment
fn main(in: FSInput) -> FSOutput {
    var out: FSOutput;

    var uv_x: f32;
    var uv_y: f32;

    let id = in.state >> 16u;
    let seed = in.state & 0xFFFFu;

    let info = sprite_params[id];

    let albedo_offset = info.variant_offset + (seed % sprite_params[id].variant_count);
    uv_x = in.uv.x / 16.0 + f32(albedo_offset % 16u) / 16.0;
    uv_y = in.uv.y / 16.0 + f32(albedo_offset / 16u) / 16.0;
    let uv_albedo = vec2<f32>(uv_x, uv_y);

    out.color = textureSample(u_texture, u_sampler, uv_albedo);
    return out;
}