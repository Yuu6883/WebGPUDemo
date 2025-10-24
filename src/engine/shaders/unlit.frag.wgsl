// unlit.frag.wgsl
@group(0) @binding(0)
var u_texture: texture_2d<f32>;

@group(0) @binding(1)
var u_sampler: sampler;

struct FSInput {
    @location(0) uv_scale: vec2<f32>,
    @location(1) @interpolate(flat) state: u32
};

struct FSOutput {
    @location(0) color: vec4<f32>,
};

@fragment
fn main(in: FSInput) -> FSOutput {
    var out: FSOutput;

    let colors = array<vec4<f32>, 5>(
        vec4<f32>(1.0, 1.0, 1.0, 1.0),
        vec4<f32>(1.0, 0.0, 0.0, 1.0),
        vec4<f32>(0.0, 1.0, 0.0, 1.0),
        vec4<f32>(0.0, 0.0, 1.0, 1.0),
        vec4<f32>(0.0, 1.0, 1.0, 1.0)
    );

    out.color = colors[in.state >> 16];
    return out;
}