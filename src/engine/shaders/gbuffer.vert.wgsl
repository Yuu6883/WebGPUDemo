struct Uniforms {
    modelMatrix : mat4x4<f32>;
    normalModelMatrix : mat4x4<f32>;
};

struct Camera {
    viewProjectionMatrix : mat4x4<f32>;
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(1) @binding(0) var<uniform> camera : Camera;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>;
    @location(0) worldPos: vec3<f32>;  // position in world space
    @location(1) worldNorm: vec3<f32>;    // normal in world space
    @location(2) fragUV: vec2<f32>;
};

@stage(vertex)
fn main(@location(0) position : vec3<f32>,
        @location(1) normal : vec3<f32>,
        @location(2) uv : vec2<f32>) -> VertexOutput {
    var output : VertexOutput;
    output.worldPos = (uniforms.modelMatrix * vec4<f32>(position, 1.0)).xyz;
    output.Position = camera.viewProjectionMatrix * vec4<f32>(output.worldPos, 1.0);
    output.worldNorm = normalize((uniforms.normalModelMatrix * vec4<f32>(normal, 1.0)).xyz);
    output.fragUV = uv;
    return output;
}