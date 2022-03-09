@group(0) @binding(0) var gBufferPosition: texture_2d<f32>;
@group(0) @binding(1) var gBufferNormal: texture_2d<f32>;
@group(0) @binding(2) var gBufferAlbedo: texture_2d<f32>;

struct LightData {
    position : vec4<f32>;
    color : vec3<f32>;
    radius : f32;
};

struct LightsBuffer {
    lights: array<LightData>;
};
@group(1) @binding(0) var<storage, read_write> lightsBuffer: LightsBuffer;

struct Config {
    numLights : u32;
};
@group(1) @binding(1) var<uniform> config: Config;

struct CanvasConstants {
    size: vec2<f32>;
};
@group(2) @binding(0) var<uniform> canvas : CanvasConstants;

@stage(fragment)
fn main(@builtin(position) coord : vec4<f32>)
     -> @location(0) vec4<f32> {

    var result = vec3<f32>(0.0, 0.0, 0.0);

    let position = textureLoad(
        gBufferPosition,
        vec2<i32>(floor(coord.xy)),
        0
    );

    if (position.w > 10000.0) {
        discard;
    }

    let normal = textureLoad(
        gBufferNormal,
        vec2<i32>(floor(coord.xy)),
        0
    ).xyz;

    let albedo = textureLoad(
        gBufferAlbedo,
        vec2<i32>(floor(coord.xy)),
        0
    ).rgb;

    for (var i : u32 = 0u; i < config.numLights; i = i + 1u) {
        let dist = lightsBuffer.lights[i].position.xyz - position.xyz;
        let distance = length(dist);
        if (distance > lightsBuffer.lights[i].radius) {
            continue;
        }
        let lambert = max(dot(normal, normalize(dist)), 0.0);
        result = result + vec3<f32>(
        lambert * pow(1.0 - distance / lightsBuffer.lights[i].radius, 2.0) * lightsBuffer.lights[i].color * albedo);
    }

    // some manual ambient
    result = result + vec3<f32>(0.2, 0.2, 0.2);

    // return vec4<f32>(result, 1.0);

    return vec4<f32>(((normal + vec3<f32>(1.0)) / 2.0), 1.0);
    // return vec4<f32>(((position.xyz) + 0.5), 1.0);
}
