struct Uniforms {
    modelMatrix : mat4x4<f32>;
    normalModelMatrix : mat4x4<f32>;
};

struct Camera {
    viewProjectionMatrix : mat4x4<f32>;
    eyePos : vec3<f32>;
};

struct Sphere {
    radius : f32;
    num : u32;
};

struct Particle {
    position: vec3<f32>;
};

struct Fluid {
    data: array<Particle>;
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(1) @binding(0) var<uniform> camera : Camera;
@group(2) @binding(0) var<uniform> sphere : Sphere;
@group(2) @binding(1) var<storage, read> fluid : Fluid;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>;
    @location(0) sphere_center : vec3<f32>;  // sphere center in world space
    @location(1) ray_dir : vec3<f32>;  // ray direction
};

@stage(vertex)
fn vert_main(
    @builtin(vertex_index) VertexIndex : u32,
    @builtin(instance_index) InstanceIndex : u32) -> VertexOutput {

    var cube_pos = array<vec3<f32>, 36>(
        vec3<f32>(-0.5, 0.5, 0.5),
        vec3<f32>(-0.5, -0.5, 0.5),
        vec3<f32>(0.5, -0.5, 0.5),
        vec3<f32>(-0.5, 0.5, 0.5),
        vec3<f32>(0.5, -0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, -0.5, 0.5),
        vec3<f32>(0.5, -0.5, -0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(0.5, -0.5, -0.5),
        vec3<f32>(0.5, 0.5, -0.5),
        vec3<f32>(0.5, -0.5, 0.5),
        vec3<f32>(-0.5, -0.5, 0.5),
        vec3<f32>(-0.5, -0.5, -0.5),
        vec3<f32>(0.5, -0.5, 0.5),
        vec3<f32>(-0.5, -0.5, -0.5),
        vec3<f32>(0.5, -0.5, -0.5),
        vec3<f32>(0.5, 0.5, -0.5),
        vec3<f32>(-0.5, 0.5, -0.5),
        vec3<f32>(-0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, -0.5),
        vec3<f32>(-0.5, 0.5, 0.5),
        vec3<f32>(0.5, 0.5, 0.5),
        vec3<f32>(-0.5, -0.5, -0.5),
        vec3<f32>(-0.5, 0.5, -0.5),
        vec3<f32>(0.5, 0.5, -0.5),
        vec3<f32>(-0.5, -0.5, -0.5),
        vec3<f32>(0.5, 0.5, -0.5),
        vec3<f32>(0.5, -0.5, -0.5),
        vec3<f32>(-0.5, 0.5, -0.5),
        vec3<f32>(-0.5, -0.5, -0.5),
        vec3<f32>(-0.5, -0.5, 0.5),
        vec3<f32>(-0.5, 0.5, -0.5),
        vec3<f32>(-0.5, -0.5, 0.5),
        vec3<f32>(-0.5, 0.5, 0.5)
    );

    let center = fluid.data[InstanceIndex].position;
    let world_pos = (uniforms.modelMatrix * vec4<f32>(center + cube_pos[VertexIndex] * sphere.radius, 1.0)).xyz;

    var output : VertexOutput;
    output.sphere_center = (uniforms.modelMatrix * vec4<f32>(center, 1.0)).xyz;
    output.ray_dir = world_pos - camera.eyePos;
    output.Position = camera.viewProjectionMatrix * vec4<f32>(world_pos, 1.0);

    return output;
}

struct GBufferOutput {
    @location(0) position : vec4<f32>;
    @location(1) normal : vec4<f32>;
    // Textures: diffuse color, specular color, smoothness, emissive etc. could go here
    @location(2) albedo : vec4<f32>;
};

fn sphIntersect(ro: vec3<f32>, rd: vec3<f32>, sph: vec4<f32>) -> f32 {
    let oc = ro - sph.xyz;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - sph.w * sph.w;
    let h = b * b - c;
    
    return select(- b - sqrt(h), -1.0, h < 0.0);
}

@stage(fragment)
fn frag_main(
        @location(0) sphere_center: vec3<f32>,
        @location(1) ray_dir: vec3<f32>) -> GBufferOutput {

    let rd = normalize(ray_dir);
    let hit = sphIntersect(camera.eyePos, rd, vec4<f32>(sphere_center, sphere.radius));

    var output : GBufferOutput;

    if (hit < 0.0) {
        output.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        output.normal   = vec4<f32>(1.0, 0.0, 0.0, 1.0);

        // TODO
        let c = 1.0;
        output.albedo = vec4<f32>(c, c, c, 1.0);
        return output;
    }

    let world_pos = rd * hit + camera.eyePos;

    output.position = vec4<f32>(world_pos, 1.0);
    output.normal   = vec4<f32>(normalize(world_pos - sphere_center), 1.0);

    // TODO
    let c = 0.2;
    output.albedo = vec4<f32>(c, c, c, 1.0);
    return output;
}