struct Uniforms {
    modelMatrix : mat4x4<f32>;
    normalModelMatrix : mat4x4<f32>;
};

struct Camera {
    viewProjectionMatrix : mat4x4<f32>;
    eyePos : vec3<f32>;
};

struct Params {
    radius : f32;
    num : u32;
    spawn : u32;
    gravity : f32;
    air_den : f32;
    drag : f32;
    elas : f32;
    fric : f32;
    wind : vec3<f32>;
};

struct RenderParticle {
    position: vec3<f32>;
    velocity: vec3<f32>;
};

struct RenderParticles {
    data: array<RenderParticle>;
};

struct Indices {
    data: array<u32>;
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(1) @binding(0) var<uniform> camera : Camera;
@group(2) @binding(0) var<uniform> params : Params;
@group(2) @binding(1) var<storage, read> r_particles : RenderParticles;
@group(2) @binding(2) var<storage, read> indices : Indices;

struct VertexOutput {
    @builtin(position) Position : vec4<f32>;
    @location(0) sphere_center : vec3<f32>;  // sphere center in world space
    @location(1) ray_dir : vec3<f32>;        // ray direction
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

    let p = r_particles.data[indices.data[InstanceIndex]].position;
    let center = vec3<f32>(p[0], p[1], p[2]);
    let world_pos = (uniforms.modelMatrix * vec4<f32>(center + cube_pos[VertexIndex] * params.radius * 2.0, 1.0)).xyz;

    var output : VertexOutput;
    output.sphere_center = (uniforms.modelMatrix * vec4<f32>(center, 1.0)).xyz;
    output.ray_dir = world_pos - camera.eyePos;
    output.Position = camera.viewProjectionMatrix * vec4<f32>(world_pos, 1.0);

    return output;
}

struct GBufferOutput {
    @builtin(frag_depth) depth : f32;
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
    
    return select(- b - sqrt(h), 0.0, h < 0.0);
}

@stage(fragment)
fn frag_main(
        @location(0) sphere_center: vec3<f32>,
        @location(1) ray_dir: vec3<f32>) -> GBufferOutput {

    let rd = normalize(ray_dir);
    let hit = sphIntersect(camera.eyePos, rd, vec4<f32>(sphere_center, params.radius));

    var output : GBufferOutput;

    if (hit <= 0.0) {
        discard;
    }

    let world_pos = rd * hit + camera.eyePos;

    output.position = vec4<f32>(world_pos, 1.0);
    output.normal   = vec4<f32>(normalize(world_pos - sphere_center), 1.0);

    // TODO
    let c = 0.5;
    output.albedo = vec4<f32>(c, c, c, 1.0);

    let clipPos = camera.viewProjectionMatrix * vec4<f32>(world_pos, 1.0);
    output.depth = clipPos.z / clipPos.w;

    return output;
}

struct StageParticle {
    position: vec4<f32>;
    velocity: vec3<f32>;
};

struct StageParticles {
    data: array<StageParticle>;
};

struct DT {
    value: f32;
};

@group(2) @binding(1) var<storage, read_write> w_particles : RenderParticles;
@group(2) @binding(3) var<storage, read> stage : StageParticles;

@stage(compute) @workgroup_size(256)
fn spawn(
    @builtin(workgroup_id)        blockIdx :  vec3<u32>,
    @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
    let index = blockIdx.x * 256u + threadIdx.x;
    if (index >= params.spawn) {
        return;
    }

    let particle = stage.data[index];
    let write_index = u32(particle.position.w);
    w_particles.data[write_index].position = particle.position.xyz;
    w_particles.data[write_index].velocity = particle.velocity;
}

@group(3) @binding(0) var<uniform> dt : DT;

let pi = 3.1415926;
let epsi = 0.0001;

@stage(compute) @workgroup_size(256)
fn update(
    @builtin(workgroup_id)        blockIdx :  vec3<u32>,
    @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
    let index = blockIdx.x * 256u + threadIdx.x;
    if (index >= params.num) {
        return;
    }

    let i = indices.data[index];
    let p = w_particles.data[i];

    // mass = 1
    var force = vec3<f32>(0.0, params.gravity, 0.0);
    var v = p.velocity;
    let v_close = v - params.wind;
    force = force - 0.5 * params.air_den * length(v_close) * v_close *
        params.drag * pi * params.radius * params.radius;

    let pos = p.position + v * dt.value;

    // Ground collision
    if (pos.y < 0.0) {
        let vy = v.y;
        v = vec3<f32>(v.x, 0.0, v.z);
        var vlen = length(v);
        if (vlen > epsi) {
            v = v / vlen;
            let e = abs((1.0 + params.elas) * vy);
            let vlen = max(0.0, vlen - params.fric * e);
            v = v * vlen;
        }
        v = vec3<f32>(v.x, abs(vy) * params.elas, v.z);
    }

    w_particles.data[i].velocity = v + force * dt.value;
    w_particles.data[i].position = vec3<f32>(pos.x, max(pos.y, 0.0), pos.z);
}