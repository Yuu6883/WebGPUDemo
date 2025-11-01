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

struct LightInfo {
    color: vec3<f32>,
    direction: vec3<f32>,
};

@group(1) @binding(0) var u_texture: texture_2d<f32>;
@group(1) @binding(1) var u_sampler: sampler;
@group(1) @binding(2) var<uniform> lights: array<LightInfo, 512>;
@group(1) @binding(3) var<storage, read> sprite_params: array<SpriteInfo>;

struct FSInput {
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) state: u32,
    @location(3) @interpolate(flat) rotation: f32
};

struct FSOutput {
    @location(0) color: vec4<f32>,
};

fn adjust_vibrance(color: vec3<f32>, vibrance: f32) -> vec3<f32> {
    // Compute perceived luminance
    let gray = dot(color, vec3<f32>(0.299, 0.587, 0.114));

    // Measure saturation (0 = gray, 1 = fully saturated)
    let sat = distance(color, vec3<f32>(gray));

    // Boost muted colors more than saturated ones
    let boost = vibrance * (1.0 - sat);

    return mix(vec3<f32>(gray), color, 1.0 + boost);
}

fn adjust_saturation(color: vec3<f32>, saturation: f32) -> vec3<f32> {
    // Compute luminance using perceptual weights
    let gray = dot(color, vec3<f32>(0.299, 0.587, 0.114));
    return mix(vec3<f32>(gray), color, saturation);
}

@fragment
fn main(in: FSInput) -> FSOutput {
    var out: FSOutput;

    var uv_x: f32;
    var uv_y: f32;

    let id = in.state >> 16u;
    let seed = in.state & 0xFFFFu;

    let info = sprite_params[id];

    let albedo_offset = info.variant_offset + (seed % sprite_params[id].variant_count) * 3u;
    uv_x = in.uv.x / 16.0 + f32(albedo_offset % 16u) / 16.0;
    uv_y = in.uv.y / 16.0 + f32(albedo_offset / 16u) / 16.0;
    let uv_albedo = vec2<f32>(uv_x, uv_y);

    let normal_offset = albedo_offset + 1u;
    uv_x = in.uv.x / 16.0 + f32(normal_offset % 16u) / 16.0;
    uv_y = in.uv.y / 16.0 + f32(normal_offset / 16u) / 16.0;
    let uv_normal = vec2<f32>(uv_x, uv_y);

    let roughness_offset = albedo_offset + 2u;
    uv_x = in.uv.x / 16.0 + f32(roughness_offset % 16u) / 16.0;
    uv_y = in.uv.y / 16.0 + f32(roughness_offset / 16u) / 16.0;
    let uv_roughness = vec2<f32>(uv_x, uv_y);


    // --- sample textures ---
    let albedo = textureSample(u_texture, u_sampler, uv_albedo);
    let normal = textureSample(u_texture, u_sampler, uv_normal);
    let roughness = textureSample(u_texture, u_sampler, uv_roughness);

    // decode normal
    let n_tangent = normalize(normal.rgb * 2.0 - 1.0);

    // build 2D rotation around Z
    let c = cos(in.rotation);
    let s = sin(in.rotation);
    let N_world = normalize(vec3<f32>(
        n_tangent.x * c + n_tangent.y * s,
        -n_tangent.x * s + n_tangent.y * c,
        n_tangent.z
    ));
    let N = mix(vec3<f32>(0.0, 0.0, 1.0), N_world, info.normal_strength);

    // --- combine lighting ---
    var color = albedo.rgb * info.ambient_light;           // ambient

    // let L = normalize(vec3<f32>(-1.0, -1.0, 2.0));

    // let V = vec3<f32>(0.0, 0.0, 1.0);
    // let H = normalize(L + V);

    // let NdotL = max(dot(N, L), 0.1);
    // let NdotH = max(dot(N, H), 0.0);

    //     // --- Blinn-Phong specular ---
    // let spec = info.specular_strength * pow(NdotH, (1.0 - roughness.a) * info.specular_power);

    // color = color + albedo.rgb * NdotL; // + spec * info.specular_purity;

    var diffuse_accum = vec3<f32>(0.0);
    var spec_accum = vec3<f32>(0.0);

    for(var i: u32 = info.light_offset; i < info.light_offset + info.light_count; i = i + 1u) {
        let light = lights[i];
        let L = normalize(light.direction);

        let V = vec3<f32>(0.0, 0.0, 1.0);
        let H = normalize(L + V);

        let NdotL = max(dot(N, L), 0.1);
        let NdotH = max(dot(N, H), 0.0);

            // --- Blinn-Phong specular ---
        let spec = pow(NdotH, (1.0 - roughness.a) * info.specular_power);

        diffuse_accum = diffuse_accum + albedo.rgb * NdotL * light.color * 2.1; // diffuse
        spec_accum = spec_accum + spec;
    }

    // let light_norm = 1.0 / f32(max(info.light_count, 1u));
    // diffuse_accum *= light_norm;
    // spec_accum *= light_norm;
    color = color + diffuse_accum + roughness.rgb * roughness.a;
    color = color * info.brightness;

    // color = color + diffuse_accum; //  + spec_accum * info.specular_purity + roughness.rgb * (1.0 - roughness.a);

    // // --- apply sprite brightness ---
    // color = color * info.brightness;

    // // --- optional simple subsurface scatter effect ---
    // color = mix(color, color * roughness.rgb, info.sss_amount * info.sss_contrast);

    // out.color = vec4<f32>(color, albedo.a);
    // out.color = vec4<f32>(NdotL, NdotL, NdotL, albedo.a);
    // let sp =  spec * spec_tint;
    // out.color = vec4<f32>(0, select(1.0, 0.0, info.specular_power <= 0.0), 0.0, albedo.a);
    // color = roughness.rgb;
    
    color = adjust_vibrance(color, info.light_width);
    // color = adjust_saturation(color, 1.0 + info.light_width);
    // color = color / (color + vec3(1.0));
    // color = pow(color, vec3(1.0 / 2.2));
    out.color = vec4<f32>(color, albedo.a);
    return out;
}