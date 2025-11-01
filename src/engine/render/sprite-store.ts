import { checkDevice, GDevice } from './base';
import SpriteParam from './sprite-param';
import TextureAtlas from './texture';

export default class SpriteStore {
    // buffers
    public readonly lightBuffer: GPUBuffer;
    public readonly spriteParamsBuffer: GPUBuffer;
    public readonly camUB: GPUBuffer;
    public readonly tickBuffer: GPUBuffer;

    public readonly spriteSampler: GPUSampler;
    public readonly spriteTexture: GPUTexture;

    readonly spriteParamAB = new ArrayBuffer(SpriteParam.BYTES * 65536);
    readonly spriteParams: SpriteParam[] = Array.from(
        { length: 65536 },
        (_, i) =>
            new SpriteParam(new DataView(this.spriteParamAB, i * SpriteParam.BYTES)),
    );

    constructor() {
        checkDevice();
        const device = GDevice.device;

        this.lightBuffer = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT * 512 * 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.spriteParamsBuffer = device.createBuffer({
            label: 'Sprite Params',
            size: 65536 * SpriteParam.BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.camUB = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT * 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.tickBuffer = device.createBuffer({
            size: Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.spriteTexture = device.createTexture({
            size: [TextureAtlas.DIM, TextureAtlas.DIM, 1],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.spriteSampler = device.createSampler({
            magFilter: 'nearest',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            addressModeW: 'clamp-to-edge',
        });
    }

    updateSpriteTexture(atlas: TextureAtlas) {
        const device = GDevice.device;
        const queue = device.queue;
        const texture = this.spriteTexture;
        queue.copyExternalImageToTexture(
            { source: atlas.scratch },
            { texture, mipLevel: 0, origin: [0, 0, 0] },
            [texture.width, texture.height, 1],
        );

        let totalLights = 0;
        for (let i = 0; i < atlas.entries.length; i++) {
            const param = this.spriteParams[i];
            const entry = atlas.entries[i];

            param.variant_offset = entry.offset;
            param.variant_count = entry.variant;
            param.uv_scale = [1, 1];

            if (entry.flat) {
                param.flags = 1;
                param.size = [1, 1];
                continue;
            }

            param.size = [0.75 / 2, 0.75 / 2];
            param.normal_strength = entry.normal_strength;
            param.light_width = entry.light_width;
            param.brightness = entry.brightness;
            param.specular_strength = entry.specular_strength;
            param.specular_power = entry.specular_power;
            param.specular_purity = entry.specular_purity;
            param.sss_contrast = entry.sss_contrast;
            param.sss_amount = entry.sss_amount;
            param.ambient_light = entry.ambient_light;
            // param.ambient_light = [0.5, 0.5, 0.5];
            param.flags = 0;

            totalLights += entry.lights.length;
        }

        console.log(atlas.entries);

        const lights = new Float32Array(totalLights * 8);
        let lightOffset = 0;

        for (let i = 0; i < atlas.entries.length; i++) {
            const entry = atlas.entries[i];
            if (entry.flat) continue;

            const param = this.spriteParams[i];

            for (let j = 0; j < entry.lights.length; j++) {
                const light = entry.lights[j];
                const offset = (lightOffset + j) * 8;
                lights[offset + 0] = light.color[0];
                lights[offset + 1] = light.color[1];
                lights[offset + 2] = light.color[2];
                lights[offset + 3] = light.direction[0];
                lights[offset + 4] = light.direction[1];
                lights[offset + 5] = light.direction[2];
            }
            param.light_offset = lightOffset;
            param.light_count = entry.lights.length;

            lightOffset += entry.lights.length;
        }

        queue.writeBuffer(this.lightBuffer, 0, lights);
        queue.writeBuffer(this.spriteParamsBuffer, 0, this.spriteParamAB);
    }

    updateTick(tick: number) {
        if (!this.tickBuffer) return;

        const device = GDevice.device;
        const queue = device.queue;
        queue.writeBuffer(this.tickBuffer, 0, new Uint32Array([tick]));
    }

    updateCam(camBuf: Float32Array) {
        if (!this.camUB) return;

        const device = GDevice.device;
        const queue = device.queue;
        queue.writeBuffer(this.camUB, 0, camBuf.buffer);
    }
}
