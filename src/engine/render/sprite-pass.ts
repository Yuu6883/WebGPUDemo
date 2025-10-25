import Renderer, { checkDevice, GDevice } from './base';

import SpriteVertWGSL from '../shaders/sprite.vert.wgsl'; // reuse your vertex shader
import SpriteFragWGSL from '../shaders/unlit.frag.wgsl'; // new forward fragment shader (per-material lighting)
import { RenderPass } from './interfaces';
import TextureAtlas from './texture';
import SpriteParam from './sprite-param';

export class SpritePass implements RenderPass {
    private camUB: GPUBuffer;

    private basePipeline: GPURenderPipeline;
    private vertModule: GPUShaderModule;
    private fragModule: GPUShaderModule;
    private viewGroup: GPUBindGroup;
    private samplerGroup: GPUBindGroup;

    readonly spriteParamAB = new ArrayBuffer(SpriteParam.BYTES * 65536);
    readonly spriteParams: SpriteParam[] = Array.from(
        { length: 65536 },
        (_, i) =>
            new SpriteParam(new DataView(this.spriteParamAB, i * SpriteParam.BYTES)),
    );

    private drawNum = 0;
    private posXBuffer: GPUBuffer;
    private posYBuffer: GPUBuffer;
    private stateBuffer: GPUBuffer;
    private spriteParamsBuffer: GPUBuffer;
    private tickBuffer: GPUBuffer;
    private lightBuffer: GPUBuffer;

    private spriteTexture: GPUTexture;

    async init() {
        checkDevice();
        if (this.basePipeline) return;
        const device = GDevice.device;

        this.camUB = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT * 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // create shaders
        this.vertModule = device.createShaderModule({ code: SpriteVertWGSL });
        this.fragModule = device.createShaderModule({ code: SpriteFragWGSL });

        // pipeline
        this.basePipeline = await device.createRenderPipelineAsync({
            layout: 'auto',
            vertex: {
                module: this.vertModule,
                entryPoint: 'main',
                buffers: [],
            },
            fragment: {
                module: this.fragModule,
                entryPoint: 'main',
                targets: [
                    {
                        format: GDevice.format,
                        blend: {
                            color: {
                                srcFactor: 'src-alpha',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                            alpha: {
                                srcFactor: 'one',
                                dstFactor: 'one-minus-src-alpha',
                                operation: 'add',
                            },
                        },
                    },
                ],
            },
            depthStencil: {
                depthWriteEnabled: false,
                depthCompare: 'less',
                format: 'depth24plus-stencil8',
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'none',
            },
        });

        this.spriteParamsBuffer = device.createBuffer({
            label: 'Sprite Params',
            size: 65536 * SpriteParam.BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.tickBuffer = device.createBuffer({
            size: Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.lightBuffer = device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT * 512 * 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.expandBuffers();

        this.spriteTexture = device.createTexture({
            size: [TextureAtlas.DIM, TextureAtlas.DIM, 1],
            format: 'rgba8unorm',
            usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });

        const sampler = device.createSampler({
            magFilter: 'nearest',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            addressModeW: 'clamp-to-edge',
        });

        this.samplerGroup = device.createBindGroup({
            label: 'Sprite Sampler+Texture',
            layout: this.basePipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: this.spriteTexture.createView(),
                },
                {
                    binding: 1,
                    resource: sampler,
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.lightBuffer,
                    },
                },
            ],
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

            param.size = [0.75 / 2, 0.75 / 2];
            param.uv_scale = [1, 1];
            param.variant_offset = entry.offset;
            param.variant_count = entry.variant;
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

            totalLights += entry.lights.length;
        }

        const lights = new Float32Array(totalLights * 8);
        let lightOffset = 0;

        for (let i = 0; i < atlas.entries.length; i++) {
            const entry = atlas.entries[i];
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

    expandBuffers() {
        if (!this.basePipeline) return;
        const device = GDevice.device;

        const bufferSize =
            (Math.floor(this.drawNum / 4096) + 1) * 4096 * Int32Array.BYTES_PER_ELEMENT; // 4 bytes per sprite for i32/u32

        if (bufferSize <= this.posXBuffer?.size) return;
        console.log(
            `Expanding buffer size to ${(bufferSize / 1024 / 1024).toFixed(3)}MB`,
        );

        this.posXBuffer?.destroy();
        this.posYBuffer?.destroy();
        this.stateBuffer?.destroy();

        this.posXBuffer = device.createBuffer({
            label: 'Sprite X Positions',
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.posYBuffer = device.createBuffer({
            label: 'Sprite Y Positions',
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.stateBuffer = device.createBuffer({
            label: 'Sprite States',
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // bind group
        this.viewGroup = device.createBindGroup({
            layout: this.basePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.posXBuffer } },
                { binding: 1, resource: { buffer: this.posYBuffer } },
                { binding: 2, resource: { buffer: this.stateBuffer } },
                { binding: 3, resource: { buffer: this.spriteParamsBuffer } },
                { binding: 4, resource: { buffer: this.camUB } },
                { binding: 5, resource: { buffer: this.tickBuffer } },
            ],
        });
    }

    updateSprites(posX: Int32Array, posY: Int32Array, state: Uint32Array) {
        const device = GDevice.device;
        const queue = device.queue;

        if (posX.length !== posY.length || posX.length !== state.length) {
            throw new Error('Sprite buffer lengths mismatch');
        }
        const size = (this.drawNum = state.length);
        this.expandBuffers();

        if (!this.posXBuffer || !this.posYBuffer || !this.stateBuffer) return;
        queue.writeBuffer(
            this.posXBuffer,
            0,
            posX.buffer,
            posX.byteOffset,
            Int32Array.BYTES_PER_ELEMENT * size,
        );
        queue.writeBuffer(
            this.posYBuffer,
            0,
            posY.buffer,
            posY.byteOffset,
            Int32Array.BYTES_PER_ELEMENT * size,
        );

        queue.writeBuffer(
            this.stateBuffer,
            0,
            state.buffer,
            state.byteOffset,
            Uint32Array.BYTES_PER_ELEMENT * size,
        );
    }

    updateTick(tick: number) {
        if (!this.tickBuffer) return;

        const device = GDevice.device;
        const queue = device.queue;
        queue.writeBuffer(this.tickBuffer, 0, new Uint32Array([tick]));
    }

    resize(width: number, height: number) {}

    allocUniform(): {
        index: number;
        offset: number;
        buffer: GPUBuffer;
        model: Float32Array;
        layout: GPUBindGroupLayout;
    } {
        return null;
    }
    freeUniformIndex() {}

    render(dt: number, now: number, output: GPUTexture, camBuf: Float32Array) {
        if (!this.basePipeline) return;

        const device = GDevice.device;
        const queue = device.queue;

        queue.writeBuffer(
            this.camUB,
            0,
            camBuf.buffer,
            0,
            Float32Array.BYTES_PER_ELEMENT * 16,
        );

        // single-pass forward render to provided output (color + depth)
        const cmd = device.createCommandEncoder();

        const colorAttachment: GPURenderPassColorAttachment = {
            view: output.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
        };

        const passDesc: GPURenderPassDescriptor = {
            colorAttachments: [colorAttachment],
            depthStencilAttachment: {
                view: Renderer.DefaultDepthStencilView,
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
                stencilClearValue: 0,
                stencilLoadOp: 'clear',
                stencilStoreOp: 'store',
            },
        };

        const pass = cmd.beginRenderPass(passDesc);
        pass.setViewport(0, 0, output.width, output.height, 0, 1);
        pass.setPipeline(this.basePipeline);
        pass.setBindGroup(0, this.viewGroup);
        pass.setBindGroup(1, this.samplerGroup);

        if (this.drawNum) pass.draw(6, this.drawNum);

        pass.end();
        queue.submit([cmd.finish()]);
    }
}
