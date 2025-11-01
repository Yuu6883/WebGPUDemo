import Renderer, { checkDevice, GDevice } from './base';

import SpriteVertWGSL from '../shaders/sprite.pbr.vert.wgsl'; // reuse your vertex shader
import SpriteFragWGSL from '../shaders/sprite.pbr.frag.wgsl'; // new forward fragment shader (per-material lighting)
import { RenderPass } from './interfaces';
import SpriteStore from './sprite-store';

export class PBRSpritePass implements RenderPass {
    private readonly store: SpriteStore;

    private basePipeline: GPURenderPipeline;
    private vertModule: GPUShaderModule;
    private fragModule: GPUShaderModule;

    private drawNum = 0;
    private posXBuffer: GPUBuffer;
    private posYBuffer: GPUBuffer;
    private stateBuffer: GPUBuffer;

    private drawGroup: GPUBindGroup;
    public uniformGroup: GPUBindGroup;

    constructor(store: SpriteStore) {
        this.store = store;
    }

    async init() {
        checkDevice();
        if (this.basePipeline) return;
        const device = GDevice.device;

        this.expandBuffers();
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

        this.uniformGroup = device.createBindGroup({
            label: 'Sprite Sampler+Texture+Uniforms',
            layout: this.basePipeline.getBindGroupLayout(1),
            entries: [
                {
                    binding: 0,
                    resource: this.store.spriteTexture.createView(),
                },
                {
                    binding: 1,
                    resource: this.store.spriteSampler,
                },
                { binding: 2, resource: { buffer: this.store.lightBuffer } },
                { binding: 3, resource: { buffer: this.store.spriteParamsBuffer } },
                { binding: 4, resource: { buffer: this.store.camUB } },
                { binding: 5, resource: { buffer: this.store.tickBuffer } },
            ],
        });
    }

    expandBuffers() {
        if (!this.basePipeline) return;
        const device = GDevice.device;

        const bufferSize =
            (Math.floor(this.drawNum / 65536) + 1) * 65536 * Int32Array.BYTES_PER_ELEMENT; // 4 bytes per sprite for i32/u32

        if (bufferSize <= this.posXBuffer?.size) return;
        console.log(
            `Expanding pbr sprite buffers size to ${(bufferSize / 1024 / 1024).toFixed(
                3,
            )}MB`,
        );

        this.posXBuffer?.destroy();
        this.posYBuffer?.destroy();
        this.stateBuffer?.destroy();

        this.posXBuffer = device.createBuffer({
            label: 'PBR Sprite X Positions',
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.posYBuffer = device.createBuffer({
            label: 'PBR Sprite Y Positions',
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.stateBuffer = device.createBuffer({
            label: 'PBR Sprite States',
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // bind group
        this.drawGroup = device.createBindGroup({
            layout: this.basePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.posXBuffer } },
                { binding: 1, resource: { buffer: this.posYBuffer } },
                { binding: 2, resource: { buffer: this.stateBuffer } },
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

    render(dt: number, now: number, output: GPUTexture, clear: boolean) {
        if (!this.basePipeline) return;

        const device = GDevice.device;
        const queue = device.queue;

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
        pass.setBindGroup(0, this.drawGroup);
        pass.setBindGroup(1, this.uniformGroup);

        if (this.drawNum) pass.draw(6, this.drawNum);

        pass.end();
        queue.submit([cmd.finish()]);
    }
}
